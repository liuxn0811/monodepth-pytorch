import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import itertools
# develop branch
# custom modules
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='5,6'
from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader

# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)
import multiprocessing
multiprocessing.set_start_method('spawn', True)

def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('--data_dir',default='/media/public/disk2/liu/kitti_raw_data/train',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images'
                        )
    parser.add_argument('--val_data_dir',default='/media/public/disk2/liu/kitti_raw_data/val',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images'
                        )
    parser.add_argument('--model_path',default='./models1/all/forward.pth', help='path to the trained model')
    parser.add_argument('--Bmodel_path',default='./models1/all/backward.pth', help='path to the trained model')
    parser.add_argument('--output_directory',default='./data/raw_output1',
                        help='where save dispairities\
                        for tested images'
                        )
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='resnet18_md',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--Bmodel', default='resnet18_md',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--Bpretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epoch1', default=20,
                        help='number of total epochs to run')
    parser.add_argument('--epoch2', default=15,
                        help='number of total epochs to run')
    parser.add_argument('--epoch3', default=20,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=8,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
        ],
            help='lowest and highest values for gamma,\
                        brightness and color respectively'
            )
    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=True)
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Model:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, input_channels=args.input_channels, pretrained=args.pretrained)
        self.backward = get_model(args.Bmodel, input_channels=args.input_channels, pretrained=args.Bpretrained)
        self.model = self.model.to(self.device)
        self.backward = self.backward.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
            self.backward = torch.nn.DataParallel(self.backward)

        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            self.Boptimizer = optim.Adam(self.backward.parameters(),
                                        lr=args.learning_rate)
            self.Goptimizer = optim.Adam (itertools.chain(self.model.parameters(), self.backward.parameters()),lr=args.learning_rate)
            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, args.mode,
                                                                 args.augment_parameters,
                                                                 False, args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)
        else:
            self.model.load_state_dict(torch.load(args.model_path))
            #self.backward.load_state_dict(torch.load(args.Bmodel_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width

        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode, args.augment_parameters,
                                                     args.do_augmentation, args.batch_size,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers)


        if 'cuda' in self.device:
            torch.cuda.synchronize()

    
    def freeze_model(self,model):
        model.eval()
        for params in model.parameters():
            params.requires_grad = False
    
    def unfreeze_model(self,model):
        model.train()
        for params in model.parameters():
            params.requires_grad = True
    
    def img_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs
    
    def apply_disp(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        pyramid = self.img_pyramid(img, 4)
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in disp]
        left_est = [self.apply_disp(pyramid[i],-disp_left_est[i]) for i in range(4)]
        return left_est

    def generate_image_right(self, img, disp):
        pyramid = self.img_pyramid(img, 4)
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in disp]
        right_est = [self.apply_disp(pyramid[i],disp_right_est[i]) for i in range(4)]
        return right_est
    
    def train(self):
        losses = []
        val_losses = []
        best_loss = float('Inf')
        best_val_loss = float('Inf')

        running_val_loss = 0.0
        self.freeze_model(self.model)
        #self.freeze_model(self.backward)
        for data in self.val_loader:
            
            data = to_device(data, self.device)
            left = data['left_image']
            right = data['right_image']
            disps = self.model(left)
            #righti = self.generate_image_right(left,disps)
            #disps1 = self.backward(righti[0])
            loss = self.loss_function(disps, disps, [left, right],'forward')
            val_losses.append(loss.item())
            running_val_loss += loss.item()

        running_val_loss /= self.val_n_img / self.args.batch_size
        print('Val_loss:', running_val_loss)

        for epoch in range(self.args.epoch1):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.unfreeze_model(self.model)
            for data in self.loader:
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']

                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model(left)
                loss = self.loss_function(disps,disps, [left, right],'forward')
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                running_loss += loss.item()

            running_val_loss = 0.0
            self.freeze_model(self.model)
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                disps = self.model(left)
                loss = self.loss_function(disps, disps, [left, right], 'forward')
                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print (
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
            self.save('./models1/forward/' + str(epoch+1)+'.pth')
            if running_val_loss < best_val_loss:
                self.save('./models1/forward/'+ 'cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')
        
        for epoch in range(self.args.epoch2):
            if self.args.adjust_lr:
                adjust_learning_rate(self.Boptimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.unfreeze_model(self.backward)
            for data in self.loader:
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']

                # One optimization iteration
                self.Boptimizer.zero_grad()
                disps = self.model(left)
                righti = self.generate_image_right(left,disps)
                disps1 = self.backward(righti[0])
                loss = self.loss_function(disps,disps1, [left, right],'backward')
                loss.backward()
                self.Boptimizer.step()
                losses.append(loss.item())
                running_loss += loss.item()

            running_val_loss = 0.0
            self.freeze_model(self.backward)
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                disps = self.model(left)
                righti = self.generate_image_right(left,disps)
                disps1 = self.backward(righti[0])
                loss = self.loss_function(disps,disps1, [left, right],'backward')
                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print (
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
            self.save('./models1/backward/' + str(epoch+1)+'.pth')
            if running_val_loss < best_val_loss:
                self.save('./models1/backward/' + 'cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')
        
        for epoch in range(self.args.epoch3):
            if self.args.adjust_lr:
                adjust_learning_rate(self.Goptimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.unfreeze_model(self.model)
            self.unfreeze_model(self.backward)
            for data in self.loader:
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']

                # One optimization iteration
                self.Goptimizer.zero_grad()
                disps = self.model(left)
                righti = self.generate_image_right(left,disps)
                disps1 = self.backward(righti[0])
                loss = self.loss_function(disps,disps1, [left, right],'all')
                loss.backward()
                self.Goptimizer.step()
                losses.append(loss.item())
                running_loss += loss.item()

            running_val_loss = 0.0
            self.freeze_model(self.model)
            self.freeze_model(self.backward)
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                disps = self.model(left)
                righti = self.generate_image_right(left,disps)
                disps1 = self.backward(righti[0])
                loss = self.loss_function(disps,disps1, [left, right],'all')
                
                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print (
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
            self.save('./models1/all/' + str(epoch+1)+'_forward.pth')
            self.Bsave('./models1/all/' + str(epoch+1)+'_backward.pth')
            if running_val_loss < best_val_loss:
                self.save(self.args.model_path[:-4] + '_cpt.pth')
                self.Bsave(self.args.Bmodel_path[:-4] + '_cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')
        print ('Finished Training. Best loss:', best_loss)
        self.save(self.args.model_path)
        self.Bsave(self.args.Bmodel_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def Bsave(self, path):
        torch.save(self.backward.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def test(self):
        self.model.eval()
        disparities = np.zeros((self.n_img,
                               self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.n_img,
                                  self.input_height, self.input_width),
                                  dtype=np.float32)
        with torch.no_grad():
            for (i, data) in enumerate(self.loader):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                disps = self.model(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] = \
                    post_process_disparity(disps[0][:, 0, :, :]\
                                           .cpu().numpy())

        np.save(self.output_directory + '/disparities.npy', disparities)
        np.save(self.output_directory + '/disparities_pp.npy',
                disparities_pp)
        print('Finished Testing')


def main():
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    main()

