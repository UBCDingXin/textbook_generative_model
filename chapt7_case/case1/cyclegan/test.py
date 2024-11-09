#!/usr/bin/python3

import argparse
import sys
import os

from torch import nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../data/monet2photo/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='/netG_B2A.pth', help='B2A generator checkpoint file')

parser.add_argument('--lambda1', type=float, default=5.0)
parser.add_argument('--lambda2', type=float, default=10.0)

opt = parser.parse_args()
print(opt)

output_path = "./output/lam1_{}_lam2_{}".format(opt.lambda1, opt.lambda2)
os.makedirs(output_path, exist_ok=True)
ckpts_path = output_path + "/ckpts"
os.makedirs(ckpts_path, exist_ok=True)

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

netG_A2B.cuda()
netG_B2A.cuda()

netG_A2B = nn.DataParallel(netG_A2B)
netG_B2A = nn.DataParallel(netG_B2A)

# Load state dicts
netG_A2B.load_state_dict(torch.load(output_path + opt.generator_A2B, weights_only=True))
netG_B2A.load_state_dict(torch.load(output_path + opt.generator_B2A, weights_only=True))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
input_A = torch.tensor((opt.batchSize, opt.input_nc, opt.size, opt.size), dtype=torch.float).cuda()
input_B = torch.tensor((opt.batchSize, opt.output_nc, opt.size, opt.size), dtype=torch.float).cuda()


# Dataset loader
transforms_ = [ transforms.Resize(opt.size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
test_sample_A2B_path = output_path + "/test_samples/A2B"
test_sample_B2A_path = output_path + "/test_samples/B2A"
os.makedirs(test_sample_A2B_path, exist_ok=True)    
os.makedirs(test_sample_B2A_path, exist_ok=True)  

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = batch['A'].type(torch.float).cuda()
    real_B = batch['B'].type(torch.float).cuda()

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).cpu().data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).cpu().data + 1.0)
    
    real_A = (0.5*(real_A+1.0)).cpu().data
    real_B = (0.5*(real_B+1.0)).cpu().data

    # Save image files    
    vis_A2B = torch.cat([real_A, fake_B], dim=0)
    vis_B2A = torch.cat([real_B, fake_A], dim=0)
    save_image(vis_A2B, test_sample_A2B_path + '/%04d.png' % (i+1))
    save_image(vis_B2A, test_sample_B2A_path + '/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
