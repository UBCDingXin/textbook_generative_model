#!/usr/bin/python3

import argparse
import itertools
import os

from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision.utils import save_image

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../data/CT2MRI/images/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--show_freq', type=int, default=5)
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')

parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=10.0)

opt = parser.parse_args()
print(opt)


output_path = "./output/lam1_{}_lam2_{}".format(opt.lambda1, opt.lambda2)
os.makedirs(output_path, exist_ok=True)

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

netG_A2B.cuda()
netG_B2A.cuda()
netD_A.cuda()
netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

netG_A2B = nn.DataParallel(netG_A2B)
netG_B2A = nn.DataParallel(netG_B2A)
netD_A = nn.DataParallel(netD_A)
netD_B = nn.DataParallel(netD_B)


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
target_real = torch.ones(opt.batchSize, dtype=torch.float, requires_grad=False).cuda()
target_fake = torch.zeros(opt.batchSize, dtype=torch.float, requires_grad=False).cuda()

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ v2.ToImage(),
                v2.RandomRotation(degrees=(0, 30)),
                v2.RandomResizedCrop((opt.size, opt.size), scale=(0.8, 1.2)), 
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)


# 从A域和B域分别采样nrow张图像用于训练中的测试
nrow = 10
vis_data = next(iter(dataloader))  # 获取第一个批次
vis_A = vis_data['A'][0:min(nrow, opt.batchSize)]
vis_B = vis_data['B'][0:min(nrow, opt.batchSize)]
# print(vis_A.shape)
# print(vis_B.shape)
# print(vis_A.min(), vis_A.max())
# print(vis_B.min(), vis_B.max())

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    netG_A2B.train()
    netG_B2A.train()
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch['A'].type(torch.float).cuda()
        real_B = batch['B'].type(torch.float).cuda()
        
        # skip the last batch
        if (i+1)==len(dataloader):
            continue
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*opt.lambda1
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*opt.lambda1

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake.view(-1), target_real.view(-1))

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake.view(-1), target_real.view(-1))

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*opt.lambda2

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*opt.lambda2

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real.view(-1), target_real.view(-1))

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake.view(-1), target_fake.view(-1))

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real.view(-1), target_real.view(-1))
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake.view(-1), target_fake.view(-1))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        

    # 每隔一定轮数输出图像
    if (epoch+1) % opt.show_freq==0:
        netG_A2B.eval()
        netG_B2A.eval()
        with torch.inference_mode():
            ## A to B
            fake_B = netG_A2B(vis_A.cuda()).cpu()
            vis_images_AB = torch.cat([vis_A, fake_B],dim=0)
            save_file_AB = output_path + '/samples/A2B_epoch{}.png'.format(epoch+1)
            os.makedirs(os.path.dirname(save_file_AB), exist_ok=True)
            save_image(vis_images_AB.data, save_file_AB, nrow=nrow, normalize=True)
            ## B to A
            fake_A = netG_B2A(vis_B.cuda()).cpu()
            vis_images_BA = torch.cat([vis_B, fake_A],dim=0)
            save_file_BA = output_path + '/samples/B2A_epoch{}.png'.format(epoch+1)
            os.makedirs(os.path.dirname(save_file_BA), exist_ok=True)
            save_image(vis_images_BA.data, save_file_BA, nrow=nrow, normalize=True)
    ##end if
            

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), output_path + '/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), output_path + '/netG_B2A.pth')
    torch.save(netD_A.state_dict(), output_path + '/netD_A.pth')
    torch.save(netD_B.state_dict(), output_path + '/netD_B.pth')
###################################
