import argparse
import os
from math import log10

import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py

import pytorch_ssim
from data_utils import CustomDataset, display_transform
# from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--data_path', type=str, default='../data/OLI2MSI.h5')
parser.add_argument('--crop_size', default=480, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=10, type=int, help='train epoch number')
parser.add_argument('--train_batch_size', default=8, type=int)
parser.add_argument('--test_batch_size', default=1, type=int)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_works', default=0, type=int)
parser.add_argument('--save_freq', default=10, type=int)
opt = parser.parse_args()

if __name__ == '__main__':

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    
    ## 载入数据
    hf = h5py.File(opt.data_path, 'r')
    imgs_train_hr = hf['imgs_train_hr'][:]
    imgs_train_lr = hf['imgs_train_lr'][:]
    imgs_test_hr = hf['imgs_test_hr'][:]
    imgs_test_lr = hf['imgs_test_lr'][:]
    hf.close()
    print(imgs_train_hr.shape)
    print(imgs_train_lr.shape)
    print(imgs_test_hr.shape)
    print(imgs_test_lr.shape)
    
    
    ## 定义data loader
    train_set = CustomDataset(imgs_hr=imgs_train_hr, imgs_lr=imgs_train_lr, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_works, batch_size=opt.train_batch_size, shuffle=True, pin_memory=True)
    val_set = CustomDataset(imgs_hr=imgs_test_hr, imgs_lr=imgs_test_lr, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.num_works, batch_size=opt.test_batch_size, shuffle=False, pin_memory=True)
    
    ## 准备生成器和判别器
    netG = Generator(UPSCALE_FACTOR).cuda()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator().cuda()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    generator_criterion = GeneratorLoss().cuda()
    
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
    
    ## 实例化优化器
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    ## 开始训练
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            real_img = target
            real_img = real_img.float().cuda()
            z = data
            z = z.float().cuda()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            ############################
            # (2) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img.detach()).mean()
            d_loss = 1 - real_out + fake_out

            optimizerD.zero_grad()
            d_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerD.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                lr = lr.float().cuda()
                hr = hr.float().cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
                
                # print(val_lr.cpu().squeeze(0).shape)
                # print(val_hr.cpu().squeeze(0).shape)
                # print(sr.cpu().squeeze(0).shape)
                
                val_images.extend(
                    [display_transform()(val_lr.cpu().squeeze(0)),  #LR图像，image grid第一列
                     display_transform()(val_hr.cpu().squeeze(0)),  #HR图像，image grid第二列
                     display_transform()(sr.cpu().squeeze(0))]      #SR图像，image grid第三列
                    )
                
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    
        # save model parameters
        if (epoch+1)%opt.save_freq==0:
            torch.save(netG.state_dict(), 'epochs/netG_%d_epoch_%d.pth' % (UPSCALE_FACTOR, epoch))
            torch.save(netD.state_dict(), 'epochs/netD_%d_epoch_%d.pth' % (UPSCALE_FACTOR, epoch))
            
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
    
    ##end for