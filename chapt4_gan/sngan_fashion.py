''' 在Fashion MNIST训练一个SNGAN '''

## 载入必要的包
import torch
import torch.utils.data
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
from torch.nn.utils import spectral_norm
import random

import os
import timeit

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from DiffAugment_pytorch import DiffAugment
from inception_v3 import Inception3
from eval_metrics import cal_FID, compute_IS


# seeds
seed=2024
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(seed)

###################################################
## 参数设定
IMG_SIZE=28
NUM_CLASS=10
NC=1
NFAKE=10000

DIM_Z=128
GEN_SIZE=64
DISC_SIZE=64

EPOCHS=200
RESUME_EPOCH=0
BATCH_SIZE=256
LR_D=1e-4
LR_G=1e-4
NUM_ITER_D=2 #每个循环判别器更新几次
PRINT_FREQ=1
SAVE_FREQ=50

policy = "color,translation,cutout"


## 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 输出路径
output_path = "./output/sngan"
os.makedirs(output_path, exist_ok=True)


###################################################
# 数据载入与预处理
transform = transforms.Compose([
    # transforms.Resize([IMG_SIZE,IMG_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


###################################################
# 模型定义

class ResBlockGenerator(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            # nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            # nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)



class generator(nn.Module):
    def __init__(self, dim_z):
        super(generator, self).__init__()
        self.dim_z = dim_z
        self.dense = nn.Linear(self.dim_z, 7 * 7 * (GEN_SIZE*4))
        self.final = nn.Conv2d(GEN_SIZE, NC, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator((GEN_SIZE*4), (GEN_SIZE*2)), #7->14
            ResBlockGenerator((GEN_SIZE*2), (GEN_SIZE)), #14->28
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        z = z.view(-1, self.dim_z)
        out = self.dense(z)
        out = out.view(-1, (GEN_SIZE*4), 7, 7)
        out = self.model(out)
        return out


class discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(discriminator, self).__init__()
        self.ngpu = ngpu

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(NC, DISC_SIZE, stride=1), #28-->28
                ResBlockDiscriminator(DISC_SIZE  , DISC_SIZE*2, stride=2), #28--->14
                ResBlockDiscriminator(DISC_SIZE*2, DISC_SIZE*4, stride=2), #14--->7
            )
        self.snconv = nn.Conv2d(DISC_SIZE*4, DISC_SIZE*4, kernel_size=4, stride=1, padding=0)
        self.snconv = spectral_norm(self.snconv)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(DISC_SIZE*4, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = spectral_norm(self.fc)

    def forward(self, x):
        features = self.model(x)
        features = self.relu(self.snconv(features))
        features = torch.sum(features, dim=(2, 3)) # Global pooling
        out = self.fc(features)
        return out


# if __name__=="__main__":
#     def get_parameter_number(net):
#         total_num = sum(p.numel() for p in net.parameters())
#         trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
#         return {'Total': total_num, 'Trainable': trainable_num}
    
#     netG = generator(z_dim=128).cuda()
#     netD = discriminator().cuda()

#     z = torch.randn(5, DIM_Z).cuda()
#     x = netG(z)
#     print(x.size())
#     o = netD(x)
#     print(o.size())
    
#     print('G:', get_parameter_number(netG))
#     print('D:', get_parameter_number(netD))


###################################################
# 实例化生成器和判别器
netG = generator(dim_z=DIM_Z).to(device)
netD = discriminator().to(device)


###################################################
# 训练函数的定义
def train(netG, netD, resume_epoch=0, device="cuda"):

    ## 将生成器和判别器网络移动到指定的设备上
    netG = netG.to(device)
    netD = netD.to(device)

    ## 分别定义生成器和判别器的优化器
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

    if resume_epoch>0:
        save_file = output_path + "/ckpts_in_train/ckpt_epoch_{}.pth".format(resume_epoch)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    #end if

    ## 生成10个固定的噪声向量，用于训练中的样本可视化
    n_row=10
    z_fixed = torch.randn(n_row**2, DIM_Z, dtype=torch.float).to(device)

    d_loss_all = []
    g_loss_all = []

    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, EPOCHS):

        # 将生成器和判别器设置为训练模式
        netG.train()
        netD.train()

        # 将train_loader转换为迭代器
        data_iter = iter(train_loader)
        batch_idx = 0

        d_loss_tmp = 0
        g_loss_tmp = 0
        while (batch_idx < len(train_loader)):
            ############################
            # (1) 更新判别器 D
            ############################
            ## 每个循环中，每更新一次生成器，更新NUM_ITER_D次判别器

            for indx_d in range(NUM_ITER_D):

                if batch_idx == len(train_loader):
                    break

                ## 采样一批真实图像
                (batch_train_images,_) = next(data_iter)
                batch_idx += 1

                # 计算当前batch的样本量
                batch_size_current = batch_train_images.shape[0]
                # 将batch中的图像转换成float类型并移动至指定设备上
                batch_train_images = batch_train_images.type(torch.float).to(device)

                # 采样高斯噪声
                z = torch.randn(batch_size_current, DIM_Z, dtype=torch.float).to(device)
                # 生成一批虚假图像
                gen_imgs = netG(z)

                # 清除旧的梯度
                optimizerD.zero_grad()

                # 计算判别器的损失函数
                d_out_real = netD(DiffAugment(batch_train_images, policy=policy))
                d_out_fake = netD(DiffAugment(gen_imgs.detach(), policy=policy))
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                d_loss = d_loss_real + d_loss_fake
                # 梯度反向传播
                d_loss.backward()
                # 优化器参数更新
                optimizerD.step()

                d_loss_tmp += d_loss.item()
            ##end for _

            ############################
            # (2) 更新生成器 G
            ############################
            # 清除旧的梯度
            optimizerG.zero_grad()
            # 采样高斯噪声
            z = torch.randn(batch_size_current, DIM_Z, dtype=torch.float).to(device)
            # 生成一批虚假图像
            gen_imgs = netG(z)
            # 判别器的输出
            g_out_fake = netD(DiffAugment(gen_imgs, policy=policy))
            # 计算生成器的损失函数，
            g_loss = - g_out_fake.mean()
            # 梯度反向传播
            g_loss.backward()
            # 优化器参数更新
            optimizerG.step()
            g_loss_tmp+=g_loss.item()
        ##end while

        d_loss_all.append(d_loss_tmp/((indx_d+1)*(batch_idx+1)*batch_size_current))
        g_loss_all.append(g_loss_tmp/((batch_idx+1)*batch_size_current))

        print ("\r SNGAN: [Epoch %d/%d] [D loss: %.3f] [G loss: %.3f] [Time: %.3f]" % (epoch+1, EPOCHS, d_loss.item(), g_loss.item(), timeit.default_timer()-start_time))

        # 每PRINT_FREQ个epoch生成100个样本用于可视化
        if (epoch+1)%PRINT_FREQ==0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed)
                gen_imgs = gen_imgs.detach()
                save_file = output_path + "/{}.png".format(epoch+1)
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                save_image(gen_imgs.data, save_file, nrow=10, normalize=True)

        if (epoch+1)%SAVE_FREQ==0:
            save_file = output_path + "/ckpts_in_train/ckpt_epoch_{}.pth".format(epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'netG_state_dict': netG.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch
    return netG, np.array(d_loss_all), np.array(g_loss_all)

###################################################
# 训练
netG, d_loss_all, g_loss_all = train(netG, netD, resume_epoch=RESUME_EPOCH, device=device)



###################################################
# 采样并可视化

# 定义采样函数
def sample(netG, nfake, batch_size=100):
    '''
    输入：
    预训练的生成器netG
    采样的数量nfake
    采样的批处理大小batch_size
    '''
    netG=netG.to(device)
    netG.eval()
    output = []
    with torch.no_grad():
        num_got = 0
        while num_got < nfake:
            z = torch.randn(batch_size, DIM_Z, dtype=torch.float).to(device)
            batch_fake_images = netG(z)
            output.append(batch_fake_images.cpu().detach().numpy())
            num_got += batch_size
    output = np.concatenate(output, axis=0)
    return output[0:nfake] #(n,c,h,w)

sample_images = sample(netG, nfake=25, batch_size=25)
sample_images = [np.transpose(sample_images[i],axes=[1,2,0])*0.5+0.5 for i in range(len(sample_images))]
fig = plt.figure(figsize=(5, 5))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

for ax, im in zip(grid, sample_images):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
plt.savefig(output_path + "/fake_imgs_epoch_{}.png".format(EPOCHS), dpi=500,bbox_inches="tight")





#########################################
# 评价指标计算

PreNetFIDIS = Inception3(num_classes=NUM_CLASS, aux_logits=True, transform_input=False, input_channel=NC).to(device)
# PreNetFIDIS = nn.DataParallel(PreNetFIDIS)
filename_ckpt = './output/dcgan_fashion_mnist/ckpt_InceptionV3_epoch200.pth'
checkpoint_PreNet = torch.load(filename_ckpt, weights_only=True)
PreNetFIDIS.load_state_dict(checkpoint_PreNet['net_state_dict'])


## 生成虚假样本
start_time = timeit.default_timer()
fake_images = sample(netG, nfake=NFAKE, batch_size=500)
fake_images = ((fake_images*0.5+0.5)*255.0).astype(int)
print("Fake image range:", len(fake_images), fake_images.min(), fake_images.max())
print(fake_images.shape)
print("采样速度：{}（张/秒）".format(NFAKE/(timeit.default_timer()-start_time)))

## 准备测试样本用于计算FID
real_images = train_dataset.train_data[:,np.newaxis,:,:].numpy()
print("Real image range:", len(real_images), real_images.min(), real_images.max())
print(real_images.shape)


## 计算IS
indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
IS, IS_std = compute_IS(PreNetFIDIS, fake_images[indx_shuffle_fake], batch_size = 500, splits=10, resize = (299,299), verbose=False, normalize_img = True)
print("\n IS of {} fake images: {:.3f}({:.3f}).".format(NFAKE, IS, IS_std))

## 计算FID
indx_shuffle_real = np.arange(len(real_images)); np.random.shuffle(indx_shuffle_real)
indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
FID_score = cal_FID(PreNetFIDIS, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=200, resize = (299,299), norm_img = True)
print("\n FID of {} fake images: {:.3f}.".format(NFAKE, FID_score))
