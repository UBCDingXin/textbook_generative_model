''' 在Fashion MNIST上训练一个DCGAN '''

## 载入必要的包
import torch
import torch.utils.data
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
from scipy.ndimage import zoom
import random

import os
import timeit

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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

EPOCHS=100
RESUME_EPOCH=0
BATCH_SIZE=256
DIM_Z=128
LR_D=1e-4
LR_G=1e-4
NUM_ITER_D=2 #每个循环判别器更新几次
PRINT_FREQ=1
SAVE_FREQ=50


## 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 输出路径
output_path = "./output/dcgan"
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

# 定义生成器类
class generator(nn.Module):
    def __init__(self, dim_z=128, out_channels=1):
        super(generator, self).__init__()
        self.dim_z = dim_z
        self.out_channels = out_channels

        # 卷积模块
        self.conv = nn.Sequential(
            # 输入维度 (n,128,1,1)
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0),#h=h+3
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 特征图维度 (n,512,4,4)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0),#h=h+3
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 特征图维度 (n,256,7,7)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),#h=h*2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 特征图维度 (n,128,14,14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),#h=h*2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 特征图维度 (n,64,28,28)
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1), #h=h
            nn.Tanh()
            # 输出维度 (n,1,28,28)
        )
    def forward(self, input):
        input = input.view(-1, self.dim_z, 1, 1)
        output = self.conv(input)
        return output


# 定义判别器类
class discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(discriminator, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            # 输入维度：(n,1,28,28)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), #h=h/2
            nn.LeakyReLU(0.2, inplace=True),
            # 输入维度：(n,64,14,14)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #h=h/2
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图维度：(n,128,7,7)
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0), #h=h-3
            nn.LeakyReLU(0.2, inplace=True),
            # 输入维度：(n,256,4,4)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0), #h=h-3
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图维度：(n,512,1,1)
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), #h=h
            nn.Sigmoid()
            # 特征图维度：(n,1,1,1)
        )
    def forward(self, input):
        output = self.conv(input)
        return output.view(-1, 1)


###################################################
# 实例化生成器和判别器
netG = generator(dim_z=DIM_Z, out_channels=1).to(device)
netD = discriminator(in_channels=1).to(device)


###################################################
# 训练函数的定义
def train(netG, netD, resume_epoch=0, device="cuda"):

    ## 将生成器和判别器网络移动到指定的设备上
    netG = netG.to(device)
    netD = netD.to(device)

    ## 分别定义生成器和判别器的优化器
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

    ## 判别器的分类损失
    criterion = nn.BCELoss()

    if resume_epoch>0:
        save_file = output_path + "/ckpts_in_train/ckpt_niter_{}.pth".format(resume_epoch)
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

                # 样本的标签：真实为1，虚假为0;用于定义损失函数
                real_gt = torch.ones(batch_size_current,1).to(device)
                fake_gt = torch.zeros(batch_size_current,1).to(device)

                # 清除旧的梯度
                optimizerD.zero_grad()

                # 计算判别器的损失函数
                prob_real = netD(batch_train_images)
                prob_fake = netD(gen_imgs.detach())
                real_loss = criterion(prob_real, real_gt)
                fake_loss = criterion(prob_fake, fake_gt)
                d_loss = (real_loss + fake_loss) / 2
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
            dis_out = netD(gen_imgs)
            # 计算生成器的损失函数，即公式(4-22)
            g_loss = criterion(dis_out, real_gt)
            # 梯度反向传播
            g_loss.backward()
            # 优化器参数更新
            optimizerG.step()
            g_loss_tmp+=g_loss.item()
        ##end while

        d_loss_all.append(d_loss_tmp/((indx_d+1)*(batch_idx+1)*batch_size_current))
        g_loss_all.append(g_loss_tmp/((batch_idx+1)*batch_size_current))

        print ("\r DCGAN: [Epoch %d/%d] [D loss: %.3f] [G loss: %.3f] [D prob real:%.3f] [D prob fake:%.3f] [Time: %.3f]" % (epoch+1, EPOCHS, d_loss.item(), g_loss.item(), prob_real.mean().item(), prob_fake.mean().item(), timeit.default_timer()-start_time))

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
            save_file = output_path + "/ckpts_in_train/ckpt_niter_{}.pth".format(epoch+1)
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


# ########################
# ## 可视化loss
# np.save(output_path + '/d_loss_vs_epoch{}.npy'.format(EPOCHS), d_loss_all)
# np.save(output_path + '/g_loss_vs_epoch{}.npy'.format(EPOCHS), g_loss_all)
# # d_loss_all = numpy.load('d_loss_vs_epoch{}.npy'.format(EPOCHS))
# # g_loss_all = numpy.load('g_loss_vs_epoch{}.npy'.format(EPOCHS))
# # 绘制折线图
# fig = plt.figure()
# plt.plot(np.arange(EPOCHS), d_loss_all, label="D", color="blue", marker="o", markersize=2)
# plt.plot(np.arange(EPOCHS), g_loss_all, label="G", color="red", marker="*", markersize=2) #linestyle='--',
# plt.grid(True, linestyle='--')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig(output_path + "/training_loss_vs_epoch{}.pdf".format(EPOCHS), dpi=500, bbox_inches="tight")




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
