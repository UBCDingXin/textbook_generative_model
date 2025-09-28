''' 在Fashion MNIST上训练一个VAE '''

## 载入必要的包
import torch  
import torch.utils.data  
from torch import nn, optim  
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms  
from torch.nn import functional as F
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import timeit
from scipy.ndimage import zoom
import random

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
## 超参数设定
IMG_SIZE=28
NUM_CLASS=10
NC=1

EPOCHS=200
RESUME_EPOCH=0
BATCH_SIZE=256
LATENT_DIM=64
LR=1e-4
WEIGHT_DECAY=1e-4
ALPHA=1e-4

NFAKE=10000 #生成数据用于评价


## 设定设备  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)

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



# 输出路径
output_path = "./output"
os.makedirs(output_path, exist_ok=True)

###################################################
# 输出25张图像用于可视化
dataiter = iter(train_loader)
image = next(dataiter)

num_samples = 25
sample_images = [((image[0][i,0]*0.5+0.5)*255.0).int() for i in range(num_samples)] 

fig = plt.figure(figsize=(5, 5))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

for ax, im in zip(grid, sample_images):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
plt.savefig(output_path + "/example_imgs.png", dpi=500, bbox_inches="tight")


###################################################
# VAE类的定义
class VAE(nn.Module):
    # 初始化方法
    def __init__(self, img_size, latent_dim): 
        # 继承初始化方法
        super(VAE, self).__init__()  
        # 输入图片的通道数、高度、宽度
        self.in_channel, self.img_h, self.img_w = img_size 
        # 隐藏编码Z的维度
        self.latent_dim = latent_dim  
        
        ## 开始构建推断网络（又称编码器Encoder）
        # 卷积层
        self.encoder = nn.Sequential(
            ## 输入维度 (n,1,28,28)
            nn.Conv2d(in_channels=self.in_channel, out_channels=32, kernel_size=3, stride=2, padding=1),#h=h//2 14
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(),
            ## 特征图维度 (n,32,14,14)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),#h=h//2 7
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(),
            ## 特征图维度 (n,64,7,7)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0),#h=h-3 4
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(),
            ## 特征图维度 (n,128,4,4)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),#h=h//2 2
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(),
            ## 特征图维度 (n,256,2,2)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0),#h=h 2
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(),
            ## 特征图维度 (n,512,2,2)
        )
        # 全连接层：将特征向量转化为分布均值mu
        self.fc_mu = nn.Linear(512*4, self.latent_dim) #
        # 全连接层：将特征向量转化为分布方差的对数log(var)
        self.fc_var = nn.Linear(512*4, self.latent_dim) 
        
        ## 开始构建生成网络（又称解码器Decoder）
        # 全连接层
        self.decoder_input = nn.Linear(self.latent_dim, 512*4)  
        # 转置卷积层
        self.decoder = nn.Sequential(
            ## 输入维度 (n,512,2,2)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),#h=2h
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            ## 特征图维度 (n,256,4,4)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0, output_padding=0),#h=h+3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            ## 特征图维度 (n,128,7,7)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),#h=2h
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            ## 特征图维度 (n,64,14,14)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),#h=2h
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            ## 特征图维度 (n,32,28,28)
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, output_padding=0),#h=h
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            ## 特征图维度 (n,32,28,28)
            nn.Conv2d(in_channels=32, out_channels=self.in_channel, kernel_size=1, stride=1, padding=0), #h=h
            nn.Tanh()
            ## 输出图像维度 (n,1,28,28)
        )
 
    # 定义编码过程
    def encode(self, x):  
        result = self.encoder(x)  # Encoder结构,(n,1,28,28)-->(n,512,2,2)
        result = torch.flatten(result, 1)  # 将特征层转化为特征向量,(n,512,2,2)-->(n,512*4)
        mu = self.fc_mu(result)  # 计算分布均值mu,(n,512*4)-->(n,128)
        log_var = self.fc_var(result)  # 计算分布方差的对数log(var),(n,512*4)-->(n,128)
        return [mu, log_var]  # 返回分布的均值和方差对数
 
    # 定义解码过程
    def decode(self, z):  
        x_hat = self.decoder_input(z).view(-1, 512, 2, 2)  # 将采样变量Z转化为特征向量，再转化为特征层,(n,128)-->(n,512*4)-->(n,512,2,2)
        x_hat = self.decoder(x_hat)  # decoder结构,(n,512,2,2)-->(n,1,28,28)
        return x_hat  # 返回生成样本
 
    # 重参数化
    def reparameterize(self, mu, log_var): 
        std = torch.exp(log_var)  # 分布标准差std
        eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
        return mu + eps * std  # 返回对应正态分布中的采样值
 
    # 前传函数
    def forward(self, x):  
        mu, log_var = self.encode(x)  # 经过编码过程，得到分布的均值mu和方差对数log_var
        z = self.reparameterize(mu, log_var)  # 经过重参数技巧，得到分布采样变量Z
        x_hat = self.decode(z)  # 经过解码过程，得到生成样本Y
        return [x_hat, x, mu, log_var]  # 返回生成样本Y，输入样本X，分布均值mu，分布方差对数log_var
 
    # 定义生成过程
    def sample(self, n, device):  
        z = torch.randn(n, self.latent_dim).to(device)  # 从标准正态分布中采样得到n个采样变量Z，长度为latent_dim
        images = self.decode(z)  # 经过解码过程，得到生成样本Y
        return images
    

#########################################
# 模型训练

# 实例化模型
model = VAE(img_size=[1,IMG_SIZE,IMG_SIZE], latent_dim=LATENT_DIM).to(device)  
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) 

# 定义损失函数
def loss_function(x, x_hat, mean, log_var, alpha=ALPHA):
    reproduction_loss = F.mse_loss(x_hat, x)
    term1 = 0.5*torch.mean(torch.sum(torch.pow(log_var.exp(), 2), dim=1))
    term2 = 0.5*torch.mean(torch.sum(torch.pow(mean, 2), dim=1))
    term3 = -0.5*torch.mean(2*torch.sum(log_var,dim=1))
    # 常数d可被忽略，不影响优化
    KLD = term1 + term2 + term3
    return reproduction_loss + alpha * KLD

# 定义训练函数  
def train(model, optimizer, epochs, resume_epoch, device):
    
    if resume_epoch>0:
        save_file = output_path + "/ckpts_in_train/ckpt_niter_{}.pth".format(resume_epoch)
        checkpoint = torch.load(save_file)
        model.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    # 用于存储每个epoch的平均损失
    loss_over_epochs = []

    for epoch in range(resume_epoch, epochs):
        model.train()
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            batch_size = x.size(0)
            optimizer.zero_grad()
            x_hat, _, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        ##end for   

        ## 存储每个epoch的平均loss
        loss_over_epochs.append(overall_loss/(batch_idx*batch_size))

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))

        # 每10个epoch生成100个样本用于可视化
        if (epoch+1)%10==0:
            model.eval()
            with torch.no_grad():
                sample_images = model.sample(100, device)  # 获得100个生成样本
                sample_images = sample_images.detach().cpu()
                save_file = output_path + "/{}.png".format(epoch+1)
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                save_image(sample_images.data, save_file, nrow=10, normalize=True)
    
        # 每50个epoch将模型参数存储一次
        if (epoch+1)%50==0:
            save_file = output_path + "/ckpts_in_train/ckpt_niter_{}.pth".format(epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'net_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
            
    ##end for epoch
    
    return np.array(loss_over_epochs)

# 运行训练  
loss_over_epochs = train(model, optimizer, epochs=EPOCHS, resume_epoch=RESUME_EPOCH, device=device)


#########################################
# 采样以及可视化
model.eval()
sample_images = model.sample(25, device)  
sample_images = sample_images.detach().cpu()

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
filename_ckpt = output_path + '/ckpt_InceptionV3_epoch200.pth'
checkpoint_PreNet = torch.load(filename_ckpt, weights_only=True)
PreNetFIDIS.load_state_dict(checkpoint_PreNet['net_state_dict'])


## 生成虚假样本
start_time = timeit.default_timer()
fake_images = []
ngot=0
batch_size=10000
while ngot<NFAKE:
    fake_images_tmp = model.sample(batch_size, device)
    fake_images_tmp = fake_images_tmp.detach().cpu().numpy()
    fake_images_tmp = ((fake_images_tmp*0.5+0.5)*255.0).astype(int)
    fake_images.append(fake_images_tmp)
    ngot+=len(fake_images_tmp)
fake_images = np.concatenate(fake_images,axis=0)
fake_images = fake_images[0:NFAKE]
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




########
# 计算测试集的IS和FID作为参照
real_images2 = test_dataset.test_data[:,np.newaxis,:,:].numpy()
zoom_factors = (1, 1, IMG_SIZE/28, IMG_SIZE/28)
real_images2 = zoom(real_images2, zoom_factors, order=3)
print("{} Real image 2 range:", len(real_images2), real_images2.min(), real_images2.max())
print(real_images2.shape)

## 计算IS
indx_shuffle_test = np.arange(len(real_images2)); np.random.shuffle(indx_shuffle_test)
IS2, IS_std2 = compute_IS(PreNetFIDIS, real_images2[indx_shuffle_test], batch_size = 500, splits=10, resize = (299,299), verbose=False, normalize_img = True)
print("\n IS of {} test images: {:.3f}({:.3f}).".format(len(real_images2), IS2, IS_std2))

## 计算FID
indx_shuffle_real = np.arange(len(real_images)); np.random.shuffle(indx_shuffle_real)
indx_shuffle_real2 = np.arange(len(real_images2)); np.random.shuffle(indx_shuffle_real2)
FID_score2 = cal_FID(PreNetFIDIS, real_images[indx_shuffle_real], real_images2[indx_shuffle_real2], batch_size=200, resize = (299,299), norm_img = True)
print("\n FID of {} test images: {:.3f}.".format(len(real_images2), FID_score2))
