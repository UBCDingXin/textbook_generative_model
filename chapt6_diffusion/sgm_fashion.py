## 参考：https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=DfOkg5jBZcjF


import os
import math
from abc import abstractmethod
import timeit
import random
import h5py

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt  
import functools
from torch.optim import Adam
from torchvision.utils import make_grid
from scipy import integrate
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




############################################################
## 实验设置

setup_name="sgm_s1"

# 文件路径
data_path = "../data"
output_path = "./output/{}".format(setup_name)
os.makedirs(output_path, exist_ok=True)
path_to_saved_models = os.path.join(output_path, "saved_models")
os.makedirs(path_to_saved_models, exist_ok=True)
path_to_saved_images = os.path.join(output_path, "saved_images")
os.makedirs(path_to_saved_images, exist_ok=True)
eval_model_path = "./output"

## 参数设置
IMG_SIZE=28
NC=1
NUM_CLASS=10

EPOCHS=500
RESUME_EPOCH=0
BATCH_SIZE=128
LR=1e-4
SAVE_FREQ=50 #每个XX个epoch，保存一次
SHOW_FREQ=5 #每个XX个epoch，输出一次图片
SIGMA=25.0

SNet_NC=[256, 512, 1024, 2048]

NFAKE=10000 #生成数据用于评价
SAMPLE_BATCH_SIZE=1000
SAMPLER="pc_sampler" # ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler']
SAMPLE_NUM_STEPS=500

## 设定设备  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)

# 数据载入与预处理  
transform = transforms.Compose([  
    # transforms.Resize([IMG_SIZE,IMG_SIZE]),
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])  
train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform) 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
test_dataset = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform) 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)




############################################################
## 定义分数网络S_theta

class GaussianFourierProjection(nn.Module):
    """用于编码时间步长的高斯随机特征。"""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # 在初始化时随机采样权重。这些权重在优化过程中是固定的，并且不可训练。
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """一个将输出重塑为特征图的全连接层。"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]

class ScoreNet(nn.Module):
    """一个基于U-Net架构构建的时间依赖型的基于分数的模型。"""

    def __init__(self, marginal_prob_std, channels=SNet_NC, embed_dim=256, input_channel=NC, output_channel=NC):
        """初始化一个时间依赖的基于分数的网络。

        Args:
            marginal_prob_std: 一个以时间t为输入并给出扰动核p_{0t}(x(t) | x(0))标准差的函数
            channels: 每种分辨率特征图的通道数。
            embed_dim: 高斯随机特征嵌入的维度。
        """
        super().__init__()
        # 用于时间的高斯随机特征嵌入层
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        # 分辨率逐渐降低的编码层
        self.conv1 = nn.Conv2d(input_channel, channels[0], 3, stride=2, padding=1, bias=True) #28->14
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=True) #14->7
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 4, stride=1, padding=0, bias=True) #7->4
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=True) #4->2
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

        # 分辨率逐渐升高的解码层
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=1, bias=True) #2->4
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 4, stride=1, padding=0, bias=True)  # 4->7
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 4, stride=2, padding=1, bias=True) # 7->14
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], output_channel, 4, stride=2, padding=1) # 14->28

        # 激活函数
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
  
    def forward(self, x, t): 
        # 获取时间t的高斯随机特征嵌入  
        embed = self.act(self.embed(t))    
        h1 = self.conv1(x)            
        ## 融入来自t的信息
        h1 += self.dense1(embed)
        ## 群组归一化
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # 输出的标准化
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h




############################################################
# Set up the SDE
def marginal_prob_std(t, sigma):
    """计算p_0t(x(t)|x(0))的均值和标准差。 
    输入:    
    t: 一个时间步向量.
    sigma: SDE中的sigma
    Returns: 标准差
    """    
    # t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """
    计算我们随机微分方程（SDE）的扩散系数。
    Args:
    t: 一个时间步向量.
    sigma: SDE中的sigma
    Returns: 扩散系数
    """
    return torch.tensor(sigma**t, device=device)

marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=SIGMA)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=SIGMA)


############################################################
## Define the loss function
def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """基于评分的生成模型的损失函数。.

    Args:
    model: A PyTorch model instance that represents a 
        time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss





############################################################
## 实例化
# score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
score_model = score_model.to(device)
optimizer = Adam(score_model.parameters(), lr=LR)

if RESUME_EPOCH>0:
    save_file = path_to_saved_models + "/ckpts_in_train/ckpt_epoch_{}.pth".format(RESUME_EPOCH)
    checkpoint = torch.load(save_file, weights_only=False)
    score_model.load_state_dict(checkpoint['model'])
    torch.set_rng_state(checkpoint['rng_state'])

############################################################
## 训练
start_time = timeit.default_timer()
for epoch in range(RESUME_EPOCH, EPOCHS):
    score_model.train()
    avg_loss = 0.
    num_items = 0
    for x, y in train_loader:
        x = x.to(device)    
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    # Print the averaged training loss so far.
    print("\r Epoch:{}/{}, Average Loss: {:.5f}, Time:{:.3f}".format(epoch+1, EPOCHS, avg_loss/num_items, timeit.default_timer()-start_time))
    # Update the checkpoint after each epoch of training.
    if (epoch+1)%(SAVE_FREQ)==0:
        save_file = path_to_saved_models + "/ckpts_in_train/ckpt_epoch_{}.pth".format(epoch+1)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        torch.save({
                'model': score_model.state_dict(),
                'rng_state': torch.get_rng_state(),
        }, save_file)
    # end if
# end for




############################################################
## 定义不同采样器

# 1. Define the Euler-Maruyama sampler
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           nfake,
                           batch_size=100, 
                           num_steps=500,
                           num_channels=1,
                           img_size=32, 
                           device='cuda', 
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.    
    """
    
    if batch_size>nfake:
        batch_size = nfake
    assert nfake%batch_size==0
    
    fake_images = [] #存放采样的图片
    ngot=0 # 已获得样本的数量
    while ngot<nfake:
        t = torch.ones(batch_size, device=device)
        init_x = torch.randn(batch_size, num_channels, img_size, img_size, device=device) \
        * marginal_prob_std(t)[:, None, None, None]
        time_steps = torch.linspace(1., eps, num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            for i in trange(len(time_steps)):
                time_step = time_steps[i]   
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                g = diffusion_coeff(batch_time_step)
                mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
        fake_images.append(mean_x.cpu().numpy())
        ngot+=len(mean_x)
        print("\r Got {}/{} fake images.".format(ngot, nfake))
    fake_images = np.concatenate(fake_images, axis=0)
    return fake_images[0:nfake]
    




# 2. Define the Predictor-Corrector sampler
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               nfake,
               batch_size=100, 
               num_steps=500, 
               snr=0.16,
               num_channels=1,
               img_size=32,                 
               device='cuda',
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
        of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns: 
    Samples.
    """
    
    if batch_size>nfake:
        batch_size = nfake
    assert nfake%batch_size==0
    
    fake_images = [] #存放采样的图片
    ngot=0 # 已获得样本的数量
    while ngot<nfake:
        t = torch.ones(batch_size, device=device)
        init_x = torch.randn(batch_size, num_channels, img_size, img_size, device=device) * marginal_prob_std(t)[:, None, None, None]
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():  
            for i in trange(len(time_steps)):
                time_step = time_steps[i]
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                # Corrector step (Langevin MCMC)
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

                # Predictor step (Euler-Maruyama)
                g = diffusion_coeff(batch_time_step)
                x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
                x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
            # The last step does not include any noise
        fake_images.append(x_mean.cpu().numpy())
        ngot+=len(x_mean)
        print("\r Got {}/{} fake images.".format(ngot, nfake))
    fake_images = np.concatenate(fake_images, axis=0)
    return fake_images




# 3. Define the ODE sampler (double click to expand or collapse)
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                nfake,
                batch_size=100, 
                num_steps=None,
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                z=None,
                num_channels=1,
                img_size=32,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
        of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
    """
    
    if batch_size>nfake:
        batch_size = nfake
    assert nfake%batch_size==0
    
    fake_images = [] #存放采样的图片
    ngot=0 # 已获得样本的数量
    while ngot<nfake:
        t = torch.ones(batch_size, device=device)
        # Create the latent code
        if z is None:
            init_x = torch.randn(batch_size, num_channels, img_size, img_size, device=device) * marginal_prob_std(t)[:, None, None, None]
        else:
            init_x = z
        
        shape = init_x.shape

        def score_eval_wrapper(sample, time_steps):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
            with torch.no_grad():    
                score = score_model(sample, time_steps)
            return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((shape[0],)) * t    
            g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
            return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
    
        # Run the black-box ODE solver.
        res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
        print(f"Number of function evaluations: {res.nfev}")
        x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

        fake_images.append(x.cpu().numpy())
        ngot+=len(x)
        print("\r Got {}/{} fake images.".format(ngot, nfake))
    fake_images = np.concatenate(fake_images, axis=0)
    return fake_images



############################################################
## 采样

# 载入预训练的网络
save_file = path_to_saved_models + "/ckpts_in_train/ckpt_epoch_{}.pth".format(EPOCHS)
checkpoint = torch.load(save_file, weights_only=False)
score_model.load_state_dict(checkpoint['model'])

if SAMPLER=="Euler_Maruyama_sampler":
    sampler = Euler_Maruyama_sampler
elif SAMPLER=="pc_sampler":
    sampler = pc_sampler
else:
    sampler = ode_sampler

dump_fake_images_filename = os.path.join(path_to_saved_images, 'fake_data_{}.h5'.format(NFAKE))
if not os.path.isfile(dump_fake_images_filename):
    fake_images = sampler(score_model, 
                            marginal_prob_std_fn,
                            diffusion_coeff_fn, 
                            nfake=NFAKE,
                            batch_size=SAMPLE_BATCH_SIZE, 
                            num_steps=SAMPLE_NUM_STEPS,
                            num_channels=NC,
                            img_size=IMG_SIZE, 
                            device=device)
    fake_images = np.clip(fake_images, -1.0, 1.0)
    assert fake_images.min()<=0 and fake_images.min()>=-1 and fake_images.max()<=1
    fake_images = ((fake_images*0.5+0.5)*255.0).astype(int)
    print("Fake image range:", fake_images.min(), fake_images.max())
    print(fake_images.shape)
    with h5py.File(dump_fake_images_filename, "w") as f:
        f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
else:
    with h5py.File(dump_fake_images_filename, "r") as f:
        fake_images = f['fake_images'][:]


#########################################
# 评价指标计算

PreNetFIDIS = Inception3(num_classes=NUM_CLASS, aux_logits=True, transform_input=False, input_channel=NC).to(device)
# PreNetFIDIS = nn.DataParallel(PreNetFIDIS)
filename_ckpt = './output/ckpt_InceptionV3_epoch200.pth'
checkpoint_PreNet = torch.load(filename_ckpt, weights_only=True)
PreNetFIDIS.load_state_dict(checkpoint_PreNet['net_state_dict'])

## 准备测试样本用于计算FID
real_images = train_dataset.train_data[:,np.newaxis,:,:].numpy()

## 计算IS
indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
IS, IS_std = compute_IS(PreNetFIDIS, fake_images[indx_shuffle_fake], batch_size = 500, splits=10, resize = (299,299), verbose=False, normalize_img = True)
print("\n IS of {} fake images: {:.3f}({:.3f}).".format(NFAKE, IS, IS_std))

## 计算FID
indx_shuffle_real = np.arange(len(real_images)); np.random.shuffle(indx_shuffle_real)
indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
FID_score = cal_FID(PreNetFIDIS, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=200, resize = (299,299), norm_img = True)
print("\n FID of {} fake images: {:.3f}.".format(NFAKE, FID_score))


#########################################
# 可视化25张图像
nrow= 5
sample_images = fake_images[0:nrow**2]/255.0
sample_images = [np.transpose(sample_images[i],axes=[1,2,0]) for i in range(len(sample_images))]
fig = plt.figure(figsize=(nrow, nrow))
grid = ImageGrid(fig, 111, nrows_ncols=(nrow, nrow), axes_pad=0.1)
for ax, im in zip(grid, sample_images):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
plt.savefig(output_path + "/fake_imgs_epoch_{}.png".format(EPOCHS), dpi=500,bbox_inches="tight")