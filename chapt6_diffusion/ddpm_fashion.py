### 以下代码重点参考：https://github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddpm_mnist.ipynb

import os
import math
from abc import abstractmethod
import timeit
import random

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import h5py
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

setup_name="ddpm_s1"

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
NUM_CLASS=10
NC=1

EPOCHS=200
RESUME_EPOCH=0
BATCH_SIZE=128
LR=5e-5
TIMESTEPS=1000
SAVE_FREQ=20 #每个XX个epoch，保存一次
SHOW_FREQ=5 #每个XX个epoch，输出一次图片
VAR_SCHEDULER="cosine" #方差表类型

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
train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)




############################################################
## 定义UNet

# 使用正弦位置嵌入来编码时间步长 (参见https://arxiv.org/abs/1706.03762)
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    timesteps: 一个长度为N的1维张量
    dim: 输出的维度
    max_prediod：编码的一个参数，它控制着嵌入的最小频率
    输出：一个[N x dim]的时间编码
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# 将 TimestepEmbedSequential 定义为支持 time_emb 作为额外输入
class TimestepBlock(nn.Module):
    """
    任何模块的forward方法将时间嵌入作为第二输入参数
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        将该模块应用于`x`，同时给定`emb`作为时间其嵌入。
        """

# 一个Sequential模块，它将时间嵌入传递给子模块以作为其额外的输入。
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

# 使用群组归一化
def norm_layer(channels):
    return nn.GroupNorm(32, channels)

# 残差块（Residual block）
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels) )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` 的维度是 `[batch_size, in_dim, height, width]`
        `t` 的维度是 `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # 加入时间嵌入
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

# 注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

# 上采样
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

# 下采样
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


# 带有注意力机制和时间嵌入的完整UNet网络
class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        参数：
        x: an [N x C x H x W] Tensor of inputs.
        timesteps: a 1-D batch of timesteps.
        输出: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)




############################################################
## 定义diffusion

## 设置方差表
# 线性表
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
# 余弦表
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

## 高斯扩散过程
class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps

        # 计算方差表beta：线性表或者余弦表
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        #根据公式（6-8）
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # 计算一些与q(x_t | x_{t-1})均值或方差相关的常数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 计算q(x_{t-1} | x_t, x_0) 的方差与均值
        # 方差：参照公式（6-33）
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 因为在扩散开始时后验方差为0，所以此处对数计算被截断，以防止NaN
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        # 均值：根据公式（6-32）
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # 在给定时刻t下，获取一些与alpha相关的常数值。例如:sqrt{bar{alpha}_t}
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # 正向扩散过程：q(x_t | x_0)
    # 根据公式（6-13），根据x_0，生成x_t
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # 计算q(x_t | x_0)的均值与方差；根据公式（6-14）
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # 计算q(x_{t-1} | x_t, x_0)的均值和方差；根据公式（6-32）与（6-33）
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 根据x_t和预测的噪声，计算x_0; 根据公式（6-52）
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # 计算p_theta(x_{t-1} | x_t)的均值和方差
    # 根据公式（6-41）和（6-42）
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # 根据x_t，预测噪声epsilon
        pred_noise = model(x_t, t)
        # 根据x_t和预测噪声，以及公式（6-52），计算x_0的预测值
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    # 去噪步骤：根据加噪图像x_t和预测噪声pred_noise，采样t-1时刻的样本x_{t-1}
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        # 获得均值和方差
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # 当 t == 0，无噪声
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # 采样 x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    # 整个逆向扩散过程
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # 从纯噪声开始，利用self.p_sample，按照t=T,...,0的顺序采样，直到x_0
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
        return img.cpu()

    # 采样函数
    @torch.no_grad()
    def sample(self, model, nfake, image_size, batch_size=100, channels=3, to_numpy=False, unnorm_to_zero2one=True):

        if batch_size>nfake:
            batch_size = nfake
        assert nfake%batch_size==0

        fake_images = [] #存放采样的图片
        ngot=0 # 已获得样本的数量
        while ngot<nfake:
            batch_fake_images = self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
            fake_images.append(batch_fake_images)
            ngot+=len(batch_fake_images)
            print("\r Got {}/{} fake images.".format(ngot, nfake))
        fake_images=torch.cat(fake_images,dim=0)

        # 去归一化至[0,1]
        if unnorm_to_zero2one:
            fake_images = (fake_images + 1) * 0.5
        # 将torch tensor转换为numpy array
        if to_numpy:
            fake_images = fake_images.numpy()

        return fake_images

    # 计算训练损失；即公式（6-91）
    def train_losses(self, model, x_start, t):
        # 采样源噪声
        noise = torch.randn_like(x_start)
        # 获得加噪图像 x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        # 根据x_t，用模型预测源噪声
        predicted_noise = model(x_noisy, t)
        # 计算预测源噪声与真实源噪声之间的MSE损失
        loss = F.mse_loss(noise, predicted_noise)
        return loss



# ### 测试加噪过程
# image = Image.open("./dog.png").convert("RGB")

# print((np.array(image)).shape)

# image_size = 128
# transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.PILToTensor(),
#     transforms.ConvertImageDtype(torch.float),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])

# x_start = transform(image).unsqueeze(0)

# gaussian_diffusion = GaussianDiffusion(timesteps=500)
# images_list = []
# for idx, t in enumerate([0, 50, 100, 200, 499]):
#     x_noisy = gaussian_diffusion.q_sample(x_start, t=torch.tensor([t]))
#     noisy_image = (x_noisy.squeeze().permute(1, 2, 0) + 1) * 127.5
#     noisy_image = noisy_image.numpy().astype(np.uint8)
#     images_list.append(noisy_image)

# fig, ax = plt.subplots(1, 5, figsize=(15, 3))  # 调整 figsize 根据图像尺寸和需求

# for i, image in enumerate(images_list):
#     ax[i].imshow(image)
#     ax[i].axis('off')  # 隐藏坐标轴

# plt.subplots_adjust(wspace=0, hspace=0)
# plt.savefig('./noisy_images.png', bbox_inches='tight', pad_inches=0)
# plt.close()


############################################################
## 模型实例化、定义训练器等

model = UNetModel(
    in_channels=NC,
    model_channels=64,
    out_channels=NC,
    channel_mult=(1, 2, 2),
    attention_resolutions=[]
)
model.to(device)

if RESUME_EPOCH>0:
    save_file = path_to_saved_models + "/ckpts_in_train/ckpt_epoch_{}.pth".format(RESUME_EPOCH)
    checkpoint = torch.load(save_file, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    torch.set_rng_state(checkpoint['rng_state'])

gaussian_diffusion = GaussianDiffusion(timesteps=TIMESTEPS, beta_schedule=VAR_SCHEDULER)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



############################################################
## 训练
start_time = timeit.default_timer()
for epoch in range(RESUME_EPOCH, EPOCHS):
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        batch_size = images.shape[0]
        images = images.to(device)

        # 对批次中的每个样本均匀地采样t
        t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()

        loss = gaussian_diffusion.train_losses(model, images, t)

        loss.backward()
        optimizer.step()
    ##end for
    print("epoch:[{}/{}], Loss:{}, [Time: {:.4f}]".format(epoch+1, EPOCHS, loss.item(), timeit.default_timer()-start_time))

    if (epoch+1)%(SAVE_FREQ)==0:
        save_file = path_to_saved_models + "/ckpts_in_train/ckpt_epoch_{}.pth".format(epoch+1)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        torch.save({
                'model': model.state_dict(),
                'rng_state': torch.get_rng_state(),
        }, save_file)

    if (epoch+1)%(SHOW_FREQ)==0:
        model.eval()
        with torch.inference_mode():
            gen_imgs = gaussian_diffusion.sample(model, nfake=100, image_size=IMG_SIZE, batch_size=100, channels=NC, to_numpy=False, unnorm_to_zero2one=True)
            save_file = path_to_saved_images + '/imgs_in_train/{}.png'.format(epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            save_image(gen_imgs.data, save_file, nrow=10, normalize=True)
##end for



############################################################
## 采样

sample_images = gaussian_diffusion.sample(model, 25, image_size=IMG_SIZE, batch_size=25, channels=NC, to_numpy=True, unnorm_to_zero2one=False)
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
filename_ckpt = './output/ckpt_InceptionV3_epoch200.pth'
checkpoint_PreNet = torch.load(filename_ckpt, weights_only=True)
PreNetFIDIS.load_state_dict(checkpoint_PreNet['net_state_dict'])


## 生成虚假样本
dump_fake_images_filename = os.path.join(path_to_saved_images, 'fake_data_{}.h5'.format(NFAKE))
if not os.path.isfile(dump_fake_images_filename):
    fake_images = gaussian_diffusion.sample(model, NFAKE, image_size=IMG_SIZE, batch_size=2500, channels=NC, to_numpy=True, unnorm_to_zero2one=False)
    assert fake_images.min()<=0 and fake_images.min()>=-1 and fake_images.max()<=1
    fake_images = ((fake_images*0.5+0.5)*255.0).astype(int)
    print("Fake image range:", fake_images.min(), fake_images.max())
    print(fake_images.shape)
    with h5py.File(dump_fake_images_filename, "w") as f:
        f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
else:
    with h5py.File(dump_fake_images_filename, "r") as f:
        fake_images = f['fake_images'][:]

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
