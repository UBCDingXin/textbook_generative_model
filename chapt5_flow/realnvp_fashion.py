import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torch.nn.functional as F

import os
import timeit

import matplotlib.pyplot as plt
from torchvision.utils import save_image
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

INPUT_DIM = IMG_SIZE**2 # 将图像拉平成向量后的长度
HIDDEN_DIM = 1024 # 隐藏层维度
N_COUPLING_LAYERS = 30

EPOCHS=500
RESUME_EPOCH=500
LR = 5e-5
WEIGHT_DECAY=1e-4
BATCH_SIZE = 256

NFAKE=10000

## 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 输出路径
output_path = "./output/fashion_mnist"
os.makedirs(output_path, exist_ok=True)
path_to_saved_models = os.path.join(output_path, "saved_models")
os.makedirs(path_to_saved_models, exist_ok=True)
path_to_saved_images = os.path.join(output_path, "saved_images")
os.makedirs(path_to_saved_images, exist_ok=True)


###################################################
# 数据载入与预处理
transform = transforms.Compose([
    # transforms.Resize([IMG_SIZE,IMG_SIZE]),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)



###################################################
# 构建网络

# 定义仿射耦合层
class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AffineCoupling, self).__init__()

        self.scale_net = nn.Sequential(
            nn.Linear(input_dim//2, hidden_dim),
            nn.GroupNorm(16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim//2),
            nn.Tanh()  # 为保持训练稳定性，限制尺度输出的范围
        )

        self.translate_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.GroupNorm(16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )

    def forward(self, x):
        """
        Affine Coupling层的正向传播。
        从训练样本到噪声。（x_k -> x_0）
        """
        # 将输入分成两半
        x1, x2 = x.chunk(2, dim=1)

        # 计算缩放和平移
        log_scale = self.scale_net(x1)
        translate = self.translate_net(x1)

        # 对x2应用仿射变换
        # 使用exp再参数化技巧确保缩放因子为正，从而使仿射变换可逆
        z1 = x1
        z2 = x2 * torch.exp(log_scale) + translate

        # 计算雅可比矩阵的对数行列式
        log_det_jacobian = log_scale.sum(dim=1)

        return torch.cat([z1, z2], dim=1), log_det_jacobian

    def inverse(self, z):
        z1, z2 = z.chunk(2, dim=1)

        log_scale = self.scale_net(z1)
        translate = self.translate_net(z1)

        x1 = z1
        x2 = (z2 - translate) * torch.exp(-log_scale)

        return torch.cat([x1, x2], dim=1)


class RandomPermutation(nn.Module):
    """
    在RealNVP模型中，层与层之间采用固定的随机置换。
    我们将在每个仿射耦合层之间添加一个置换层。
    """

    def __init__(self, input_dim):
        super(RandomPermutation, self).__init__()
        self.permutation = torch.randperm(input_dim)

    def forward(self, x):
        return x[:, self.permutation]

    def inverse(self, x):
        inverse_permutation = torch.argsort(self.permutation)
        return x[:, inverse_permutation]


class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_coupling_layers):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim

        # 创建一个由仿射耦合层和置换层组成的序列
        layers = []
        for _ in range(n_coupling_layers):
            layers.append(AffineCoupling(input_dim, hidden_dim))
            layers.append(RandomPermutation(input_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        log_det_jacobian_total = 0

        # 应用耦合层和置换层的序列
        for layer in self.layers:
            if isinstance(layer, AffineCoupling):
                x, log_det_jacobian = layer(x)
                log_det_jacobian_total += log_det_jacobian
            else:
                x = layer(x)

        return x, log_det_jacobian_total

    def inverse(self, z):
        # 通过各层进行逆向传播
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


###################################################
# 实例化模型
model = RealNVP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, n_coupling_layers=N_COUPLING_LAYERS)
model = model.to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


###################################################
# 训练过程

def train(model, optimizer, epochs, resume_epoch, device):

    if resume_epoch>0:
        save_file = path_to_saved_models + "/ckpts_in_train/ckpt_epoch_{}.pth".format(resume_epoch)
        checkpoint = torch.load(save_file)
        model.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    # 用于存储每个epoch的平均损失
    loss_over_epochs = []
    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        model.train()
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            batch_size = x.size(0)
            x = x.view(x.size(0), -1)
            optimizer.zero_grad()

            # 计算对数似然
            # Transform data and compute the log likelihood
            z, log_det_jacobian = model(x)
            log_prob_reference = -0.5 * (z**2 + torch.log(torch.tensor(2 * np.pi))).sum(dim=1)
            log_likelihood = log_prob_reference + log_det_jacobian

            # Minimize the negative log likelihood
            loss = -log_likelihood.mean()

            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        ##end for

        loss_over_epochs.append(overall_loss/(batch_idx*batch_size))

        print("\r Epoch:{}/{}, Avg. Loss: {:.3f}, Time: {:.3f}".format(epoch + 1, EPOCHS, overall_loss/(batch_idx*batch_size), timeit.default_timer()-start_time ))


        # 每10个epoch生成100个样本用于可视化
        if (epoch+1)%50==0:
            model.eval()
            with torch.no_grad():
                z_gen = torch.randn(100, INPUT_DIM).to(device)
                sample_images = model.inverse(z_gen)
                sample_images = torch.clamp(sample_images, min=0, max=1)
                sample_images = sample_images.detach().cpu().view(100, 1, IMG_SIZE, IMG_SIZE)
                save_file = path_to_saved_images + "/images_in_train/{}.png".format(epoch+1)
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                save_image(sample_images.data, save_file, nrow=10, normalize=True)


        # 每50个epoch将模型参数存储一次
        if (epoch+1)%50==0:
            save_file = path_to_saved_models + "/ckpts_in_train/ckpt_epoch_{}.pth".format(epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'net_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    ##end for

    return np.array(loss_over_epochs)


loss_over_epochs = train(model, optimizer, epochs=EPOCHS, resume_epoch=RESUME_EPOCH, device=device)



###################################################
# 可视化
nrow = 5
model.eval()

with torch.no_grad():
    z_gen = torch.randn(nrow**2, INPUT_DIM).to(device)
    sample_images = model.inverse(z_gen)
    sample_images = torch.clamp(sample_images, min=0, max=1)
    sample_images = sample_images.detach().cpu().view(nrow**2, 1, IMG_SIZE, IMG_SIZE).numpy()

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

## 准备测试样本用于计算FID
real_images = train_dataset.train_data[:,np.newaxis,:,:].numpy()
print("Real image range:", len(real_images), real_images.min(), real_images.max())
print(real_images.shape)

## 生成虚假样本
start_time = timeit.default_timer()
model.eval()
fake_images = []
ngot=0
batch_size = 1000
while ngot<NFAKE:
    z_gen = torch.randn(batch_size, INPUT_DIM).to(device)
    batch_fake_images = model.inverse(z_gen)
    batch_fake_images = torch.clamp(batch_fake_images, min=0, max=1)
    batch_fake_images = batch_fake_images.detach().cpu().view(batch_size, 1, IMG_SIZE, IMG_SIZE).numpy()
    batch_fake_images = (batch_fake_images*255.0).astype(int)
    fake_images.append(batch_fake_images)
    ngot+=len(batch_fake_images)
fake_images = np.concatenate(fake_images, axis=0)
fake_images = fake_images[0:NFAKE]
print("Fake image range:", len(fake_images), fake_images.min(), fake_images.max())
print(fake_images.shape)
print("采样速度：{}（张/秒）".format(NFAKE/(timeit.default_timer()-start_time)))

## 计算IS
indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
IS, IS_std = compute_IS(PreNetFIDIS, fake_images[indx_shuffle_fake], batch_size = 500, splits=10, resize = (299,299), verbose=False, normalize_img = True)
print("\n IS of {} fake images: {:.3f}({:.3f}).".format(NFAKE, IS, IS_std))

## 计算FID
indx_shuffle_real = np.arange(len(real_images)); np.random.shuffle(indx_shuffle_real)
indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
FID_score = cal_FID(PreNetFIDIS, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=200, resize = (299,299), norm_img = True)
print("\n FID of {} fake images: {:.3f}.".format(NFAKE, FID_score))
