"""
Compute
Frechet Inception Discrepency (FID), ref "https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py"
for a set of fake images

use numpy array
Xr: high-level features for real images; nr by d array
Yr: labels for real images
Xg: high-level features for fake images; ng by d array
Yg: labels for fake images
IMGSr: real images
IMGSg: fake images

"""

import os
import gc
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data
import sys
from scipy.stats import entropy
from scipy import linalg
from torch.autograd import Variable


# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width
    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')

def normalize_images(batch_images):
    batch_images = batch_images/255.0
    batch_images = (batch_images - 0.5)/0.5
    return batch_images

##############################################################################
# FID scores
##############################################################################
def FID(Xr, Xg, eps=1e-10):
    '''
    两个多元高斯分布X_r ~ N(mu_1, C_1)和X_g ~ N(mu_2, C_2)之间的FID距离
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Xr和Xg分别为图片中通过CNN提取出的高级别（high level）特征
    '''
    #计算样本均值
    MUr = np.mean(Xr, axis = 0)
    MUg = np.mean(Xg, axis = 0)
    mean_diff = MUr - MUg
    #计算样本协方差矩阵
    SIGMAr = np.cov(Xr.transpose())
    SIGMAg = np.cov(Xg.transpose())

    # 计算矩阵的平方根，并处理可能的奇异矩阵问题
    covmean, _ = linalg.sqrtm(SIGMAr.dot(SIGMAg), disp=False)
    covmean = covmean.real
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(SIGMAr.shape[0]) * eps
        covmean = linalg.sqrtm((SIGMAr + offset).dot(SIGMAg + offset))

    #计算FID分数
    fid_score = mean_diff.dot(mean_diff) + np.trace(SIGMAr + SIGMAg - 2*covmean)

    return fid_score

##test
#Xr = np.random.rand(10000,1000)
#Xg = np.random.rand(10000,1000)
#print(FID(Xr, Xg))

# compute FID from raw images
def cal_FID(PreNetFID, IMGSr, IMGSg, batch_size = 500, resize = None, norm_img = False):
    #resize: if None, do not resize; if resize = (H,W), resize images to 3 x H x W
    
    PreNetFID.eval()

    nr = IMGSr.shape[0]
    ng = IMGSg.shape[0]

    nc = IMGSr.shape[1] #IMGSr is nrxNCxIMG_SIExIMG_SIZE
    img_size = IMGSr.shape[2]

    if batch_size > min(nr, ng):
        batch_size = min(nr, ng)
        # print("FID: recude batch size to {}".format(batch_size))

    #compute the length of extracted features
    with torch.no_grad():
        test_img = torch.from_numpy(IMGSr[0].reshape((1,nc,img_size,img_size))).type(torch.float).cuda()
        if resize is not None:
            test_img = nn.functional.interpolate(test_img, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
        if norm_img:
            test_img = normalize_images(test_img)
        _, test_features = PreNetFID(test_img)
        d = test_features.shape[1] #length of extracted features

    Xr = np.zeros((nr, d))
    Xg = np.zeros((ng, d))

    with torch.no_grad():
        tmp = 0
        pb1 = SimpleProgressBar()
        for i in range(nr//batch_size):
            imgr_tensor = torch.from_numpy(IMGSr[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            if resize is not None:
                imgr_tensor = nn.functional.interpolate(imgr_tensor, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
            if norm_img:
                imgr_tensor = normalize_images(imgr_tensor)
            _, Xr_tmp = PreNetFID(imgr_tensor)
            Xr[tmp:(tmp+batch_size)] = Xr_tmp.detach().cpu().numpy()
            tmp+=batch_size
            pb1.update(min(max(tmp/nr*100,100), 100))
        del Xr_tmp,imgr_tensor; gc.collect()
        torch.cuda.empty_cache()

        tmp = 0
        pb2 = SimpleProgressBar()
        for j in range(ng//batch_size):
            imgg_tensor = torch.from_numpy(IMGSg[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            if resize is not None:
                imgg_tensor = nn.functional.interpolate(imgg_tensor, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
            if norm_img:
                imgg_tensor = normalize_images(imgg_tensor)
            _, Xg_tmp = PreNetFID(imgg_tensor)
            Xg[tmp:(tmp+batch_size)] = Xg_tmp.detach().cpu().numpy()
            tmp+=batch_size
            pb2.update(min(max(tmp/ng*100, 100), 100))
        del Xg_tmp,imgg_tensor; gc.collect()
        torch.cuda.empty_cache()

    fid_score = FID(Xr, Xg, eps=1e-6)

    return fid_score




##############################################################################
# Inception Scores
##############################################################################

def compute_IS(PreNetIS, IMGSg, batch_size = 500, splits=1, resize = None, verbose=False, normalize_img = True):
    #resize: if None, do not resize; if resize = (H,W), resize images to 3 x H x W

    # PreNetIS = PreNetIS.cuda()
    PreNetIS.eval()

    N = IMGSg.shape[0]
    

    #compute the number of classes
    with torch.no_grad():
        test_img = torch.from_numpy(IMGSg[0:2].reshape((2,IMGSg.shape[1],IMGSg.shape[2],IMGSg.shape[3]))).type(torch.float).cuda()
        if resize is not None:
            test_img = nn.functional.interpolate(test_img, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
        if normalize_img:
            test_img = normalize_images(test_img)
        test_output, _ = PreNetIS(test_img)
        nc = test_output.shape[1] #number of classes

    # Get predictions
    def get_pred(x):
        x, _ = PreNetIS(x)
        return F.softmax(x,dim=1).data.cpu().numpy()

    preds = np.zeros((N, nc))

    with torch.no_grad():
        tmp = 0
        if verbose:
            pb = SimpleProgressBar()
        for j in range(N//batch_size):
            imgg_tensor = torch.from_numpy(IMGSg[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            if resize is not None:
                imgg_tensor = nn.functional.interpolate(imgg_tensor, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
            if normalize_img:
                imgg_tensor = normalize_images(imgg_tensor)
            preds[tmp:(tmp+batch_size)] = get_pred(imgg_tensor)
            tmp+=batch_size
            if verbose:
                pb.update(min(100,float(j+1)*100/(N//batch_size)))

    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)