# [《生成式视觉模型原理与实践》](https://www.hxedu.com.cn/hxedu/hg/book/bookInfo.html?code=G0507040)的配套资料

本代码仓库（[GitHub地址](https://github.com/UBCDingXin/textbook_generative_model)）包含了《生成式视觉模型原理与应用》（丁鑫、许祖衡、陈哲）的配套资料。


## 数据集和评价模型下载：

第3章~第6章实验中，用于计算FID和IS分数的Inception V3模型的checkpoint（文件名为“ckpt_InceptionV3_epoch200.pth”），可从[此处](https://pan.baidu.com/s/1l14o4YPwiHGlED_PRKRAfA?pwd=3sdb)下载。下载完毕后，请将“ckpt_InceptionV3_epoch200.pth”放置每个实验的output文件夹中。

第7章前三个实验的数据集可由以下网址下载：</br>
[monet2photo数据集](https://pan.baidu.com/s/1QFZAAwctcBCNMGt9Gj1kdA?pwd=mcex)</br>
[CT2MRI数据集](https://pan.baidu.com/s/1AOTmcboIswwjKN2Gr_EqSQ?pwd=8ppp)</br>
[OLI2MSI数据集](https://pan.baidu.com/s/1Qg9CqAwKMhvF_03-RSMG-A?pwd=dqd2)</br>

## 代码运行
- 第3章：在命令行执行：
```python vae_fashion.py```

- 第4章：在命令行执行：
```python dcgan_fashion.py```
与
```python sngan_fashion.py```

- 第5章：在命令行执行：
```python realnvp_fashion.py```

- 第6章：在命令行执行：
```python ddpm_fashion.py ```
与
```python sgm_fashion.py ```

- 第7章：在命令行执行各自文件夹下的`run.sh`文件（linux系统）。


## 致谢
本书实验部分的代码参考或利用了以下资料：</br>
1. https://github.com/aitorzip/PyTorch-CycleGAN
2. https://github.com/leftthomas/SRGAN
3. https://github.com/christiancosgrove/pytorch-spectral-normalization-gan 
4. https://github.com/voletiv/self-attention-GAN-pytorch/tree/master 
5. https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=DfOkg5jBZcjF
6. https://github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddpm_mnist.ipynb