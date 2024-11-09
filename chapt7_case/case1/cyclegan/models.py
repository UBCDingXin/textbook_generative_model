import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# class Discriminator(nn.Module):
#     def __init__(self, input_nc):
#         super(Discriminator, self).__init__()

#         # A bunch of convolutions one after another
#         model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
#                     nn.InstanceNorm2d(128), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
#                     nn.InstanceNorm2d(256), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(256, 512, 4, padding=1),
#                     nn.InstanceNorm2d(512), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         # FCN classification layer
#         model += [nn.Conv2d(512, 1, 4, padding=1)]

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         x =  self.model(x)
#         print(F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).shape)
#         # Average pooling and flatten
#         return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# class Discriminator(nn.Module):
#     def __init__(self, input_nc):
#         super(Discriminator, self).__init__()

#         # A bunch of convolutions one after another
#         model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
#                     nn.InstanceNorm2d(128), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
#                     nn.InstanceNorm2d(256), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(256, 512, 3, stride=2, padding=1),
#                     nn.InstanceNorm2d(512), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(512, 1024, 3, stride=2, padding=1),
#                     nn.InstanceNorm2d(1024), 
#                     nn.LeakyReLU(0.2, inplace=True) ]
        
#         model += [  nn.Conv2d(1024, 512, 3, stride=2, padding=1)]

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         x =  self.model(x)
#         return x.view(x.size()[0], -1)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(512, 1024, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(1024), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        
        model += [  nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)]

        self.model = nn.Sequential(*model)
        
        self.linear = nn.Linear(1024*4, 1)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)

    def forward(self, x):
        out =  self.model(x)
        out = out.view(-1, 1024*4)
        out = self.linear(out)
        return out