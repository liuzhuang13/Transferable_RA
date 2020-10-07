### code from https://github.com/leftthomas/SRGAN/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch.autograd import Variable

from math import sqrt
import numpy as np 
import pdb


class SimpleNet(nn.Module):
    def __init__(self, upscale_factor, channel):
        super(SimpleNet, self).__init__()
        # if channel == 'RGB':
        #     init_channel = 3
        # elif channel == 'YCbCr':
        #     init_channel = 1
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight)


class ResNet(nn.Module):
    def __init__(self, upscale_factor, channel, residual=False):
        upsample_block_num = int(math.log(upscale_factor, 2))

        super(ResNet, self).__init__()

        self.residual=residual

        c = channel
        self.block1 = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        if self.residual:
            # print('i am residual')
            return torch.clamp(x - (F.tanh(block8) + 1) / 2, 0, 1)
        else:
            # print('i am not')
            return (F.tanh(block8) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))

#----------densenet#
def xavier(param):
    init.xavier_uniform(param)
class SingleLayer(nn.Module):
    def __init__(self, inChannels,growthRate):
        super(SingleLayer, self).__init__()
        self.conv =nn.Conv2d(inChannels,growthRate,kernel_size=3,padding=1, bias=True)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
class SingleBlock(nn.Module):
    def __init__(self, inChannels,growthRate,nDenselayer):
        super(SingleBlock, self).__init__()
        self.block= self._make_dense(inChannels,growthRate, nDenselayer)
        
    def _make_dense(self,inChannels,growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels,growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)
                
    def forward(self, x):
        out=self.block(x)
        return out

class SRDenseNet(nn.Module):
    def __init__(self,inChannels,growthRate,nDenselayer,nBlock):
        super(SRDenseNet,self).__init__()
        
        self.conv1 = nn.Conv2d(3,growthRate,kernel_size=3, padding=1,bias=True)
        
        inChannels = growthRate
        
        self.denseblock = self._make_block(inChannels,growthRate, nDenselayer,nBlock)
        inChannels +=growthRate* nDenselayer*nBlock
        
        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=128, kernel_size=1,padding=0, bias=True)
        
        self.convt1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        
        self.convt2 =nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        
        self.conv2 =nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3,padding=1, bias=True)
    
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         xavier(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
    
    def _make_block(self, inChannels,growthRate, nDenselayer,nBlock):
        blocks =[]
        for i in range(int(nBlock)):
            blocks.append(SingleBlock(inChannels,growthRate,nDenselayer))
            inChannels += growthRate* nDenselayer
        return nn.Sequential(* blocks)  
    
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.denseblock(out)                                      
        out = self.Bottleneck(out)
        out = self.convt1(out)
        out = self.convt2(out)
                                         
        HR = self.conv2(out)
        return HR

if __name__ == '__main__':
    a = torch.randn(3, 1, 10, 10)
    a = Variable(a)
    net = SRDenseNet(16,16,8,8) #default upscale is 4
    # pdb.set_trace()
