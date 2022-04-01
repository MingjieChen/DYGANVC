import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class DisRes(nn.Module):
    
    """Residual block in Discriminator"""
    def __init__(self, dim_in, dim_out, ):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3,1,1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3,1,1)

        self.learned_sc = dim_in != dim_out
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1,1,0, bias = False)
        self.act = nn.LeakyReLU(0.2)
    def forward(self,x ):
        short_cut = x
        if self.learned_sc:
            short_cut = self.conv1x1(short_cut)
        short_cut = F.avg_pool2d(short_cut, 2)
        
        x = self.act(x)
        x = self.conv1(x)
        x = F.avg_pool2d(x,2)
        x = self.act(x)
        x = self.conv2(x)
        

        out = x + short_cut

        out =  out / math.sqrt(2)
        return out
class Discriminator128(nn.Module):
    def __init__(self, config):
        super(Discriminator128, self).__init__()

        num_speakers = config['num_speakers']
        
        # Initial layers.
        
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        

        self.down_sample_1 = DisRes(64, 128)
        self.down_sample_2 = DisRes(128, 128)
        self.down_sample_3 = DisRes(128, 128)
        self.down_sample_4 = DisRes(128, 128)
        
        
        blocks = []
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(128,128, 5,1,2)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        self.blocks = nn.Sequential(*blocks)            
        self.dis_conv = nn.Conv2d(128, num_speakers, kernel_size = 1, stride = 1, padding = 0 )

    def forward(self, x, c_trg):
        
        x = self.conv_layer_1(x) 

        x = self.down_sample_1(x)
        
        
        x = self.down_sample_2(x)
        
        x = self.down_sample_3(x)
        
        x = self.down_sample_4(x)
        
        x = self.blocks(x)
        

        x = self.dis_conv(x)
        

        b, c, h, w = x.size()
        x = x.view(b,c)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)

        x = x[idx, c_trg.long()]

        return x
