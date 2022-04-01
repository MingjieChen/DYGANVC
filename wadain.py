import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import math
class EqualLinear(nn.Module):
    
    def __init__(self, dim_in, dim_out, bias = True, bias_init = 0, lr_mul = 1, activation = None):
        
        super().__init__()

        self.weight = nn.Parameter(torch.randn(dim_out, dim_in) , requires_grad = True)

        self.bias = nn.Parameter(torch.zeros(dim_out).fill_(bias_init), requires_grad = True)
        
        self.activation = activation

        self.lr_mul = lr_mul

    def forward(self, x):
        
        out = F.linear(x, self.weight , bias = self.bias * self.lr_mul)
        return out
class WadaIN(nn.Module):
    '''a stylegan2 module'''
    
    def __init__(self, dim_in, dim_out, kernel_size, use_act = True, spk_emb_dim = 128 ):
        
        super().__init__()
        self.use_act = use_act
        if self.use_act:
            self.act = nn.LeakyReLU(0.2)
            self.dim_out = dim_out
        else:
            self.dim_out = dim_out
        self.style_linear = EqualLinear(spk_emb_dim, dim_in, bias_init=1)
        fan_in = dim_in * kernel_size
        self.scale = 1 / math.sqrt(fan_in)
        self.weight = nn.Parameter(torch.randn(1, self.dim_out, dim_in, kernel_size), requires_grad = True)
        if kernel_size %2 ==0:
            self.padding = (kernel_size //2, kernel_size // 2 - 1)
        else:
            self.padding = (kernel_size // 2, kernel_size // 2)    
        self.dim_in = dim_in
        self.kernel_size = kernel_size
    def forward(self, inputs):
        
        x, c_trg = inputs
        batch_size, in_channel, t = x.size()
        

        s = self.style_linear(c_trg).view(batch_size, 1, in_channel, 1)
        # scale weights
        weight = self.weight * s

        # demodulate
        demod = torch.rsqrt(weight.pow(2).sum([2,3]) + 1e-8)
        demod_mean = torch.mean(weight.view(batch_size, self.dim_out, -1), dim = 2)
        weight = weight  * demod.view(batch_size, self.dim_out, 1,1)    

        weight = weight.view(batch_size * self.dim_out, self.dim_in, self.kernel_size)

        x = x.reshape(1, batch_size * in_channel, t)
        
        x = F.pad(x, self.padding, mode = 'reflect')
        out = F.conv1d(x, weight, padding = 0, groups = batch_size) 

        _, _, new_t = out.size()

        out = out.view(batch_size, self.dim_out, new_t) 
        if self.use_act:
        
            out = self.act(out)
        return (out, c_trg)
