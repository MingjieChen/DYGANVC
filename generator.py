import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F
from .wadain import WadaIN, ScaledWadaIN
from .dynamic_conv import DynamicConv 
import math
from .adain import AdaIN
from .lightweight_conv import LightweightConvModule
class IDBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x    
class DownSample(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x,   c_trg = x
        x = x
        return (F.avg_pool1d(x, 2),   c_trg)


class UpSample(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x,   c_trg = x
        x = x
        return (F.interpolate(x, scale_factor=2, mode='nearest'),   c_trg)

class FF(nn.Module):
    '''feedforward block for dynamic conv'''
    def __init__(self, dim_in, dim_hid, dim_out, kernel = 1, res = True, use_ln = True, stride = 1 ):
        super().__init__()
        self.conv1 = nn.Conv1d(dim_in, dim_hid, kernel_size = kernel, stride = 1, padding = kernel // 2)
        #self.relu = nn.LeakyReLU(0.2)
        #self.relu = nn.ReLU()
        self.act = Swish()
        self.conv2 = nn.Conv1d(dim_hid, dim_out, kernel_size = kernel, stride = stride, padding = kernel // 2)
        self.res = res
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(dim_in)
        if dim_out != dim_in:
            self.learned_sc = True
            self.short_cut = nn.Conv1d(dim_in, dim_out, 1,1,0)
        else:
            self.learned_sc = False    
    def forward(self, x):
        if isinstance(x, tuple):
            x,   c_trg = x
            tuple_input = True
        else:
            tuple_input = False
        residual = x
        if self.use_ln:
            x = self.ln(x.transpose(1,2)).transpose(1,2).contiguous()
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.res:
            if self.learned_sc:
                x += self.short_cut(residual)
            else:
                x += residual    
        x = x / math.sqrt(2)    
        if tuple_input:    
            return (x,   c_trg)    
        else:
            return x    
class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, x):
        if isinstance(x, tuple):
            x,   c_trg = x
            return (x * torch.sigmoid(x),   c_trg)
        else:
            return x * torch.sigmoid(x)        
class AdainFF(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, kernel1 = 1, kernel2 = 1,  res = False, use_ln = False, spk_emb_dim = 128, use_act2 = False, beta = True, *args, **kwargs):
        super().__init__()
        #self.conv1 = WadaIN(dim_in, dim_hid, kernel1, spk_emb_dim = spk_emb_dim, use_act = True, beta = beta)
        #self.conv1 = AdaIN(dim_hid, dim_out, kernel_size = kernel1, style_num = spk_emb_dim)
        self.conv1 = nn.Conv1d(dim_in, dim_hid, kernel1, 1, kernel1//2)
        self.conv2 = AdaIN(dim_hid, dim_out, kernel_size = kernel2, style_num = spk_emb_dim)
        self.use_ln = use_ln
        self.res = res
        self.act = nn.LeakyReLU(0.2)
        if dim_out != dim_in and res:
            self.learned_sc = True
            self.short_cut = nn.Conv1d(dim_in, dim_out, 1,1,0)
        else:
            self.learned_sc = False    
        if use_ln:
            self.ln = nn.LayerNorm(dim_in)
    def forward(self, x):

        inputs,   c_trg = x
        residual = inputs

        if self.use_ln:
            out = self.ln(inputs.transpose(1,2)).transpose(1,2).contiguous()
        else:
            out = inputs.contiguous()
        out  = self.conv1(out)
        out = self.act(out)
        out,_  = self.conv2((out,   c_trg))
        out = self.act(out)
        if self.res:
            #out += residual
            #out += self.short_cut((residual, c_trg))[0]
            if self.learned_sc:
                out += self.short_cut(residual)
            else:
                out += residual    
            #out = out / math.sqrt(2)    
        return (out,   c_trg)
class DoubleWadainFF(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, kernel1 = 1, kernel2 = 1,  res = False, use_ln = False, spk_emb_dim = 128, use_act2 = False, beta = True, *args, **kwargs):
        super().__init__()
        self.conv1 = WadaIN(dim_in, dim_hid, kernel1, spk_emb_dim = spk_emb_dim, use_act = True, beta = beta)
        #self.conv1 = nn.Conv1d(dim_in, dim_hid, kernel1, 1, kernel1 // 2)
        self.conv2 = WadaIN(dim_hid, dim_out, kernel2, use_act = use_act2, spk_emb_dim = spk_emb_dim, beta = beta)
        #self.conv2 = nn.Conv1d(dim_hid, dim_out, kernel2, 1, kernel2 // 2)
        self.use_ln = use_ln
        self.res = res
        self.act = nn.ReLU()
        if dim_out != dim_in and res:
            self.learned_sc = True
            self.short_cut = nn.Conv1d(dim_in, dim_out, 1,1,0)
        else:
            self.learned_sc = False    
        if use_ln:
            self.ln = nn.LayerNorm(dim_in)
    def forward(self, x):

        inputs,   c_trg = x
        residual = inputs

        if self.use_ln:
            out = self.ln(inputs.transpose(1,2)).transpose(1,2).contiguous()
        else:
            out = inputs.contiguous()
        out,_  = self.conv1((out,   c_trg))
        #out = self.conv2(out)
        out,_  = self.conv2((out,   c_trg))
        #out = self.act(out)
        if self.res:
            #out += residual
            #out += self.short_cut((residual, c_trg))[0]
            if self.learned_sc:
                out += self.short_cut(residual)
            else:
                out += residual    
            #out = out / math.sqrt(2)    
        return (out,   c_trg)
class WadainFF1(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, kernel1 = 1, kernel2 = 1,  res = False, use_ln = False, spk_emb_dim = 128, use_act2 = False, beta = True, *args, **kwargs):
        super().__init__()
        self.conv1 = WadaIN(dim_in, dim_hid, kernel1, spk_emb_dim = spk_emb_dim, use_act = True, beta = beta)
        #self.conv1 = nn.Conv1d(dim_in, dim_hid, kernel1, 1, kernel1 // 2)
        #self.conv2 = WadaIN(dim_hid, dim_out, kernel2, use_act = use_act2, spk_emb_dim = spk_emb_dim, beta = beta)
        self.conv2 = nn.Conv1d(dim_hid, dim_out, kernel2, 1, kernel2 // 2)
        self.use_ln = use_ln
        self.res = res
        self.act = nn.ReLU()
        if dim_out != dim_in and res:
            self.learned_sc = True
            self.short_cut = nn.Conv1d(dim_in, dim_out, 1,1,0)
        else:
            self.learned_sc = False    
        if use_ln:
            self.ln = nn.LayerNorm(dim_in)
    def forward(self, x):

        inputs,   c_trg = x
        residual = inputs

        if self.use_ln:
            out = self.ln(inputs.transpose(1,2)).transpose(1,2).contiguous()
        else:
            out = inputs.contiguous()
        out,_  = self.conv1((out,   c_trg))
        #out = self.conv1(out)
        #out = self.act(out)
        #out,_  = self.conv2((out,   c_trg))
        out = self.conv2(out)
        if self.res:
            #out += residual
            #out += self.short_cut((residual, c_trg))[0]
            if self.learned_sc:
                out += self.short_cut(residual)
            else:
                out += residual    
            #out = out / math.sqrt(2)    
        return (out,   c_trg)
class WadainFF(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, kernel1 = 1, kernel2 = 1,  res = False, use_ln = False, spk_emb_dim = 128, use_act2 = False, beta = True, *args, **kwargs):
        super().__init__()
        #self.conv1 = WadaIN(dim_in, dim_hid, kernel1, spk_emb_dim = spk_emb_dim, use_act = True, beta = beta)
        self.conv1 = nn.Conv1d(dim_in, dim_hid, kernel1, 1, kernel1 // 2)
        self.conv2 = WadaIN(dim_hid, dim_out, kernel2, use_act = use_act2, spk_emb_dim = spk_emb_dim, beta = beta)
        #self.conv2 = nn.Conv1d(dim_hid, dim_out, kernel2, 1, kernel2 // 2)
        self.use_ln = use_ln
        self.res = res
        self.act = nn.ReLU()
        if dim_out != dim_in and res:
            self.learned_sc = True
            self.short_cut = nn.Conv1d(dim_in, dim_out, 1,1,0)
        else:
            self.learned_sc = False    
        if use_ln:
            self.ln = nn.LayerNorm(dim_in)
    def forward(self, x):

        inputs,   c_trg = x
        residual = inputs

        if self.use_ln:
            out = self.ln(inputs.transpose(1,2)).transpose(1,2).contiguous()
        else:
            out = inputs.contiguous()
        #out,_  = self.conv1((out,   c_trg))
        out = self.conv1(out)
        out = self.act(out)
        out,_  = self.conv2((out,   c_trg))
        if self.res:
            #out += residual
            #out += self.short_cut((residual, c_trg))[0]
            if self.learned_sc:
                out += self.short_cut(residual)
            else:
                out += residual    
            #out = out / math.sqrt(2)    
        return (out,   c_trg)

class ScaledWadainFF(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, kernel1 = 1, kernel2 = 1,  res = False, use_ln = False, spk_emb_dim = 128, use_act2 = False, beta = True, scale= 0.1, *args):
        super().__init__()
        #self.conv1 = WadaIN(dim_in, dim_hid, kernel1, spk_emb_dim = spk_emb_dim, use_act = True, beta = beta)
        self.conv1 = nn.Conv1d(dim_in, dim_hid, kernel1, 1, kernel1 // 2)
        self.conv2 = ScaledWadaIN(dim_hid, dim_out, kernel2, use_act = use_act2, spk_emb_dim = spk_emb_dim, beta = beta, scale=scale)
        self.use_ln = use_ln
        self.res = res
        self.act = nn.ReLU()
        if dim_out != dim_in and res:
            self.learned_sc = True
            self.short_cut = nn.Conv1d(dim_in, dim_out, 1,1,0)
        else:
            self.learned_sc = False    
        if use_ln:
            self.ln = nn.LayerNorm(dim_in)
    def forward(self, x):

        inputs,   c_trg = x
        residual = inputs

        if self.use_ln:
            out = self.ln(inputs.transpose(1,2)).transpose(1,2).contiguous()
        else:
            out = inputs.contiguous()
        #out,_  = self.conv1((out,   c_trg))
        out = self.conv1(out)
        out = self.act(out)
        out,_  = self.conv2((out,   c_trg))
        if self.res:
            #out += residual
            #out += self.short_cut((residual, c_trg))[0]
            if self.learned_sc:
                out += self.short_cut(residual)
            else:
                out += residual    
            #out = out / math.sqrt(2)    
        return (out,   c_trg)


class Generator0(nn.Module):
    """Generator network."""
    def __init__(self,config):
        super(Generator0, self).__init__()
        # Down-sampling layers
        
        in_feat_dim=config['in_feat_dim']

        in_out_dim = config['hidden_size'].split('_')[0]
        self.conv1 = nn.Sequential( 
                                    nn.Conv1d(in_feat_dim, int(in_out_dim), 5, 1, 2),
                                    nn.LeakyReLU(0.2),
                                    #nn.Conv1d(int(in_out_dim), int(in_out_dim), 5,1,2)
                                    )
        # res blocks
        res_kernels = config['kernel']
        res_heads = config['num_heads']
        res_wff_kernel1 = config['res_wff_kernel1']
        res_wff_kernel2 = config['res_wff_kernel2']
        res_dim_in=config['hidden_size']
        num_res_blocks = config['num_res_blocks']
        use_kconv = config['use_kconv']
        res_use_ln = config['res_use_ln']
        res_wff_use_act2 = config['res_wff_use_act2']
        res_wff_use_res = config['res_wff_use_res']
        res_wadain_use_ln = config['res_wadain_use_ln']
        spk_emb_dim = config['spk_emb_dim']
        wadain_beta = config['wadain_beta']
        if 'ff_block' in config:
            ff_block = config['ff_block']
        else:
            ff_block = 'WadainFF'   
        if 'scale' in config:
            scale = config['scale']
        else:
            scale = 1    
        if 'use_final_in' in config:
            use_final_in = config['use_final_in']
        else:
            use_final_in = True      
        
        if 'conv_block' in config:
            conv_block = config['conv_block']
        else:
            conv_block = 'DynamicConv'                  
        
        res_blocks = []
        for ind  in range(num_res_blocks):
            dim_in = int(res_dim_in.split('_')[ind])
            dim_hid = int(dim_in * config['hid2_factor'])
            if ind < num_res_blocks -1:
                dim_out = int(res_dim_in.split('_')[ind+1])
            else:
                dim_out = dim_in    
            _kernel = int(res_kernels.split('_')[ind])
            _heads = int(res_heads.split('_')[ind])
            res_blocks += [eval(conv_block)(dim_in,dim_in, kernel_size = _kernel, num_heads = _heads, use_kconv = use_kconv, ln = res_use_ln)]
            res_blocks += [eval(ff_block)(dim_in,dim_hid, dim_out,  use_ln = res_wadain_use_ln, kernel1 = res_wff_kernel1, kernel2 = res_wff_kernel2,  use_act2 = res_wff_use_act2, spk_emb_dim = spk_emb_dim, res = res_wff_use_res, beta = wadain_beta, scale = scale )]
        self.res_blocks = nn.Sequential(*res_blocks)    

        
        # Out.
        out_in_dim = config['hidden_size'].split('_')[-1]
        out_feat_dim = config['out_feat_dim']
        if 'out_kernel' in config:
            out_kernel = config['out_kernel']
        else:
            out_kernel = 1    
        #self.out = nn.Conv1d(in_channels=int(out_in_dim), out_channels=out_feat_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        out = []
        out += [nn.Conv1d(int(out_in_dim), int(out_feat_dim), out_kernel, 1, out_kernel // 2)]
        
        #out += [nn.Conv1d(int(out_in_dim), int(out_in_dim)//2, 5, 1, 2),]
        #out += [nn.LeakyReLU(0.2)]
        
        #if use_final_in: 
        #    out += [nn.InstanceNorm1d(int(out_in_dim)//2, affine= True)]
        #    out += [nn.Conv1d(int(out_in_dim)//2, int(out_feat_dim), 5,1,2)]
        #else:
        #    out += [nn.Conv1d(int(out_in_dim)//2, int(out_feat_dim), 5,1,2)]
        self.out = nn.Sequential(*out) 
                                    
        ''''
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)    
            elif isinstance(layer, nn.Conv1d):
                torch.nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        '''                
    def forward(self, x, c_trg, vqs):
        #x = x.squeeze(1)
        vqs = vqs.squeeze(1)
        B = vqs.size(0)
        #x = torch.cat([x,vqs], dim = 1)
        x = self.conv1(vqs)
        #B = x.size(0)
        x = (x, c_trg)

        x = self.res_blocks(x)
        x,_ = x
        x = self.out(x)
        x = x.unsqueeze(1)
        return x

