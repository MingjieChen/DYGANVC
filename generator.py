import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F
from wadain import WadaIN 
from dynamic_conv import DynamicConv 
import math

class WadainFF(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, kernel1 = 1, kernel2 = 1,  res = False, use_ln = False, spk_emb_dim = 128, use_act2 = False, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(dim_in, dim_hid, kernel1, 1, kernel1 // 2)
        self.conv2 = WadaIN(dim_hid, dim_out, kernel2, use_act = use_act2, spk_emb_dim = spk_emb_dim)
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
        out = self.conv1(out)
        out = self.act(out)
        out,_  = self.conv2((out,   c_trg))
        if self.res:
            if self.learned_sc:
                out += self.short_cut(residual)
            else:
                out += residual    
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
        ff_block = config['ff_block']
        conv_block = config['conv_block']
        
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
            res_blocks += [eval(conv_block)(dim_in,dim_in, kernel_size = _kernel, num_heads = _heads, ln = res_use_ln)]
            res_blocks += [eval(ff_block)(dim_in,dim_hid, dim_out,  use_ln = res_wadain_use_ln, kernel1 = res_wff_kernel1, kernel2 = res_wff_kernel2,  use_act2 = res_wff_use_act2, spk_emb_dim = spk_emb_dim, res = res_wff_use_res)]
        self.res_blocks = nn.Sequential(*res_blocks)    

        
        # Out.
        out_in_dim = config['hidden_size'].split('_')[-1]
        out_feat_dim = config['out_feat_dim']
        out_kernel = config['out_kernel']
        
        out = []
        out += [nn.Conv1d(int(out_in_dim), int(out_feat_dim), out_kernel, 1, out_kernel // 2)]
        
        self.out = nn.Sequential(*out) 
                                    
    def forward(self, c_trg, vqs):
        vqs = vqs.squeeze(1)
        B = vqs.size(0)
        x = self.conv1(vqs)
        
        x = (x, c_trg)

        x = self.res_blocks(x)
        x,_ = x
        x = self.out(x)
        x = x.unsqueeze(1)
        return x

