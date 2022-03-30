import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import math

class LightConv(nn.Module):
    
    '''This is a light weight convolution that conducting depthwise convolution for multiple heads.
    
    '''
    def __init__(self, kernel_size, head_size, num_heads):

        super(LightConv, self).__init__()
        self.head_size = head_size

        self.kernel_size = kernel_size
        self.unfold1d = nn.Unfold(kernel_size = [self.kernel_size, 1], padding = [self.kernel_size //2, 0]) 
        self.bias = nn.Parameter(torch.zeros(num_heads * head_size), requires_grad = True)
    def forward(self, x, filters):
        # x: [B,T,C_in]
        # filters: [B,T,num_heads*kernel_size]
        # return: [B,T, num_heads*head_size]
        B,T,_ = x.size()
        conv_kernels = filters.reshape(-1,self.kernel_size,1)
        conv_kernels = torch.softmax(conv_kernels, dim = 1)

        unfold_conv_out = self.unfold1d(x.transpose(1,2).contiguous().unsqueeze(-1))
        unfold_conv_out = unfold_conv_out.transpose(1,2).reshape(B,T,-1,self.kernel_size)

        conv_out = unfold_conv_out.reshape(-1, self.head_size, self.kernel_size)
        conv_out = torch.matmul(conv_out, conv_kernels)
        conv_out = conv_out.reshape(B,T,-1)
        conv_out += self.bias.view(1,1,-1)
        return conv_out
class DynamicConv(nn.Module):   

    def __init__(self, dim_in, dim_out, kernel_size = 3, spk_emb_dim = 128, num_heads = 8, use_kconv = False, wada_kernel = 3, res = True, ln = True, rel_pos = False):
        
        super(DynamicConv, self).__init__()
        
        self.dim_out = dim_out*2
        self.dim_in = dim_in
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.res = res
        self.use_ln = ln
        self.use_kconv = use_kconv
        self.k_layer = nn.Linear(dim_in, self.dim_out)
        if use_kconv:
            self.conv_kernel_layer = nn.Sequential(
                SeparableConv1d(dim_out, dim_out, kernel_size),
                nn.Linear(dim_out, kernel_size*num_heads)
                )
        else:
            self.conv_kernel_layer = nn.Linear(dim_out, kernel_size*num_heads)
        
        self.rel_pos = rel_pos #relative positional encoding, https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/attention.py
        if rel_pos:
            self.linear_pos = nn.Linear(dim_in, dim_in)    
            
        
        
        self.lconv = LightConv(kernel_size, dim_out  // num_heads, num_heads)
        if self.use_ln:
            self.ln = nn.LayerNorm(dim_out)
        self.act = nn.GLU(dim = -1)
    
    def rel_shift(self, x):
        '''compute relative positional encoding.
            https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/attention.py
        '''    
        return
    def forward(self, inputs ):
        tuple_input = False
        if isinstance(inputs, tuple):
            x, spk_trg = inputs
            tuple_input = True
        else:
            x = inputs
        # x: [B,T,Cin]
        # spk_src: we don't use it here
        # spk_trg: [B, spk_emb_dim]
        # return: [B,T,Cout]
        x = x.transpose(1,2)
        B,T,C = x.size()
        residual = x
        if self.use_ln:
            x = self.ln(x)
        k = self.act(self.k_layer(x))

        
        # generate light weight conv kernels 
        
        weights = self.conv_kernel_layer(k) # [B,T, dim_in] -> [B,T,num_heads*kernel_size]
        weights = weights.view(B, T, self.num_heads, self.kernel_size)
        # conduct conv
        layer_out = self.lconv(k, weights) 
        if self.res:
            layer_out = layer_out + residual    
        #if self.use_ln:
        #    layer_out = self.ln(layer_out)
        if tuple_input:
            return (layer_out.transpose(1,2), spk_trg)
        else:
            return layer_out.transpose(1,2)
class SeparableConv1d(nn.Module):
    '''This is a separable conv1d that doing depthwise conv and pointwise conv'''

    def __init__(self, dim_in, dim_out, kernel_size, bias = True):
        
        super(SeparableConv1d,self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size

        self.depthconv = nn.Conv1d(dim_in, dim_in, groups = dim_in, kernel_size = kernel_size, stride = 1, padding = kernel_size // 2, bias = False)
        self.pointconv = nn.Conv1d(dim_in, dim_out, kernel_size = 1, stride = 1, padding = 0, bias = False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_normal_(self.depthconv.weight)
        nn.init.kaiming_normal_(self.pointconv.weight)

    def forward(self, x):
        # x: [B, T, Cin]
        # return: [B, T, Cout]
        out = self.pointconv(self.depthconv(x.transpose(1,2))).transpose(1,2)

        if self.bias is not None:
            out += self.bias
        return out
