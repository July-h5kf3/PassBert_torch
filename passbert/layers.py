import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def gelu(x):
    """gelu激活函数
    在GPT架构中使用的是gelu函数的近似版本
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)
    
activations = {"gelu":gelu,"swish":swish,"relu":F.relu}

class LayerNorm(nn.Module):
    def __init__(self,hidden_size,eps = 1e-12,conditional = False):
        super(LayerNorm,self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.conditional = conditional
        if conditional:
            self.dense1 = nn.Linear(2 * hidden_size,hidden_size,bias = False)
            self.dense1.weight.uniform_(0,0)
            self.dense2 = nn.Linear(2 * hidden_size,hidden_size,bias = False)
            self.dense2.weight.uniform_(0,0)
    def forward(self,x):
        if self.conditional:
            input = x[0]
            cond = x[1]
            for _ in range(len(input.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim = 1)
            
            u = input.mean(-1,keepdim = True)
            s = (input - u).pow(2).mean(-1,keepdim = True)
            x = (input - u) / torch.sqrt(s + self.eps)
            return (self.weight + self.dense1(cond)) * x + (self.bias + self.dense2(cond))
        else:
            u = x.mean(-1,keepdim = True)
            s = (x - u).pow(2).mean(-1,keepdim = True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight * x + self.bias

class MultiHeadAttention(nn.Module):
    """
    MHA
    """
    def __init__(self,hidden_size,num_attention_heads,dropout_rate,attention_scale = True,
                return_attention_scores = False):
        super(MultiHeadAttention,self).__init__()
        assert hidden_size % num_attention_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores

        self.q = nn.Linear(hidden_size,hidden_size)
        self.k = nn.Linear(hidden_size,hidden_size)
        self.v = nn.Linear(hidden_size,hidden_size)

        self.o = nn.Linear(hidden_size,hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
    
    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(self,query,key,value,attention_mask = None):
        ###query shape [batch_size,seq_len,hidden_size]
        ###key shape [batch_size,seq_len,hidden_size]
        ###value shape [batch_size,seq_len,hidden_size]
        mixed_query_layer = self.q(query)
        mixed_key_layer = self.k(key)
        mixed_value_layer = self.v(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # attention_mask shape: (batch_size, seq_len)
            # attention_scores shape: (batch_size, num_attention_heads, seq_len, seq_len)
            attention_mask = (1.0 - attention_mask) * -10000.0
            # 广播注意力掩码到正确的形状
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim = -1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs,value_layer)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.return_attention_scores:
            return self.o(context_layer),attention_probs
        else:
            return self.o(context_layer)
        
class FlashMHA(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            dropout_rate,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.dropout_rate = dropout_rate
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.o = nn.Linear(hidden_size, hidden_size)
    
    def forward(self,x,attention_mask = None):
        B, T, C = x.shape # Batch size, seqlen, hidden_size
        q, k, v = self.qkv(x).split(self.hidden_size, dim=2)

        q = q.view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = k.view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        if attention_mask is not None:
             attn_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)
        else:
             attn_mask = None
        context = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout_rate if self.training else 0,
            is_causal=False # Set to True for decoder-style causal masking
        )
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o(context)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self,hidden_size,intermediate_size,dropout_rate = 0.5,hidden_act = "gelu",is_dropout = True):
        super(PositionWiseFeedForward,self).__init__()

        self.is_dropout = is_dropout
        self.intermediate_act_fn = activations[hidden_act]
        self.intermediateDense = nn.Linear(hidden_size,intermediate_size)
        self.outputDense = nn.Linear(intermediate_size,hidden_size)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    def forward(self,x):
        if self.is_dropout:
            x = self.dropout(self.intermediate_act_fn(self.intermediateDense(x)))
        else:
            x = self.intermediate_act_fn(self.intermediateDense(x))
        
        x = self.outputDense(x)
        return x