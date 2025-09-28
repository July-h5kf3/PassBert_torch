import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu_erf(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def set_gelu(version = "erf"):
    version = version.lower()
    assert version in ["erf", "tanh"], "Invalid GELU version"
    return gelu_erf if version == "erf" else gelu_tanh

def piecewise_linear(t,schedule:dict):
    """分段线性函数
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示 t ∈ [0, 1000]时，输出从0均匀增加至1，而
    t ∈ [1000, 2000]时，输出从1均匀降低到0.1，最后
    t > 2000时，保持0.1不变。
    """
    items = sorted(schedule.items())
    if items[0][0] != 0:
        items = [(0,0.0)] + items
    
    t = torch.as_tensor(t,dtype=torch.float32)
    x = torch.tensor(items[0][1],dtype=torch.float32)

    for i in range(len(items)):
        t_begin,y_begin = items[i]
        if i != len(items) - 1:
            t_end,y_end = items[i + 1]
            slope = (y_end - y_begin) / (t_end - t_begin)
            y = y_begin + slope * (t - t_begin)
        else:
            y = torch.tensor(y_begin,dtype=torch.float32)
        
        x = torch.where(t >= t_begin,y,x)
    return x
def sequence_mask(x,mask,value=0.0,axis = 1):
    """
    mask序列
    x:(batch_size,seq_len,*)
    mask:(batch_size,seq_len)
    """
    if mask is None:
        return x
    mask = mask.to(dtype=x.dtype)
    if value == "-inf":
        value = -1e12
    elif value == "inf":
        value = 1e12
    
    for _ in range(x.dim() - mask.dim()):
        mask = mask.unsqueeze(-1)
    
    return x * mask + value * (1 - mask)

def batch_gather(params,indice):
    """
    params:(batch_size,seq_len,*)
    indice:(batch_size,num)
    output:(batch_size,num,*)
    """
    batch_size = params.size(0)
    batch_idx = torch.arange(batch_size).unsqueeze(1).to(indice.device)
    return params[batch_idx,indice]

def pool1d(x,pool_size,stride=1,padding=0,mode = "max"):
    """1d池化"""
    x = x.unsqueeze(1)
    if mode == "max":
        x = F.max_pool1d(x,kernel_size=(1,pool_size),stride=(1,stride),padding=(0,padding))
    elif mode == "avg":
        x = F.avg_pool1d(x,kernel_size=(1,pool_size),stride=(1,stride),padding=(0,padding))
    else:
        raise ValueError(f"Invalid pooling mode: {mode}")
    
    return x[:,0]

def divisible_temporal_padding(x,n):
    seq_len = x.size(1)
    pad_len = (n - seq_len % n) % n
    return F.pad(x,pad=(0,0,0,pad_len))

def swish(x):
    return x * torch.sigmoid(x)

def leaky_relu(x,alpha = 0.2):
    return F.leaky_relu(x,negative_slope=alpha)

class Sinusoidal(nn.Module):
    def __init__(self,vocab_size,depth):
        super().__init__()
        embeddings = torch.zero(vocab_size,depth)
        for pos in range(vocab_size):
            for i in range(depth // 2):
                theta = pos / (10000 ** (2.0 * i / depth))
                embeddings[pos,2*i] = torch.sin(theta)
                embeddings[pos,2*i+1] = torch.cos(theta)
        self.embeddings = nn.Parameter(embeddings,requires_grad=False)
    def forward(self,x):
        return self.embeddings[x]

def multiable_categorical_crossentropy(y_true,y_pred):
    """多标签分类的交叉熵损失"""
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss