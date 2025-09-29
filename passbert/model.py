import imp
import torch
import torch.nn as nn
import copy
import json
from .layers import LayerNorm,MultiHeadAttention,PositionWiseFeedForward,activations

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        dropout_rate,
        embedding_size = None,
        attention_head_size = None,
        attention_key_size = None,
        sequence_length = None,
        keep_tokens = None,
        compound_tokens = None,
        residual_attention_score = False,
        ignore_invalid_weights = False,
    ):
        super(Transformer,self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or self.hidden_size // self.num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0.0
        self.hidden_act = hidden_act
        self.emebdding_size = embedding_size or self.hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.residual_attention_score = residual_attention_score
        self.ignore_invalid_weights = ignore_invalid_weights
    def init_model_weights(self,module):
        raise NotImplementedError
    def variable_mapping(self):
        return {}
    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        # model = self
        state_dict = torch.load(checkpoint, map_location='cpu')
        mapping = mapping or self.variable_mapping()

        for new_key, old_key in mapping.items():
            if old_key in state_dict.keys():
                state_dict[new_key] = state_dict.pop(old_key)
        self.load_state_dict(state_dict, strict=self.ignore_invalid_weights)

def lm_mask(segment_ids):
    """
    定义下三角Attention mask
    """
    idxs = torch.arange(0,segment_ids.shape[1])
    mask = (idxs.unsqueeze(0) <= idxs.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    return mask

def unilm_mask(token_ids,segment_ids):
    """
    定义UniLM的Attention_mask(Seq2Seq模型用)
    """
    ids = segment_ids + (token_ids <= 0).long()
    idxs = torch.cumsum(ids,dim = 1)
    extended_mask = token_ids.unsqueeze(1).unsqueeze(3)
    mask = (idxs.unsqueeze(1) <= idxs.unsqueeze(2)).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)
    mask *= extended_mask
    return mask


