import importlib
from turtle import forward
import torch
import torch.nn as nn
import copy
import json
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from passbert.layers import LayerNorm,MultiHeadAttention,PositionWiseFeedForward,activations,FlashMHA
from passbert.PasswordTokenizer import get_char_type_ids

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
        **kwargs
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

class BertEmbeddings(nn.Module):
    """
    embedding层
    构造word,position,token_type embeddings
    """
    def __init__(self,vocab_size,hidden_size,max_position,segment_vocab_size,drop_rate,type_vocab_size = 6):
        super(BertEmbeddings,self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,hidden_size,padding_idx = 0)
        self.position_embeddings = nn.Embedding(max_position,hidden_size,padding_idx = 0)
        self.segment_embeddings = nn.Embedding(segment_vocab_size,hidden_size)
        self.type_embeddings = nn.Embedding(type_vocab_size,hidden_size)

        self.LayerNorm = LayerNorm(hidden_size,eps = 1e-12)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self,token_ids,segment_ids = None):
        seq_len = token_ids.size(1)
        position_ids = torch.arange(seq_len,dtype=torch.long,device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if segment_ids is not None:
            segment_ids = torch.zeros_like(token_ids)
        type_ids = get_char_type_ids(token_ids=token_ids)
        
        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        type_embeddings = self.type_embeddings(type_ids)

        embeddings = words_embeddings + position_embeddings + segment_embeddings + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
class BertLayer(nn.Module):
    """
    Transformer的单层
    顺序为:Attention--> Add --> LayerNorm --> FeedForward --> Add --> LayerNorm
    """
    def __init__(self,hidden_size,num_attention_heads,dropout_rate,intermediate_size,hidden_act,is_dropout = False):
        super(BertLayer,self).__init__()
        self.MHA = MultiHeadAttention(hidden_size,num_attention_heads,dropout_rate)
        # self.MHA = FlashMHA(hidden_size,num_attention_heads,dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNorm(hidden_size,eps = 1e-12)
        self.FFN = PositionWiseFeedForward(hidden_size,intermediate_size,hidden_act,is_dropout = is_dropout)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNorm(hidden_size,eps = 1e-12)
    
    def forward(self,hidden_states,attention_mask):
        self_attn_output = self.MHA(hidden_states,hidden_states,hidden_states,attention_mask)
        # self_attn_output = self.MHA(hidden_states,attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        hidden_states = self.layerNorm1(hidden_states)
        self_attn_output2 = self.FFN(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2(hidden_states)
        return hidden_states

class Bert(Transformer):
    def __init__(
        self,
        max_position,
        segment_vocab_size = 2,
        initializer_range = 0.02,
        with_pool = False,
        with_nsp = False,
        with_mlm = False,
        hierarchical_position = None,
        custom_position_ids = False,
        **kwargs
    ):
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.initializer_range = initializer_range
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        
        super(Bert,self).__init__(**kwargs)

        self.embeddings = BertEmbeddings(self.vocab_size,self.hidden_size,self.max_position,self.segment_vocab_size,self.dropout_rate)
        layer = BertLayer(self.hidden_size,self.num_attention_heads,self.dropout_rate,self.intermediate_size,self.hidden_act,is_dropout = False)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])
        if self.with_pool:
            self.pooler = nn.Linear(self.hidden_size,self.hidden_size)
            self.pooler_activation = nn.Tanh()
            if self.with_nsp:
                self.nsp = nn.Linear(self.hidden_size,2)#这个咱们不需要
        else:
            self.pooler = None
            self.pooler_activation = None
        
        if self.with_mlm:
            self.mlmDecoder = nn.Linear(self.hidden_size,self.vocab_size,bias = False)
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDense = nn.Linear(self.hidden_size,self.hidden_size)
            self.transform_act_fn = activations[self.hidden_act]
            self.mlmLayerNorm = LayerNorm(self.hidden_size,eps = 1e-12)
        self.apply(self.init_model_weights)

    def init_model_weights(self,module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            """
            bert参数初始化,在tf版本中是截断正态分布，torch中使用nn.init.trunc_normal_
            """
            nn.init.trunc_normal_(module.weight,mean = 0.0,std = self.initializer_range,a = -2 * self.initializer_range,b = 2 * self.initializer_range)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module,LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    def forward(self,token_ids,segment_ids = None,attention_mask = None,output_all_encoded_layers = False):

        if attention_mask is None:
            attention_mask  = (token_ids != 0).long().unsqueeze(1).unsqueeze(2)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)
        
        attention_mask = attention_mask.to(dtype = next(self.parameters()).dtype)
        hidden_states = self.embeddings(token_ids,segment_ids)
        encoded_layers =[hidden_states]
        for layer_module in self.encoderLayer:
            hidden_states = layer_module(hidden_states,attention_mask)
            if output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        
        sequence_output = encoded_layers[-1]
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:,0]))
        else:
            pooled_output = None
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        
        if self.with_mlm:
            mlm_hidden_states = self.mlmDense(sequence_output)
            mlm_hidden_states = self.transform_act_fn(mlm_hidden_states)
            mlm_hidden_states = self.mlmLayerNorm(mlm_hidden_states)
            mlm_scores = self.mlmDecoder(mlm_hidden_states)+ self.mlmBias
        else:
            mlm_scores = None
        
        if mlm_scores is not None and nsp_scores is None:
            return mlm_scores
        elif mlm_scores is not None and nsp_scores is not None:
            return mlm_scores,nsp_scores
        elif mlm_scores is not None:
            return mlm_scores
        else:
            return nsp_scores
    def variable_mapping(self):
        mapping = {
            'embeddings.word_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': 'bert.embeddings.segment_embeddings.weight',
            'embeddings.LayerNorm.weight': 'bert.embeddings.LayerNorm.weight',
            'embeddings.LayerNorm.bias': 'bert.embeddings.LayerNorm.bias',
            'pooler.weight': 'bert.pooler.dense.weight',
            'pooler.bias': 'bert.pooler.dense.bias',
            'nsp.weight': 'cls.seq_relationship.weight',
            'nsp.bias': 'cls.seq_relationship.bias',
            'mlmDense.weight': 'cls.predictions.transform.dense.weight',
            'mlmDense.bias': 'cls.predictions.transform.dense.bias',
            'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.bias',
            'mlmBias': 'cls.predictions.bias',
            'mlmDecoder.weight': 'cls.predictions.decoder.weight'
        }
        for i in range(self.num_hidden_layers):
                prefix = 'bert.encoder.layer.%d.' % i
                mapping.update({'encoderLayer.%d.multiHeadAttention.q.weight' % i: prefix + 'attention.self.query.weight',
                                'encoderLayer.%d.multiHeadAttention.q.bias' % i: prefix + 'attention.self.query.bias',
                                'encoderLayer.%d.multiHeadAttention.k.weight' % i: prefix + 'attention.self.key.weight',
                                'encoderLayer.%d.multiHeadAttention.k.bias' % i: prefix + 'attention.self.key.bias',
                                'encoderLayer.%d.multiHeadAttention.v.weight' % i: prefix + 'attention.self.value.weight',
                                'encoderLayer.%d.multiHeadAttention.v.bias' % i: prefix + 'attention.self.value.bias',
                                'encoderLayer.%d.multiHeadAttention.o.weight' % i: prefix + 'attention.output.dense.weight',
                                'encoderLayer.%d.multiHeadAttention.o.bias' % i: prefix + 'attention.output.dense.bias',
                                'encoderLayer.%d.layerNorm1.weight' % i: prefix + 'attention.output.LayerNorm.weight',
                                'encoderLayer.%d.layerNorm1.bias' % i: prefix + 'attention.output.LayerNorm.bias',
                                'encoderLayer.%d.feedForward.intermediateDense.weight' % i: prefix + 'intermediate.dense.weight',
                                'encoderLayer.%d.feedForward.intermediateDense.bias' % i: prefix + 'intermediate.dense.bias',
                                'encoderLayer.%d.feedForward.outputDense.weight' % i: prefix + 'output.dense.weight',
                                'encoderLayer.%d.feedForward.outputDense.bias' % i: prefix + 'output.dense.bias',
                                'encoderLayer.%d.layerNorm2.weight' % i: prefix + 'output.LayerNorm.weight',
                                'encoderLayer.%d.layerNorm2.bias' % i: prefix + 'output.LayerNorm.bias'
                                })

        return mapping

def build_transformer_model(
    config_path = None,
    checkpoint_path = None,
    model = 'bert',
    application = 'encoder',
    **kwargs
):
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)
    models = {
        'bert': Bert,
        'roberta': Bert
    }

    my_model = models[model]
    transformer = my_model(**configs)
    if checkpoint_path is not None:
        transformer.load_weights_from_pytorch_checkpoint(checkpoint_path)
    return transformer

