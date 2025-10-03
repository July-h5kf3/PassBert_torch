from inspect import ismemberdescriptor
import os,re,sys
import numpy
from torch.optim import lr_scheduler
from dataset import *

model_saved_path = "D:/Competition/Password/passbert_pytorch/Model_ckpt/passbert_model.ckpt"
bert_model_save_path = "D:/Competition/Password/passbert_pytorch/Model_ckpt/passbert_bert_model_rockyou.ckpt"

sequence_length = 32
batch_size = 32
config_path = "D:/Competition/Password/passbert_pytorch/Model_ckpt/passbert_config.json"
checkpoint_path = None#从零训练就设为None

lr = 0.00176
weight_decay_rate = 0.01
num_warmup_steps = 31250
num_train_steps = 125000
steps_per_epoch = 10000
grad_accum_steps = 1
epochs = num_train_steps * grad_accum_steps // steps_per_epoch
exclude_from_weight_decay = ['Norm','bias']
exclude_from_layer_adaptation = ['Norm','bias']
optimizer = 'AdamW'

lr_scheduler = {
    num_warmup_steps*grad_accum_steps: 1.0,
    3 * num_warmup_steps*grad_accum_steps: 0.9,
    num_train_steps*grad_accum_steps: 0.8,
}

dataset = 







