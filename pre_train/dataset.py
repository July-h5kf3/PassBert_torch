import argparse
import imp
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from passbert.tokenizers import PasswordTokenizer
from passbert.snippets import parallel_apply

class RobertaDataset(Dataset):
    def __init__(self,corpus,tokenizer,seq_len = 32,mask_rate = 0.15):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_rate = mask_rate

        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_cls_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer._vocab_size

    def __len__(self):
        return len(self.corpus)

    def _padding(self,seq,pad_value):
        seq = seq[:self.seq_len]
        return seq + [pad_value] * (self.seq_len - len(seq))
    def _token_process(self,token_id):
        rand = np.random.random()

        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0,self.vocab_size)
    
    def __getitem__(self,idx):
        words = self.corpus[idx]

        token_ids,mask_ids = [],[]
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)[1:-1]
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)
            
            if np.random.rand() < self.mask_rate:
                mask_ids.extend([self._token_process(t) + 1 for t in word_token_ids])
            else:
                mask_ids.extend([0] * len(word_token_ids))
        
        token_ids = [self.token_cls_id] + token_ids + [self.token_sep_id]
        mask_ids = [0] + mask_ids + [0]

        token_ids = self._padding(token_ids,self.token_pad_id)
        mask_ids = self._padding(mask_ids,0)

        segment_ids = torch.zeros_like(token_ids)
        is_masked = mask_ids != 0
        masked_token_ids = torch.where(is_masked,mask_ids - 1,token_ids)
        x = {
            'Input-Token':masked_token_ids,
            'Input-Segment':segment_ids,
            'token_ids':token_ids,
            'is_masked':is_masked.float(),
        }
        y = {
            'mlm_loss':torch.zeros(1),
            'mlm_acc':torch.zeros(1),
        }
        return x,y