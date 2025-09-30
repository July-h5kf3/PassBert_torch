import argparse
import imp
import random
import os
import numpy as np
import torch
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
import re
from passbert.tokenizers import PasswordTokenizer
from passbert.snippets import parallel_apply

class TrainingDataset(Dataset):
    def __init__(self,tokenizer,sequence_length = 512,must_concat = False):
        self.tokenizer = tokenizer
        self.seqence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_start_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer._vocab_size
        self.must_concat = must_concat
        self.data = []

    def padding(self,sequence,padding_value = None):
        """
        对单个序列进行padding
        """
        if padding_value is None:
            padding_value = self.token_pad_id
        
        sequence = sequence[:self.seqence_length]
        padding_length = self.seqence_length - len(sequence)
        
        return sequence + [padding_value] * padding_length

    def sentence_process(self,text):
        """
        单个文本的处理函数，返回处理后的instance
        """
        raise NotImplemented
    
    def paragraph_process(self,texts,starts,ends,paddings):
        """单个段落（多个文本）的处理函数
        说明：texts是单句组成的list；starts是每个instance的起始id；
              ends是每个instance的终止id；paddings是每个instance的填充id。
        做法：不断塞句子，直到长度最接近sequence_length，然后padding。

        Comments by @ChuanwangWANG
        Here we append a password as long as the length of the vector is less than seq_len (512 by default)
        the format of the final vector is like:
        [cls] *pwd [cls] *pwd [cls] *pwd [sep]
        where *pwd refers to defold pwd, i.e., pwd = [1, 2, 3] and *pwd = 1, 2, 3
        """
        instances,instance = [],[[start] for start in starts]
        for text in texts:
            sub_instance = self.sentence_process(text)
            q = [i for i in sub_instance]
            sub_instance = [i[:self.seqence_length - 2] for i in sub_instance]
            for _a,_b in zip(q,sub_instance):
                for _i,_j in zip(_a,_b):
                    assert _i == _j
            new_length = len(instaance[0]) + len(sub_instance[0])

            if new_length > self.seqence_length - 1 or not self.must_concat:
                complete_instance = []
                for item,end,pad in zip(instance,ends,paddings):
                    item.append(end)
                    item = self.padding(item,pad)
                    complete_instance.append(item)
                instances.append(complete_instance)
                instance = [[start] for start in starts]

            for item,sub_item in zip(instance,sub_instance):
                item.extend(sub_item)
                if self.must_concat:
                    item.append(self.token_cls_id)

            complete_instance = []
            for item,end,pad in zip(instance,ends,paddings):
                item.append(end)
                item = self.padding(item,pad)
                complete_instance.append(item)
            
            instances.append(complete_instance)

            return instances
    def process_corpus(self,corpus,workers = 8,max_queue_size = 2000):
        """
        处理输入语料，生成训练数据
        """
        def process_paragraphy(texts):
            return self.paragraph_process(texts)
    
        processed_data = parallel_apply(
            func = process_paragraphy,
            iterable=corpus,
            workers = workers,
            max_queue_size = max_queue_size,
        )
        self.data = []
        for batch in processed_data:
            for instance in batch:
                self.data.append(instance)
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

class TrainingDatasetRoBERTa(TrainingDataset):
    """
    预训练数据集生成(RoBERTa模式)
    """
    def __init__(self, tokenizer,word_segment,mask_rate = 0.15,sequence_length=512, must_concat=False):
        super(TrainingDatasetRoBERTa).__init__(tokenizer,sequence_length,must_concat = must_concat)
        self.word_segment = word_segment
        self.mask_rate = mask_rate
    
    def token_process(self,token_id):
        """
        以0.8的概率替换为[MASK],0.1的保持不变,0.1替换为一个随机的Token
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0,self.vocab_size)
    
    def sentence_process(self, text):
        """
        单个文本的处理函数
        流程:分词,然后转id,按照mask_rate构建全词mask的序列
        来指定哪些token是否要被mask
        """
        words = self.word_segment(text)
        rands = np.random.random(len(words))

        token_ids,mask_ids = [],[]
        for rand,word in zip(rands,words):
            word_tokens = self.tokenizer.tokenize(text = word)[1:-1]
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                word_mask_ids = [
                    self.token_process(i) + 1 for i in word_token_ids
                ]
            else:
                word_mask_ids = [0] * len(word_tokens)
            
            mask_ids.extend(word_mask_ids)
        
        return [token_ids,mask_ids]
    def paragraph_process(self,texts):
        starts = [self.token_cls_id,0]
        ends = [self.token_sep_id,0]
        paddings = [self.token_pad_id,0]
        return super(TrainingDatasetRoBERTa,self).paragraph_process(texts,starts,ends,paddings)
    def __getitem__(self, idx):
        """
        返回Pytorch张量格式数据
        """
        instance = self.data[idx]
        token_ids = torch.tensor(instance[0],dtype=torch.long)
        mask_ids = torch.tensor(instance[1],dtype=torch.long)

        segment_ids = torch.zeros_like(token_ids,dtype=torch.long)

        is_masked = (mask_ids != 0).float()

        masked_token_ids = torch.where(is_masked.bool(),mask_ids - 1,token_ids)
        return {
            'input_token':masked_token_ids,
            'input_segment':segment_ids,
            'token_ids':token_ids,
            'is_masked':is_masked
        }
#这个是实际用到的Dataset
class TrainingDatasetCPG(TrainingDataset):
    """
    预训练数据集生成器(RoBERTa模式)
    """    
    def __init__(
            self,tokenizer,word_segment,mask_rate = 0.50,sequence_length = 512
    ):
        super(TrainingDatasetCPG,self).__init__(
            tokenizer,sequence_length
        )
        self.word_segment = word_segment
        self.mask_rate = mask_rate
        print(f"TrainingDatasetCPG record")
    
    def token_process(self,token_id):
        """
        以0.8的概率替换为[MASK],0.1的保持不变,0.1替换为一个随机的Token
        sp:似乎在PassBert的预训练中,不进行这个概率的区分若mask则直接mask,主要是因为在口令这个问题上,单一字符的更改,带来的影响是巨大的
        """
        # rand = np.random.random()
        # if rand <= 0.8:
        return self.token_mask_id
        # elif rand <= 0.9:
            # return token_id
        # else:
            # return np.random.randint(0,self.vocab_size)
    def sentence_process(self, text):
        words = self.word_segment
        rands = np.random.random(len(words))

        token_ids,mask_ids = [],[]
        for rand,word in zip(rands,words):
            word_tokens = self.tokenizer.tokenize(text = word)[1:-1]
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
            #可以通过下面的语句进行调试
            # print(f"[TrainingDatasetRoberta::sentence_process]: word = {word}, word_tokens = {word_tokens}, word_token_ids = {word_token_ids}")
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                word_mask_ids = [
                    self.token_process(i) + 1 for i in word_token_ids
                ]
            else:
                word_mask_ids = [0] * len(word_tokens)
            
            mask_ids.extend(word_mask_ids)
            return [token_ids,mask_ids]
    
    def paragraph_process(self, texts):
        starts = [self.token_cls_id,0]
        ends = [self.token_sep_id,0]
        paddings = [self.token_pad_id,0]
        return super(TrainingDatasetCPG,self).paragraph_process(texts,starts,ends,paddings)
    def __getitem__(self, idx):
        instance = self.data[idx]
        token_ids = torch.tensor(instance[0],dtype=torch.long)
        mask_ids = torch.tensor(instance[1],dtype=torch.long)

        segment_ids = torch.zeros_like(token_ids,dtype=torch.long)
        is_masked = (mask_ids != 0).float()

        masked_token_ids = torch.where(is_masked.bool(),mask_ids - 1,token_ids)

        return {
            'input_token':masked_token_ids,
            'input_segment':segment_ids,
            'token_ids':token_ids,
            'is_masked':is_masked
        }

def warpper(**kwargs):
    model = kwargs['model'] #roberta
    sequence_length = kwargs['sequence_length'] #32
    workers = kwargs['workers']
    training_filename = kwargs['training_filename']
    dup_factor = kwargs['dup_factor']
    num_samples = kwargs['num_samples'] #1000000
    must_concat = kwargs['must_concat']
    tokenizer = PasswordTokenizer()

    def some_texts(training_file = training_filename,dupe_factors = dup_factor,n_samples = num_samples):
        """
        生成密码文本的生成器
        """
        luds = re.compile(r"^[a-zA-Z0-9\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]{4,30}$")

        with open(training_file,'r') as fin:
            lines = []
            for line in fin:
                line = line.strip('\r\n')
                if luds.search(line):
                    lines.append(line)
        if n_samples < len(lines):
            lines = random.sample(lines,n_samples)
            print(f"we will only use {n_samples} passwords for pretraining.",file = sys.stderr)
            pass
        
        for _ in range(dupe_factors):
            random.shuffle(lines)
            count,texts = 0,[]
            for line in lines:
                pwd = line
                texts.append(pwd)
                count += 1
                if count == 1000:
                    yield texts
                    count,texts = 0,[]
            if texts:
                yield texts
            pass
        
        pass
    
    assert model in ['roberta']

    if model == 'roberta':
        def word_segment(text):
            return list(text)
        
        TD = TrainingDatasetRoBERTa(
            tokenizer,word_segment,sequence_length = sequence_length,must_concat=must_concat
        )
        TD.process_corpus(
            corpus=tqdm(some_texts()),
            workers=workers
        )
    return TD

if __name__ == '__main__':
    pass
    # args = argparse.ArgumentParser('Data utils:generating record file')
    # args.add_argument('-i','--input',dest='training',type=str)
    
