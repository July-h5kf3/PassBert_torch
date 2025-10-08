import json
from multiprocessing import process, Pool, Manager
import re
import random
import sys
from tkinter import W
import numpy as np
import math
from tqdm import tqdm
import argparse
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from passbert.PasswordTokenizer import PasswordTokenizer

class DataProcessor:
    def __init__(self,tokenizer,sequence_length = 512,must_concat = False):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_start_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer.vocab_size
        self.must_concat = must_concat
    def padding(self,sequence,padding_value = None):
        """
        对单个序列进行padding
        """
        if padding_value is None:
            padding_value = self.token_pad_id
        
        sequence = sequence[:self.sequence_length]
        padding_length = self.sequence_length - len(sequence)
        
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
            # 处理单个句子
            # print(f"text = {text}")
            sub_instance = self.sentence_process(text)
            # print(f"sub_instance = {sub_instance}")
            q = [i for i in sub_instance]
            sub_instance = [i[:self.sequence_length - 2] for i in sub_instance]
            for _a, _b in zip(q, sub_instance):
                for _i, _j in zip(_a, _b):
                    assert _i == _j
            new_length = len(instance[0]) + len(sub_instance[0])

            # 如果长度即将溢出
            # Note that here we set each sequence contains only one password
            # if new_length > 0:
            if new_length > self.sequence_length - 1 or not self.must_concat:
                # 插入终止符，并padding
                complete_instance = []
                for item, end, pad in zip(instance, ends, paddings):
                    item.append(end)
                    item = self.padding(item, pad)
                    complete_instance.append(item)
                # 存储结果，并构建新样本
                # print(f"complete_instance = {complete_instance}")
                instances.append(complete_instance)
                instance = [[start] for start in starts]

            # 样本续接
            for item, sub_item in zip(instance, sub_instance):
                item.extend(sub_item)
                if self.must_concat:
                    item.append(self.token_cls_id)

        # 插入终止符，并padding
        complete_instance = []
        for item, end, pad in zip(instance, ends, paddings):
            item.append(end)
            item = self.padding(item, pad)
            complete_instance.append(item)

        # 存储最后的instance
        instances.append(complete_instance)
        # print("last instances = ", instances)
        return instances
            

class CPGDataProcessor(DataProcessor):
    def __init__(self,tokenizer,word_segment,mask_rate = 0.5,sequence_length = 512,mask_strategy = "WWM"):
        """
        新增加了Span Mask的掩词方式，这种方式会连续遮掩，后续考虑增加根据PCFG遮掩的方式
        """
        super().__init__(tokenizer,sequence_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate
        self.mask_strategy = mask_strategy
        if mask_strategy == "SM":
            self.max_span_len = 4
        # print(f"CPGDataProcessor record")
    def token_process(self,token_id):
        """
        以0.8的概率替换为[MASK],0.1的保持不变,0.1替换为一个随机的Token
        sp:似乎在PassBert的预训练中,不进行这个概率的区分若mask则直接mask,主要是因为在口令这个问题上,单一字符的更改,带来的影响是巨大的
        """
        return self.token_mask_id
    def sentence_process(self, text):
        """单条口令的mask逻辑"""
        words = self.word_segment(text)
        n = len(words)

        mask_flags = np.zeros(n, dtype=bool)

        if self.mask_strategy == "WWM":
            # ---------- Whole Word Masking ----------
            rands = np.random.random(n)
            for i in range(n):
                if rands[i] < self.mask_rate:
                    mask_flags[i] = True

        elif self.mask_strategy == "SM":
            num_to_mask = math.ceil(n * self.mask_rate)
            if num_to_mask > 0:
                masked_count = 0
                candidate_indices = list(range(n))
                random.shuffle(candidate_indices)

                for start_index in candidate_indices:
                    if mask_flags[start_index]:
                        continue
                    span_len = random.randint(1, self.max_span_len)
                    actual_span_len = 0
                    for i in range(start_index, min(start_index + span_len, n)):
                        if not mask_flags[i]:
                            mask_flags[i] = True
                            actual_span_len += 1
                    masked_count += actual_span_len
                    if masked_count >= num_to_mask:
                        break

        else:
            raise ValueError(f"Unsupported mask strategy: {self.mask_strategy}")

        # ---------- 生成 token_ids / mask_ids ----------
        token_ids, mask_ids = [], []
        for word, is_mask in zip(words, mask_flags):
            word_tokens = self.tokenizer.tokenize(word)[1:-1]
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)

            if is_mask:
                word_mask_ids = [self.token_process(i) for i in word_token_ids]
            else:
                word_mask_ids = [0] * len(word_token_ids)
            mask_ids.extend(word_mask_ids)

        return [token_ids, mask_ids]
    def paragraph_process(self, texts):
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        instances_list =  super().paragraph_process(texts, starts, ends, paddings)
        dict_instances = []
        for instance in instances_list:
            if instance and len(instance) == 2:
                dict_instances.append({
                    'token_ids':instance[0],
                    'mask_ids':instance[1]
                })
        return dict_instances

def write_to_jsonl(dict_instance, output_file, total_written):
    """写入JSONL格式的数据"""
    for instance in dict_instance:
        output_file.write(json.dumps(instance) + '\n')
    total_written[0] += len(dict_instance)

def process_texts(args):
    """处理一批文本"""
    texts_batch, processor = args
    dict_instances = processor.paragraph_process(texts_batch)
    return dict_instances

def some_texts(training_file, dupe_factors, n_samples):
    """生成密码文本批次"""
    luds = re.compile(r"^[a-zA-Z0-9\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]{4,30}$")
    with open(training_file, 'r', encoding='utf-8', errors='ignore') as fin:
       lines = [line.strip('\r\n') for line in fin if luds.search(line.strip('\r\n'))]
    if n_samples < len(lines):
        lines = random.sample(lines, n_samples)
        print(f"we will only use {n_samples} passwords for pretraining.", file=sys.stderr)
    # print(f"lines = {lines}")
    for _ in range(dupe_factors):
        random.shuffle(lines)
        batch_size = 1000
        for i in range(0,len(lines),batch_size):
            yield lines[i:i+batch_size]
def word_segment(text):
    """单词分割函数"""
    return list(text)

processor_instance = None

def init_worker(processor,processor_args):
    global processor_instance
    tokenizer = PasswordTokenizer()
    processor_instance = processor(tokenizer = tokenizer,**processor_args)

def worker_process_texts(texts_batch):
    global processor_instance
    return processor_instance.paragraph_process(texts_batch)

def main(args):
    tokenizer = PasswordTokenizer()
    processor_args = {
        'word_segment':word_segment,
        'sequence_length':args.seq_len,
        'mask_strategy':args.mask_strategy,
        # 'must_concat':args.must_concat,
    }
    if args.model == 'roberta' or args.model == 'cpg':
       processor = CPGDataProcessor
    else:
        raise ValueError(f"model {args.model} is not supported.")
    with open(args.training,'r',encoding='utf-8',errors='ignore') as f:
        num_lines = sum(1 for _ in f)
    total_batches = (num_lines // 1000 + 1) * args.dup_factor
    corpus_generator = some_texts(args.training, args.dup_factor, args.num_samples)

    with open(args.save_record,'w',encoding='utf-8') as output_file, Pool(processes=args.workers, initializer=init_worker, initargs=(processor, processor_args)) as pool:
        total_written = 0
        pbar = tqdm(pool.imap_unordered(worker_process_texts,corpus_generator),total = total_batches,desc="Processing")
        for dict_instances_batch in pbar:
            for instance in dict_instances_batch:
                output_file.write(json.dumps(instance) + '\n')
            total_written += len(dict_instances_batch)
            pbar.set_description(f"Processed {total_written} instances")
    print(f"Preprocessing finished. Total instances written: {total_written}")
    

if __name__ == "__main__":
    cli = argparse.ArgumentParser('Data utils: generating pre-training file for PyTorch')
    cli.add_argument('-i', '--input', dest='training', default="E:/pretrain/dataset/Rockyou.txt",type=str)
    cli.add_argument('-m', '--model', dest='model', default="roberta", type=str, required=True, choices=['roberta', 'cpg'])
    cli.add_argument('-l', '--seq-len', dest='seq_len', default=512, type=int)
    cli.add_argument("-w", '--workers', type=int, dest='workers', default=8)
    cli.add_argument("-d", '--dup-factor', type=int, dest='dup_factor', default=1)
    cli.add_argument('-n', '--num-samples', type=int, default=1000000000)
    cli.add_argument('-s', '--save-record', required=True, type=str, help='save processed data to a .jsonl file')
    cli.add_argument('-a', '--mask-strategy',required=True,default="SM",type=str,choices=['SM','WWM'])
    cli.add_argument('--must-concat', action='store_true')
    args = cli.parse_args()
    main(args)