import string
import unicodedata
import re
from .snippets import truncate_sequences

def load_vocab(dict_path,encoding = "utf-8",simplified = False,startwith = None):
    """加载词典"""
    token_dict = {}
    with open(dict_path,encoding = encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)
    
    if simplified:
        new_token_dict,keep_tokens = {},[]
        startwith = startwith or []
        for t in startwith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t,_ in sorted(token_dict.items(),key = lambda s:s[1]):
            if t not in new_token_dict and not Tokenizer._is_redundant(t):
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])
        return new_token_dict,keep_tokens
    else:
        return token_dict

def save_vocab(dict_path,token_dict,encoding = 'utf-8'):
    """保存词典"""
    with open(dict_path,'w',encoding = encoding) as writer:
        for k,v in sorted(token_dict.items(),key = lambda s:s[1]):
            writer.write(k + '\n')

class TokenizerBase:
    def __init__(self,token_start = '[CLS]',token_end = f'[SEP]',pre_tokenizer = None,token_translate = None):
        """
        参数说明：
        pre_tokenize:外部传入的分词函数，用于对文本进行预分词。
        token_translate:映射字典,主要用在tokenize之后,将某些特殊的token替换为对应的token
        """
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end
        self._pre_tokenizer = pre_tokenizer
        self._token_translate = token_translate or {}
        self._token_translate_inv = {v:k for k,v in self._token_translate.items()}
    
    def tokenize(self,text,maxlen = None):
        tokens = [self._token_translate.get(t,t) for t in self._tokenize(text)]
        if self._token_start is not None:
            tokens.insert(0,self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)
        
        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences(maxlen,-index,tokens)

        return tokens
    
    def token_to_id(self,token):
        raise NotImplementedError
    
    def tokens_to_ids(self,tokens):
        return [self.token_to_id(t) for t in tokens]
    
    def encode(self,first_text,second_text = None,maxlen = None):
        first_tokens = self.tokenize(first_text)