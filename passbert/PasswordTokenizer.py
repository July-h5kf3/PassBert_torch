import string
import unicodedata
import re
from snippets import truncate_sequences

def load_vocab(dict_path,encoding = "utf-8",simplified = False,startwith = None):
    """加载词典"""
    token_dict = {}
    with open(dict_path,encoding = encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)
    
    # if simplified:
    #     new_token_dict,keep_tokens = {},[]
    #     startwith = startwith or []
    #     for t in startwith:
    #         new_token_dict[t] = len(new_token_dict)
    #         keep_tokens.append(token_dict[t])

    #     for t,_ in sorted(token_dict.items(),key = lambda s:s[1]):
    #         if t not in new_token_dict and not Tokenizer._is_redundant(t):
    #             new_token_dict[t] = len(new_token_dict)
    #             keep_tokens.append(token_dict[t])
    #     return new_token_dict,keep_tokens
    # else:
    return token_dict

def save_vocab(dict_path,token_dict,encoding = 'utf-8'):
    """保存词典"""
    with open(dict_path,'w',encoding = encoding) as writer:
        for k,v in sorted(token_dict.items(),key = lambda s:s[1]):
            writer.write(k + '\n')
def load_default_pass_vocab(with_space = False):
    LETTERS = string.ascii_letters
    NUMBER = string.digits
    SPECIALS = string.punctuation
    token_dict = {}
    for ch in LETTERS + NUMBER + SPECIALS:
        token_dict[ch] = len(token_dict)
    if with_space:
        token_dict[' '] = len(token_dict)
    for ch in ['[PAD]','[UNK]','[MASK]']:
        token_dict[ch] = len(token_dict)
    return token_dict


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
    
    def encode(
        self,
        first_text,
        second_text = None,
        maxlen = None,
        pattern = "S*E*E",
        truncate_from = "right",
    ):
        if isinstance(first_text,str):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif isinstance(second_text,str):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text
        
        if maxlen is None:
            if truncate_from == "right":
                index = -int(self._token_end is not None) - 1
            elif truncate_from == "left":
                index = int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text is not None and pattern == "S*E*E":
                maxlen += 1
            truncate_sequences(maxlen,index,first_tokens,second_tokens)
        
            
        
        first_tokens_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_tokens_ids)

        if second_text is not None:
            if pattern == "S*E*E":
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_tokens_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        
        return first_tokens_ids,first_segment_ids

    def id_to_token(self,id):
        raise NotImplementedError
    def ids_to_tokens(self,ids):
        return [self.id_to_token(id) for id in ids]
    def decode(self,ids):
        raise NotImplementedError
    def _tokenize(self,text):
        raise NotImplementedError

class PasswordTokenizer(TokenizerBase):
    def __init__(self,word_dict=None,pwd_start = '[S]',pwd_end = '[E]'):
        super(PasswordTokenizer,self).__init__(token_start = pwd_start,token_end = pwd_end,pre_tokenizer=None,token_translate=None)
        if isinstance(word_dict,str):
            word_dict = load_vocab(dict_path=word_dict)
        elif isinstance(word_dict,list):
            word_dict = {v:k for k,v in enumerate(word_dict)}
        elif word_dict is None:
            word_dict = load_default_pass_vocab()
        
        word_dict[pwd_start] = len(word_dict)
        word_dict[pwd_end] = len(word_dict)
        for token in ['pad','unk','mask','start','end']:
            try:
                _attr = getattr(self,'_token_%s' % token)
                _token_id = word_dict[_attr]
                setattr(self,'_token_%s_id' % token,_token_id)
            except:
                print(_attr,"not found")
                pass
        
        self.token_dict = word_dict
        self.token_dict_inv = {v:k for k,v in word_dict.items()}
        self.vocab_size = len(self.token_dict)
        self._word_maxlen = max(map(lambda x:len(x),word_dict.keys()))

    def token_to_id(self, token):
        return self.token_dict.get(token,self.token_dict[self._token_unk])
    
    def id_to_token(self,id):
        return self.token_dict_inv[id]
    
    def _decode_list(self,ids):
        tokens = self._decode_list(ids)
        tokens = tokens[1:-1]
        return "".join(tokens)
    
    def encode_one(self,pwd,maxlen = 10,truncate_from = "left"):
        return super().encode(pwd,None,maxlen=maxlen,truncate_from=truncate_from)[0]
    def _tokenize(self,pwd):
        tokens,start,end = [],0,0
        while start < len(pwd):
            end = len(pwd)
            while end > start:
                sub = pwd[start:end]
                if sub in self.token_dict:
                    break
                end -= 1
            if start == end:
                tokens.append(self._token_unk)
                start += 1
            else:
                tokens.append(sub)
                start = end
        return tokens

    