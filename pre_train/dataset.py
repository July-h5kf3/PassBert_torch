import json
import torch
from torch.utils.data import Dataset,DataLoader
import sys
import os
from functools import partial

class TrainingDataset(Dataset):
    def __init__(self,file_path):
        self.file_path = file_path
        with open(self.file_path,'r',encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, index):
        line = self.lines[index]
        data = json.loads(line)
        return {
            'token_ids':data['token_ids'],
            'mask_ids' :data['mask_ids']
        }

def mlm_collate_fn(batch,pad_token_id):
    batch_token_ids = [item['token_ids'] for item in batch]
    batch_mask_ids = [item['mask_ids'] for item in batch]

    token_ids = torch.tensor(batch_token_ids,dtype=torch.long)
    mask_ids  = torch.tensor(batch_mask_ids,dtype=torch.long)

    Input_Segment = torch.zeros_like(token_ids)

    is_masked = (mask_ids != 0)

    Input_token = torch.where(is_masked,mask_ids - 1,token_ids)

    attention_mask = (token_ids != pad_token_id).long()

    return{
        'Input-Token':Input_token,
        'Input-Segment':Input_Segment,
        'token_ids':token_ids,
        'is_masked':is_masked.float(),
        'attention_mask':attention_mask,#pytorch
    }

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from passbert.PasswordTokenizer import PasswordTokenizer

    processed_file = 'train_data.jsonl'
    tokenizer = PasswordTokenizer()
    pad_token_id = tokenizer._token_pad_id
    
    dataset = TrainingDataset(file_path=processed_file)

    collate_fn = partial(mlm_collate_fn,pad_token_id = pad_token_id)

    batch_size = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    batch_data = next(iter(dataloader))
    print("\n批次中各张量的形状 (TF 命名):")
    for key, value in batch_data.items():
        expected_shape = f"(batch_size={batch_size}, seq_len={value.shape[1]})"
        print(f" - '{key}':\t{str(value.shape):<25} (应为 {expected_shape})")
    print("\n--- 验证数据内容 ---")
    # 验证逻辑与之前相同，只是从字典中取值的键名变了
    sample_idx = 0
    input_token_sample = batch_data['Input-Token'][sample_idx]
    token_ids_sample = batch_data['token_ids'][sample_idx] # 这是真实标签

    print("原始标签 ('token_ids'):")
    print(token_ids_sample.tolist()[:30], "...")
    print("\n模型输入 ('Input-Token'):")
    print(input_token_sample.tolist()[:30], "...")
    
    # 检查 Input-Segment 是否全为 0
    input_segment_sample = batch_data['Input-Segment'][sample_idx]
    is_segment_all_zeros = torch.all(input_segment_sample == 0).item()
    print(f"\n'Input-Segment' 是否全为 0: {is_segment_all_zeros}")
