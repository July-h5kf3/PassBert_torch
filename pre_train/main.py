import os
import sys
import logging
from functools import partial
from tqdm import tqdm

import torch
import torch.nn.functional as  F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from passbert.model import build_transformer_model
from passbert.PasswordTokenizer import PasswordTokenizer
from pre_train.dataset import TrainingDataset,mlm_collate_fn

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

model_saved_path = "D:/Competition/Password/passbert_pytorch/Model_ckpt"
corpus_path = os.path.join(os.path.dirname(__file__), 'train_data.jsonl')
config_path = None
checkpoint_path = None

sequence_length = 32
batch_size = 512
learning_rate = 0.00176
weight_decay_rate = 0.01
num_warmup_steps = 31250
num_train_steps = 125000
steps_per_epoch = 10000
grad_accum_steps = 1
epochs = num_train_steps * grad_accum_steps // steps_per_epoch
exclude_from_weight_decay = ['Norm','bias']

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    创建具有warmup的线性学习率调度器
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def main():
    if not torch.cuda.is_available():
        logging.error("CUDA is not available.")
        sys.exit(1)
    device = torch.device("cuda")
    logging.info(f"使用设备:{device}")
    logging.info(f"Batch Size: {batch_size}")

    dataset = TrainingDataset(file_path=corpus_path)
    tokenizer = PasswordTokenizer()
    pad_token_id = tokenizer._token_pad_id

    collate_fn = partial(mlm_collate_fn,pad_token_id = pad_token_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 设置基本的模型参数
    model_kwargs = {
        "hidden_size": 256,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
        "vocab_size": 99,
        "hidden_dropout_prob": 0.1,
        "num_attention_heads": 2,
        "type_vocab_size": 2,
        "max_position_embeddings": sequence_length,
        "num_hidden_layers": 4,
        "intermediate_size": 512,
        "attention_probs_dropout_prob": 0.1,
        "with_mlm": True
    }

    model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        **model_kwargs
    )
    model.to(device)

    logging.info(f"自定义模型加载成功，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    param_optimizer = list(model.named_parameters())
    no_decay = exclude_from_weight_decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    global_step = 0
    model.zero_grad()

    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", total=min(steps_per_epoch, len(dataloader)))

        for step,batch in enumerate(progress_bar):
            if step >= steps_per_epoch:break

            input_tokens = batch['Input-Token'].to(device)
            input_segments = batch['Input-Segment'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_ids_labels = batch['token_ids'].to(device)
            is_masked = batch['is_masked'].to(device)

            output = model(
                token_ids = input_tokens,
                segment_ids = input_segments,
                attention_mask = attention_mask
            )

            # 如果模型返回元组，取第一个元素作为logits（MLM scores）
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            loss = F.cross_entropy(logits.view(-1,model.vocab_size),token_ids_labels.view(-1),reduction='none')
            masked_loss = loss * is_masked.view(-1)
            final_loss = masked_loss.sum() / (is_masked.sum() + 1e-8)

            final_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.set_postfix({'loss': f"{final_loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.6f}"})

        model_path = os.path.join(model_saved_path, f"passbert_epoch_{epoch + 1}.pt")
        logging.info(f"Epoch {epoch + 1} 结束, 保存模型到 {model_path}")
        os.makedirs(model_saved_path, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    logging.info("训练完成！")
if __name__ == "__main__":
    main()