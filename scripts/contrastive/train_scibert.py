"""
使用对比学习方式微调SciBERT模型。
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import numpy as np
from typing import List, Dict, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContrastiveDataset(Dataset):
    def __init__(self, pairs_file: str, tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            pairs_file: 样本对文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(pairs_file, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)
        
        logger.info(f"加载了 {len(self.pairs)} 对样本")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 对两个文本进行编码
        encoding1 = self.tokenizer(
            pair['text1'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            pair['text2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移除batch维度
        encoding1 = {k: v.squeeze(0) for k, v in encoding1.items()}
        encoding2 = {k: v.squeeze(0) for k, v in encoding2.items()}
        
        return {
            'input_ids1': encoding1['input_ids'],
            'attention_mask1': encoding1['attention_mask'],
            'input_ids2': encoding2['input_ids'],
            'attention_mask2': encoding2['attention_mask'],
            'label': torch.tensor(pair['label'], dtype=torch.float)
        }

class ContrastiveModel(nn.Module):
    def __init__(self, model_name: str):
        """
        初始化对比学习模型
        
        Args:
            model_name: 预训练模型名称
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # 获取两个文本的表示
        outputs1 = self.encoder(input_ids1, attention_mask1)
        outputs2 = self.encoder(input_ids2, attention_mask2)
        
        # 使用[CLS]标记的输出作为文本表示
        embeddings1 = outputs1.last_hidden_state[:, 0, :]
        embeddings2 = outputs2.last_hidden_state[:, 0, :]
        
        # 计算余弦相似度
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        similarity = torch.sum(embeddings1 * embeddings2, dim=1)
        
        return similarity

def train(model, train_loader, optimizer, scheduler, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in progress_bar:
        # 将数据移到设备上
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        similarity = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        
        # 计算对比损失
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

def main():
    # 设置参数
    model_name = "allenai/scibert_scivocab_uncased"
    pairs_file = "output/bge_finetune/pairs/train_pairs.json"
    output_dir = "output/bge_finetune/scibert"
    max_length = 512
    batch_size = 4  # 减小batch_size
    gradient_accumulation_steps = 4  # 添加梯度累积
    num_epochs = 3
    learning_rate = 2e-5
    warmup_steps = 100
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ContrastiveModel(model_name).to(device)
    
    # 创建数据集和数据加载器
    dataset = ContrastiveDataset(pairs_file, tokenizer, max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练模型
    logger.info("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # 重置梯度
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for step, batch in enumerate(progress_bar):
            # 将数据移到设备上
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            similarity = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            
            # 计算对比损失
            loss = F.binary_cross_entropy_with_logits(similarity, labels)
            loss = loss / gradient_accumulation_steps  # 缩放损失
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # 保存模型
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"模型已保存至: {output_dir}")

if __name__ == "__main__":
    main() 