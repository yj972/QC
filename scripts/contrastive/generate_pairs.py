"""
构造BGE微调所需的正对/反对pair数据。
基于标签模式生成正负样本对：
- 正样本对：相同标签模式的样本之间
- 负样本对：不同标签模式的样本之间
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PairDataGenerator:
    def __init__(self, data_dir: str, output_dir: str):
        """
        初始化数据生成器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义标签列
        self.label_columns = [
            'methods_of_building_qubits',
            'addressing_obstacles',
            'quantum_computing_model',
            'quantum_algorithms',
            'quantum_programming'
        ]
    
    def load_data(self, test_mode: bool = False) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            test_mode: 是否使用测试模式（只加载少量数据）
        """
        logger.info("加载数据...")
        data_file = os.path.join(self.data_dir, 'trainingset_confidence_filtered.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"找不到数据文件: {data_file}")
        
        df = pd.read_csv(data_file)
        if test_mode:
            # 每个标签模式只取前2个样本
            label_patterns = {}
            for _, row in df.iterrows():
                pattern = self.get_label_pattern(row[self.label_columns].values)
                if pattern not in label_patterns:
                    label_patterns[pattern] = []
                if len(label_patterns[pattern]) < 2:  # 改为2个样本
                    label_patterns[pattern].append(row)
            
            # 合并所有样本
            df = pd.concat([pd.DataFrame(samples) for samples in label_patterns.values()], ignore_index=True)
            logger.info(f"测试模式：使用 {len(df)} 个样本")
        
        # 合并标题和摘要
        df['text'] = df['title'] + ': ' + df['abstract']
        logger.info(f"数据加载完成，总样本数: {len(df)}")
        
        return df
    
    def get_label_pattern(self, label: np.ndarray) -> str:
        """获取标签模式"""
        return '_'.join(map(str, label))
    
    def generate_pairs(self, df: pd.DataFrame, 
                      positive_ratio: float = 0.5,
                      max_pairs_per_sample: int = 3) -> List[Dict]:
        """
        生成训练数据对
        
        Args:
            df: 数据DataFrame
            positive_ratio: 正样本对的比例
            max_pairs_per_sample: 每个样本最多生成的对数
            
        Returns:
            训练数据对列表
        """
        logger.info("生成训练数据对...")
        
        # 获取所有标签模式
        label_patterns = {}
        for _, row in df.iterrows():
            pattern = self.get_label_pattern(row[self.label_columns].values)
            if pattern not in label_patterns:
                label_patterns[pattern] = []
            label_patterns[pattern].append(row)
        
        logger.info(f"共有 {len(label_patterns)} 种不同的标签模式")
        
        # 生成数据对
        pairs = []
        for pattern, samples in tqdm(label_patterns.items(), desc="生成数据对"):
            # 生成正样本对（相同标签）
            for i in range(len(samples)):
                for j in range(i + 1, min(i + max_pairs_per_sample + 1, len(samples))):
                    pairs.append({
                        'text1': samples[i]['text'],
                        'text2': samples[j]['text'],
                        'label': 1  # 正样本对
                    })
            
            # 生成负样本对（不同标签）
            other_patterns = [p for p in label_patterns.keys() if p != pattern]
            if other_patterns:
                for sample in samples:
                    # 为每个样本生成负样本对
                    n_negative = int(max_pairs_per_sample * (1 - positive_ratio))
                    for _ in range(n_negative):
                        other_pattern = random.choice(other_patterns)
                        other_sample = random.choice(label_patterns[other_pattern])
                        pairs.append({
                            'text1': sample['text'],
                            'text2': other_sample['text'],
                            'label': 0  # 负样本对
                        })
        
        # 打乱数据对顺序
        random.shuffle(pairs)
        logger.info(f"生成数据对完成，共 {len(pairs)} 对")
        return pairs
    
    def save_pairs(self, pairs: List[Dict]):
        """
        保存所有数据对作为训练集
        
        Args:
            pairs: 数据对列表
        """
        logger.info("保存数据对...")
        
        # 保存所有数据对
        train_file = os.path.join(self.output_dir, 'train_pairs.json')
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练集大小: {len(pairs)}")
        logger.info(f"数据对已保存至: {self.output_dir}")

def main():
    # 设置路径
    data_dir = "data/processed"
    output_dir = "output/bge_finetune/pairs"
    
    # 创建数据生成器
    generator = PairDataGenerator(data_dir, output_dir)
    
    # 加载完整数据
    df = generator.load_data(test_mode=False)
    
    # 生成数据对
    pairs = generator.generate_pairs(df)
    
    # 保存数据对
    generator.save_pairs(pairs)

if __name__ == "__main__":
    main() 