import pandas as pd
import numpy as np
from tqdm import tqdm

def process_labels(df):
    """
    处理标签，当quantum_algorithms和quantum_programming同时为1时，
    将quantum_programming设置为0
    """
    # 创建标签的副本
    labels = df.copy()
    
    # 找出同时为1的情况
    mask = (labels['quantum_algorithms'] == 1) & (labels['quantum_programming'] == 1)
    
    # 将quantum_programming设置为0
    labels.loc[mask, 'quantum_programming'] = 0
    
    return labels

def main():
    # 读取原始数据
    print("读取原始数据...")
    df = pd.read_csv('../trainingset.csv')
    
    # 处理标签
    print("处理标签...")
    processed_labels = process_labels(df)
    
    # 保存处理后的数据
    print("保存处理后的数据...")
    processed_labels.to_csv('processed_trainingset.csv', index=False)
    
    # 打印一些统计信息
    print("\n标签统计信息：")
    print("原始数据中quantum_algorithms和quantum_programming同时为1的数量：",
          ((df['quantum_algorithms'] == 1) & (df['quantum_programming'] == 1)).sum())
    print("处理后quantum_programming为1的数量：",
          (processed_labels['quantum_programming'] == 1).sum())
    print("处理后quantum_algorithms为1的数量：",
          (processed_labels['quantum_algorithms'] == 1).sum())

if __name__ == "__main__":
    main() 