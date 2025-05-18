"""
数据加载模块
"""

import pandas as pd
import numpy as np
from ..config.config import DATA_CONFIG

def load_data(file_path=None):
    """
    加载数据
    
    Args:
        file_path (str, optional): 数据文件路径. 默认使用配置文件中的路径
        
    Returns:
        pd.DataFrame: 加载的数据
    """
    if file_path is None:
        file_path = DATA_CONFIG["train_data_path"]
    
    df = pd.read_csv(file_path)
    
    # 数据预处理
    if 'quantum_algorithms' in df.columns and 'quantum_programming' in df.columns:
        df['quantum_programming'] = np.where(
            (df['quantum_algorithms'] == 1) & (df['quantum_programming'] == 1),
            0, 1
        )
    
    return df

def filter_by_doc_type(df, doc_type='patent'):
    """
    按文档类型筛选数据
    
    Args:
        df (pd.DataFrame): 输入数据
        doc_type (str): 文档类型
        
    Returns:
        pd.DataFrame: 筛选后的数据
    """
    return df[df['doc_type'] == doc_type] 