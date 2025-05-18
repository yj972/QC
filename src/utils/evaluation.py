"""
评估工具模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def parse_prediction(prediction: str) -> Tuple[str, float]:
    """
    解析模型预测结果
    
    Args:
        prediction (str): 模型预测结果
        
    Returns:
        Tuple[str, float]: (类别, 置信度)
    """
    try:
        lines = prediction.split('\n')
        category = lines[0].split(': ')[1].strip()
        confidence = float(lines[1].split(': ')[1].strip())
        return category, confidence
    except:
        return None, None

def evaluate_predictions(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    评估模型预测结果
    
    Args:
        results_df (pd.DataFrame): 包含预测结果的数据框
        
    Returns:
        Dict[str, float]: 评估指标
    """
    total = len(results_df)
    correct = 0
    confidence_scores = []
    
    for _, row in results_df.iterrows():
        if row['model_prediction'] == '处理失败':
            continue
            
        pred_category, confidence = parse_prediction(row['model_prediction'])
        if pred_category and confidence:
            confidence_scores.append(confidence)
            # 检查预测是否正确
            if row['true_labels'][pred_category] == 1:
                correct += 1
    
    # 计算评估指标
    accuracy = correct / total if total > 0 else 0
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    return {
        '总样本数': total,
        '正确预测数': correct,
        '准确率': accuracy,
        '平均置信度': avg_confidence
    } 