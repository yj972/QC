import torch
import torch.nn as nn
from data_processor import DataProcessor
from model import QuantumClassifier
from train import train_model, calculate_metrics
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_pipeline():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 读取完整数据集
    print("\n=== 加载数据 ===")
    df = pd.read_csv('data/processed/trainingset_fixed.csv')
    print(f"原始数据量: {len(df)} 条")
    
    # 准备标签
    label_columns = [
        'methods_of_building_qubits',
        'addressing_obstacles',
        'quantum_computing_model',
        'quantum_algorithms',
        'quantum_programming'
    ]
    labels = df[label_columns].values
    
    # 使用分层抽样抽取1000条数据
    print("\n=== 按原始分布抽取1000条数据 ===")
    # 计算每个类别的样本数
    print("\n原始数据集类别分布:")
    for i, col in enumerate(label_columns):
        count = np.sum(labels[:, i])
        print(f"{col}: {count} 样本 ({count/len(df)*100:.2f}%)")
    
    # 使用分层抽样抽取1000条数据
    _, sampled_indices = train_test_split(
        np.arange(len(df)),
        train_size=1000,
        random_state=42,
        stratify=labels
    )
    
    # 获取抽样后的数据
    df_sampled = df.iloc[sampled_indices].reset_index(drop=True)
    print(f"\n抽样后数据量: {len(df_sampled)} 条")
    
    # 打印抽样后的类别分布
    print("\n抽样后类别分布:")
    sampled_labels = df_sampled[label_columns].values
    for i, col in enumerate(label_columns):
        count = np.sum(sampled_labels[:, i])
        print(f"{col}: {count} 样本 ({count/len(df_sampled)*100:.2f}%)")
    
    # 保存临时数据文件
    temp_data_path = 'data/processed/temp_train_fixed.csv'
    df_sampled.to_csv(temp_data_path, index=False)
    
    try:
        # 初始化数据处理器
        processor = DataProcessor(
            data_path=temp_data_path,
            max_len=256,
            batch_size=32
        )
        
        # 准备数据
        print("\n=== 开始数据处理 ===")
        train_loader, val_loader, test_loader, vocab = processor.prepare_data()
        
        # 初始化模型
        print("\n=== 初始化模型 ===")
        model = QuantumClassifier(
            vocab_size=len(vocab),
            embedding_dim=100,  # 与模型结构一致
            hidden_dim=128,    # 与模型结构一致
            num_labels=5,
            dropout=0.5
        ).to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(  # 使用AdamW优化器
            model.parameters(), 
            lr=0.005,
            weight_decay=0.01  # 增加权重衰减
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.005,
            epochs=50,
            steps_per_epoch=len(train_loader)
        )
        
        # 训练模型
        print("\n=== 开始训练 ===")
        train_losses, val_losses, best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=50
        )
        
        # 在测试集上评估
        print("\n=== 在测试集上评估 ===")
        model.eval()
        test_preds = []
        test_probs = []
        test_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_preds.extend(output.cpu().numpy())
                test_probs.extend(output.cpu().numpy())
                test_labels.extend(target.cpu().numpy())
        
        # 计算测试集指标
        test_metrics = calculate_metrics(
            np.array(test_labels),
            np.array(test_preds),
            np.array(test_probs)
        )
        
        # 获取预测的类别
        y_pred = np.argmax(np.array(test_preds), axis=1)
        y_true = np.argmax(np.array(test_labels), axis=1)
        
        # 类别标签
        labels = [
            'methods_of_building_qubits',
            'addressing_obstacles',
            'quantum_computing_model',
            'quantum_algorithms',
            'quantum_programming'
        ]
        
        # 打印详细的分类报告
        print("\n=== 详细的分类报告 ===")
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y_true, y_pred, labels, 'results/confusion_matrix.png')
        
        # 打印最佳指标
        print("\n=== 最佳模型指标（验证集）===")
        print("\n宏平均指标:")
        print(f"F1: {best_metrics['macro_avg']['f1']:.4f}")
        print(f"Precision: {best_metrics['macro_avg']['precision']:.4f}")
        print(f"Recall: {best_metrics['macro_avg']['recall']:.4f}")
        print(f"Accuracy: {best_metrics['macro_avg']['accuracy']:.4f}")
        
        print("\n各类别详细指标:")
        for label in labels:
            print(f"\n{label}:")
            print(f"Precision: {best_metrics[label]['precision']:.4f}")
            print(f"Recall: {best_metrics[label]['recall']:.4f}")
            print(f"F1: {best_metrics[label]['f1']:.4f}")
            print(f"Accuracy: {best_metrics[label]['accuracy']:.4f}")
            
        # 打印测试集指标
        print("\n=== 测试集指标 ===")
        print("\n宏平均指标:")
        print(f"F1: {test_metrics['macro_avg']['f1']:.4f}")
        print(f"Precision: {test_metrics['macro_avg']['precision']:.4f}")
        print(f"Recall: {test_metrics['macro_avg']['recall']:.4f}")
        print(f"Accuracy: {test_metrics['macro_avg']['accuracy']:.4f}")
        
        print("\n各类别详细指标:")
        for label in labels:
            print(f"\n{label}:")
            print(f"Precision: {test_metrics[label]['precision']:.4f}")
            print(f"Recall: {test_metrics[label]['recall']:.4f}")
            print(f"F1: {test_metrics[label]['f1']:.4f}")
            print(f"Accuracy: {test_metrics[label]['accuracy']:.4f}")
            
    finally:
        # 清理临时文件
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)

if __name__ == "__main__":
    test_pipeline() 