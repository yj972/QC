from data_processor import QuantumTextDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss, roc_auc_score
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_processor import DataProcessor
from model import QuantumClassifier

def calculate_metrics(y_true, y_pred, y_prob):
    """计算各种评估指标"""
    metrics = {}
    
    # 获取预测的类别
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # 计算每个类别的指标
    for i, label in enumerate([
        'methods_of_building_qubits',
        'addressing_obstacles',
        'quantum_computing_model',
        'quantum_algorithms',
        'quantum_programming'
    ]):
        try:
            metrics[label] = {
                'precision': precision_score(y_true_classes == i, y_pred_classes == i, zero_division=0),
                'recall': recall_score(y_true_classes == i, y_pred_classes == i, zero_division=0),
                'f1': f1_score(y_true_classes == i, y_pred_classes == i, zero_division=0),
                'accuracy': accuracy_score(y_true_classes == i, y_pred_classes == i)
            }
        except ValueError as e:
            print(f"警告: 类别 {label} 的指标计算失败: {str(e)}")
            metrics[label] = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'accuracy': 0
            }
    
    # 计算宏平均指标
    metrics['macro_avg'] = {
        'precision': precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0),
        'recall': recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0),
        'f1': f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0),
        'accuracy': accuracy_score(y_true_classes, y_pred_classes)
    }
    
    # 计算微平均指标
    metrics['micro_avg'] = {
        'precision': precision_score(y_true_classes, y_pred_classes, average='micro', zero_division=0),
        'recall': recall_score(y_true_classes, y_pred_classes, average='micro', zero_division=0),
        'f1': f1_score(y_true_classes, y_pred_classes, average='micro', zero_division=0),
        'accuracy': accuracy_score(y_true_classes, y_pred_classes)
    }
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    """训练模型"""
    train_losses = []
    val_losses = []
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    best_metrics = None
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_probs = []
        train_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(output.cpu().detach().numpy())
            train_probs.extend(output.cpu().detach().numpy())
            train_labels.extend(target.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_preds.extend(output.cpu().numpy())
                val_probs.extend(output.cpu().numpy())
                val_labels.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 计算评估指标
        train_metrics = calculate_metrics(
            np.array(train_labels), 
            np.array(train_preds),
            np.array(train_probs)
        )
        val_metrics = calculate_metrics(
            np.array(val_labels), 
            np.array(val_preds),
            np.array(val_probs)
        )
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        print('\n训练集指标:')
        print(f"Macro F1: {train_metrics['macro_avg']['f1']:.4f}")
        print(f"Macro Precision: {train_metrics['macro_avg']['precision']:.4f}")
        print(f"Macro Recall: {train_metrics['macro_avg']['recall']:.4f}")
        
        print('\n验证集指标:')
        print(f"Macro F1: {val_metrics['macro_avg']['f1']:.4f}")
        print(f"Macro Precision: {val_metrics['macro_avg']['precision']:.4f}")
        print(f"Macro Recall: {val_metrics['macro_avg']['recall']:.4f}")
        
        print('\n各类别F1分数:')
        for label in [
            'methods_of_building_qubits',
            'addressing_obstacles',
            'quantum_computing_model',
            'quantum_algorithms',
            'quantum_programming'
        ]:
            print(f"{label}: {val_metrics[label]['f1']:.4f}")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_metrics['macro_avg']['f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_avg']['f1']
            best_metrics = val_metrics
            torch.save(model.state_dict(), 'models/best_model.pth')
            patience_counter = 0
            print('\n保存新的最佳模型!')
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses, best_metrics

def plot_metrics(train_losses, val_losses, save_dir):
    """绘制训练指标"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()

def save_metrics(metrics, save_dir):
    """保存评估指标"""
    # 将指标保存为CSV
    metrics_df = pd.DataFrame()
    
    # 添加每个类别的指标
    for label in [
        'methods_of_building_qubits',
        'addressing_obstacles',
        'quantum_computing_model',
        'quantum_algorithms',
        'quantum_programming'
    ]:
        metrics_df = pd.concat([
            metrics_df,
            pd.DataFrame({
                'Category': [label],
                'Precision': [metrics[label]['precision']],
                'Recall': [metrics[label]['recall']],
                'F1': [metrics[label]['f1']],
                'Accuracy': [metrics[label]['accuracy']]
            })
        ])
    
    # 添加宏平均和微平均指标
    for avg_type in ['macro_avg', 'micro_avg']:
        metrics_df = pd.concat([
            metrics_df,
            pd.DataFrame({
                'Category': [avg_type],
                'Precision': [metrics[avg_type]['precision']],
                'Recall': [metrics[avg_type]['recall']],
                'F1': [metrics[avg_type]['f1']],
                'Accuracy': [metrics[avg_type]['accuracy']]
            })
        ])
    
    # 保存为CSV
    metrics_df.to_csv(os.path.join(save_dir, 'evaluation_metrics.csv'), index=False)
    
    # 保存为JSON
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def prepare_data(data_path, batch_size=32, max_samples=None):
    """准备数据加载器"""
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 只选择patent数据
    df = df[df['type'] == 'patent']
    
    # 如果指定了最大样本数，则限制数据量
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    # 合并标题和摘要
    df['text'] = df['title'] + ": " + df['abstract']
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 文本预处理
    texts = []
    for text in tqdm(df['text'], desc='处理文本'):
        text = processor.clean_text(text)
        text = processor.process_quantum_terms(text)
        tokens = processor.tokenize(text)
        texts.append(tokens)
    
    # 构建词汇表
    vocab = processor.build_vocab(texts)
    
    # 准备标签
    label_columns = [
        'methods_of_building_qubits',
        'addressing_obstacles',
        'quantum_computing_model',
        'quantum_algorithms',
        'quantum_programming'
    ]
    labels = df[label_columns].values
    
    # 划分训练集、验证集和测试集 (6:2:2)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.4, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    # 创建数据集
    train_dataset = QuantumTextDataset(train_texts, train_labels, vocab)
    val_dataset = QuantumTextDataset(val_texts, val_labels, vocab)
    test_dataset = QuantumTextDataset(test_texts, test_labels, vocab)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, len(vocab), label_columns

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 准备数据
    data_path = '../../processed_trainingset.csv'
    train_loader, val_loader, test_loader, vocab_size, label_columns = prepare_data(
        data_path, 
        batch_size=32, 
        max_samples=2000
    )
    
    # 创建模型
    print("创建模型...")
    model = QuantumClassifier(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_labels=5,
        dropout=0.6
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.005,  # 增大学习率
        weight_decay=1e-4
    )
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses, best_metrics = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device,
        num_epochs=50
    )
    
    # 在测试集上评估
    print("\n在测试集上评估模型...")
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
    
    # 保存所有指标
    save_metrics(best_metrics, 'results/validation_metrics')
    save_metrics(test_metrics, 'results/test_metrics')
    
    # 绘制训练损失
    plot_metrics(train_losses, val_losses, 'results')
    
    # 保存训练配置
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': 300,
        'hidden_dim': 256,
        'num_labels': 5,
        'batch_size': 32,
        'max_len': 256,
        'dropout': 0.6,
        'learning_rate': 0.005,  # 更新配置中的学习率
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sample_size': 2000,
        'data_type': 'patent_only'
    }
    
    with open('results/training_config.json', 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main() 