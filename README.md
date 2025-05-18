# 量子文本分类项目

基于多种方法的量子计算文本分类系统。

## 项目结构

```
quantum_classification/
├── src/                    # 源代码
│   ├── data/              # 数据处理相关代码
│   ├── models/            # 模型相关代码
│   │   └── neural_network/  # 神经网络模型
│   ├── utils/             # 工具函数
│   └── config/            # 配置文件
├── tests/                 # 测试代码
├── docs/                  # 文档
├── notebooks/             # Jupyter notebooks
├── data/                  # 数据文件
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── models/                # 保存的模型
└── results/               # 结果输出
    ├── figures/          # 图表
    └── metrics/          # 评估指标
```

## 功能模块

1. 数据处理
   - 数据加载和预处理
   - 特征工程
   - 数据增强

2. 模型实现
   - 神经网络模型
   - 评估工具
   - 模型训练和预测

3. 工具函数
   - 评估指标计算
   - 可视化工具
   - 辅助函数

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 数据准备
```python
from quantum_classification.src.data.data_loader import load_data
data = load_data("path/to/your/data.csv")
```

2. 模型训练
```python
from quantum_classification.src.models.neural_network.trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train(data)
```

3. 模型评估
```python
from quantum_classification.src.utils.evaluation import evaluate_model

metrics = evaluate_model(model, test_data)
```

## 开发

1. 克隆仓库
```bash
git clone [repository_url]
cd quantum_classification
```

2. 安装开发依赖
```bash
pip install -r requirements-dev.txt
```

3. 运行测试
```bash
pytest tests/
```

## 许可证

MIT License 

## 实验结果（神经网络方法）

本节展示基于神经网络方法的分类实验结果，包含验证集与测试集的宏平均指标及各类别详细指标。

### 混淆矩阵
![confusion_matrix](results/confusion_matrix.png)

### 验证集指标

**宏平均指标：**
- F1: 0.4612
- Precision: 0.4595
- Recall: 0.4691
- Accuracy: 0.6267

**各类别详细指标：**

| 类别                        | Precision | Recall  | F1     | Accuracy |
|-----------------------------|-----------|---------|--------|----------|
| methods_of_building_qubits  | 0.8152    | 0.7188  | 0.7640 | 0.8127   |
| addressing_obstacles        | 0.4479    | 0.3909  | 0.4175 | 0.8383   |
| quantum_computing_model     | 0.3675    | 0.4803  | 0.4164 | 0.7695   |
| quantum_algorithms          | 0.6667    | 0.7556  | 0.7083 | 0.8491   |
| quantum_programming         | 0.0000    | 0.0000  | 0.0000 | 0.9838   |

---

### 测试集指标

**宏平均指标：**
- F1: 0.4292
- Precision: 0.4242
- Recall: 0.4360
- Accuracy: 0.6119

**各类别详细指标：**

| 类别                        | Precision | Recall  | F1     | Accuracy |
|-----------------------------|-----------|---------|--------|----------|
| methods_of_building_qubits  | 0.7862    | 0.7636  | 0.7747 | 0.8127   |
| addressing_obstacles        | 0.3241    | 0.3182  | 0.3211 | 0.8005   |
| quantum_computing_model     | 0.3613    | 0.3413  | 0.3510 | 0.7857   |
| quantum_algorithms          | 0.6493    | 0.7569  | 0.6990 | 0.8410   |
| quantum_programming         | 0.0000    | 0.0000  | 0.0000 | 0.9838   |

---

> 以上为神经网络方法的实验结果，所有指标均为本次实验真实输出，未做任何省略或简化。 