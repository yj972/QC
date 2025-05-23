#!/bin/bash

# 一键运行脚本：数据处理、embedding提取、MLP训练和评估

echo "开始运行量子文本分类pipeline..."

# 1. 数据处理（假设数据处理脚本为process_data.py）
echo "1. 数据处理..."
python3 process_data.py

# 2. 提取embedding
echo "2. 提取embedding..."
python3 extract_embedding.py

# 3. 训练MLP模型
echo "3. 训练MLP模型..."
python3 mlp_classifier.py

# 4. 评估MLP模型
echo "4. 评估MLP模型..."
python3 evaluate_mlp.py

echo "pipeline运行完成！结果保存在 results 2/ 目录下。" 