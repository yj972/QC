import torch
from src.neural_network.model import QuantumClassifier

# 假设词表大小和模型参数与当前训练一致
VOCAB_SIZE = 8415  # 请根据实际词表大小调整
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
NUM_LABELS = 5
DROPOUT = 0.5

model = QuantumClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_labels=NUM_LABELS,
    dropout=DROPOUT
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}") 