import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
embeddings = np.load("results_final/text_embeddings.npy")
labels = np.load("results_final/multilabel_targets.npy")

# 定义MLP结构（和训练时保持一致）
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_labels),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

input_dim = embeddings.shape[1]
hidden_dim = 256  # 如有不同请修改
num_labels = labels.shape[1]

model = MLP(input_dim, hidden_dim, num_labels)
model.load_state_dict(torch.load("results_final/mlp_classifier_best.pth", map_location="cpu"))
model.eval()

# 推理
with torch.no_grad():
    outputs = model(torch.FloatTensor(embeddings))
    preds = (outputs > 0.5).float().numpy()

# 评估
micro_acc = float(np.mean(preds == labels))
strict_acc = float(np.mean(np.all(preds == labels, axis=1)))
label_names = [
    'methods_of_building_qubits',
    'addressing_obstacles',
    'quantum_computing_model',
    'quantum_algorithms',
    'quantum_programming'
]

print("整体准确率：", micro_acc)
print("严格准确率：", strict_acc)

results = {
    "micro_accuracy": micro_acc,
    "strict_accuracy": strict_acc,
    "per_label_report": {}
}

# 创建一个大的图形，包含所有混淆矩阵
plt.figure(figsize=(20, 15))
plt.suptitle('Confusion Matrices for All Categories', fontsize=16, y=0.95)

# 为每个类别生成混淆矩阵
for i, name in enumerate(label_names):
    # 生成分类报告
    report = classification_report(labels[:, i], preds[:, i], digits=4, output_dict=True)
    results["per_label_report"][name] = report
    print(f"\n{name}")
    print(classification_report(labels[:, i], preds[:, i], digits=4))
    
    # 生成混淆矩阵
    cm = confusion_matrix(labels[:, i], preds[:, i])
    
    # 在子图中绘制混淆矩阵
    plt.subplot(2, 3, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# 调整子图之间的间距
plt.tight_layout()
# 保存整个图形
plt.savefig('results_final/confusion_matrices_all.png', bbox_inches='tight', dpi=300)
plt.close()

# 保存为json
with open("results_final/mlp_evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("评估结果已保存为 results_final/mlp_evaluation_report.json") 