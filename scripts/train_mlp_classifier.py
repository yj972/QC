import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score

# 参数
embedding_file = "embeddings.npy"
label_file = "labels.npy"
output_model = "mlp_best.pth"
num_epochs = 20
batch_size = 64
lr = 1e-3

# 加载数据
X = np.load(embedding_file)
y = np.load(label_file)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 划分训练/验证集
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# MLP模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )
    def forward(self, x):
        return self.fc(x)

model = MLPClassifier(X.shape[1], y.shape[1]).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# 训练
best_f1 = 0
for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.cuda(), yb.cuda()
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 验证
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.cuda()
            logits = model(xb)
            preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
            all_preds.append(preds)
            all_labels.append(yb.numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels, all_preds, average='micro')
    print(f"Epoch {epoch+1}: val micro-F1={f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), output_model)
        print("保存最佳模型")

print("训练完成，最佳micro-F1:", best_f1) 