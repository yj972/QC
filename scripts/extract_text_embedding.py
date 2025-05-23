import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 路径参数
model_dir = "/root/autodl-tmp/scibert_offline"
data_file = "data/processed/trainingset_confidence_filtered.csv"
output_emb = "embeddings.npy"
output_label = "labels.npy"

# 标签列
label_columns = [
    'methods_of_building_qubits',
    'addressing_obstacles',
    'quantum_computing_model',
    'quantum_algorithms',
    'quantum_programming'
]

# 加载模型和分词器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir).to(device)
model.eval()

# 加载数据
df = pd.read_csv(data_file)
texts = (df['title'] + ': ' + df['abstract']).tolist()
labels = df[label_columns].values

# 批量提取嵌入
all_embeddings = []
batch_size = 32
with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        # 取[CLS]向量
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
        print(f"Processed {i+len(batch_texts)}/{len(texts)}")

all_embeddings = np.concatenate(all_embeddings, axis=0)
np.save(output_emb, all_embeddings)
np.save(output_label, labels)
print(f"保存嵌入到 {output_emb}，标签到 {output_label}") 