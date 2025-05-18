import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_labels=5, dropout=0.5, max_len=256):
        super(QuantumClassifier, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,  # 只用1层
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # CNN层
        self.conv1 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # 池化层
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 自动推断展平后维度
        with torch.no_grad():
            dummy = torch.zeros(1, max_len, embedding_dim)
            lstm_out, _ = self.lstm(dummy)
            x = lstm_out.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.maxpool(x)
            x = self.relu(self.conv2(x))
            x = self.maxpool(x)
            x = x.view(x.size(0), -1)
            flatten_dim = x.size(1)
        self.fc = nn.Linear(flatten_dim + hidden_dim * 2, 128)
        self.out = nn.Linear(128, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.max_len = max_len
        
    def attention_net(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim*2]
        attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # [batch_size, seq_len, 1]
        context = torch.bmm(lstm_output.transpose(1, 2), attention_weights)  # [batch_size, hidden_dim*2, 1]
        return context.squeeze(2)  # [batch_size, hidden_dim*2]
        
    def forward(self, x):
        # 词嵌入
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM层
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # 注意力层
        attn_output = self.attention_net(lstm_out)  # [batch_size, hidden_dim*2]
        
        # 转换维度用于CNN
        x = lstm_out.transpose(1, 2)  # [batch_size, hidden_dim*2, seq_len]
        
        # CNN层
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 合并CNN输出和注意力输出
        x = torch.cat([x, attn_output], dim=1)
        
        # 全连接层
        x = self.dropout(self.relu(self.fc(x)))
        x = self.out(x)
        
        # 输出层 - 使用softmax进行多标签分类
        x = self.softmax(x)
        
        return x 