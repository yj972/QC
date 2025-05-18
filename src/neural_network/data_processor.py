import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from tqdm import tqdm
import os

class QuantumTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本转换为序列
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in text]
        
        # 截断或填充序列
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            sequence = sequence + [self.vocab['<PAD>']] * (self.max_len - len(sequence))
            
        return torch.tensor(sequence), torch.tensor(label, dtype=torch.float)

class DataProcessor:
    def __init__(self, data_path='data/raw/train_fixed.csv', max_len=256, batch_size=32):
        self.data_path = data_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.last_stopword = set([
            "present_invention", "invention", "claim_wherein", "wherein", "least_one", 
            "et_al", "phys_rev", "allows_us", "also_discussed", "enables_us", "first", 
            "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", 
            "tenth", "zero", "one", "two", "three", "four", "five", "six", "seven", 
            "eight", "nine", "ten", "use", "method", "base", "show", "propose", 
            "comprise", "provide", "study", "via", "also", "however", "say", "may", 
            "allow", "introduce", "include", "phy_rev_lett", "phy_rev", "state", 
            "computer", "computing", "bit", "datum", "quantum", "non", "investigate", 
            "result", "present", "give", "find", "paper", "new", "apply", "input", 
            "mode", "property", "form", "implementation", "make", "achieve", "general", 
            "low", "determine", "type", "correspond", "term", "realize", "compare", 
            "configure", "many", "walk", "particular", "define", "simple", "increase", 
            "decrease", "prove", "example", "analyze", "due", "initial", "within", 
            "develop", "without", "exist", "derive", "system"
        ])
        
        # 量子术语映射
        self.quantum_terms = {
            # 基本术语
            r"quamtum|quan_tum": "quantum",
            r"computng": "computing",
            r"qudit|qu_bit": "qubit",
            r"computers": "computer",
            r"computations|com_putation": "computation",
            r"computings": "computing",
            r"computation|computer": "computing",
            
            # 量子计算模型
            r"qed|QED": "quantum_electrodynamics",
            r"nisq": "noisy_intermediate_scale_quantum",
            r"noisy intermediate scalable quantum|noisy_intermediate_scalable_quantum": "noisy_intermediate_scale_quantum",
            r"1WQC": "one_way_quantum_computing",
            r"one way quantum computing": "one_way_quantum_computing",
            
            # 量子器件
            r"squid": "superconducting_quantum_interference_device",
            r"quantum dot|qdot": "quantum_dot",
            r"quantum circuit": "quantum_circuit",
            r"quantum gate": "quantum_gate",
            r"quantum register": "quantum_register",
            
            # 量子算法和应用
            r"\bsat\b|\bsat_|_sat\b|\bSAT\b|\bSAT_|_SAT\b": "satisfiability_problem",
            r"satisfiability problem": "satisfiability_problem",
            r"quantum algorithm": "quantum_algorithm",
            r"quantum error correcting": "quantum_error_correction",
            r"quantum_error_correcting": "quantum_error_correction",
            
            # 量子信息
            r"quantum information": "quantum_information",
            r"quantum entanglement": "quantum_entanglement",
            r"quantum state": "quantum_state",
            r"quantum measurement": "quantum_measurement",
            
            # 量子应用
            r"quantum cryptography": "quantum_cryptography",
            r"quantum communication": "quantum_communication",
            r"quantum simulation": "quantum_simulation",
            r"quantum network": "quantum_network",
            r"quantum sensor": "quantum_sensor",
            r"quantum memory": "quantum_memory",
            r"quantum processor": "quantum_processor",
            
            # 常见错误修正
            r"computingally|computingal|computingized|quantum_computingl": "computing",
            r"computing[a-z]": "computing ",
            r"quan tum": "quantum",
            r"qu bit": "qubit"
        }

    def clean_text(self, text):
        """文本清洗和标准化"""
        # 基本清理
        text = re.sub(r"_", "@", text)
        text = re.sub(r"−|-", "_", text)
        text = re.sub(r"\\r\\n|\\n", " ", text)
        text = re.sub(r"\\'|\\~|\\\"", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"_+", "_", text)
        text = re.sub(r" _ | _|_ ", "_", text)
        
        # 量子术语标准化
        for pattern, replacement in self.quantum_terms.items():
            text = re.sub(pattern, replacement, text)
            
        # 处理数字和特殊字符
        text = re.sub(r'\d+', ' ', text)  # 移除数字
        text = re.sub(r'[^\w\s]', ' ', text)  # 移除特殊字符
        text = re.sub(r'\s+', ' ', text)  # 规范化空白字符
        
        return text.lower().strip()

    def tokenize(self, text):
        """分词"""
        return [token.text for token in self.nlp(text)]

    def lemmatize(self, tokens):
        """词形还原"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_stopwords(self, tokens):
        """移除停用词"""
        return [token for token in tokens if token not in self.stopwords and token not in self.last_stopword]

    def build_vocab(self, texts, min_freq=2):
        """构建词汇表"""
        word_counts = Counter()
        for text in texts:
            word_counts.update(text)
            
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.items():
            if count >= min_freq:
                vocab[word] = len(vocab)
                
        return vocab

    def prepare_data(self):
        """准备数据"""
        print("正在加载数据...")
        # 读取数据
        df = pd.read_csv(self.data_path)
        print(f"总数据量: {len(df)} 条")
        
        # 合并标题和摘要
        df['text'] = df['title'] + ": " + df['abstract']
        
        # 文本预处理
        print("正在预处理文本...")
        texts = []
        for text in tqdm(df['text'], desc="文本处理"):
            text = self.clean_text(text)
            tokens = self.tokenize(text)
            tokens = self.lemmatize(tokens)
            tokens = self.remove_stopwords(tokens)
            texts.append(tokens)
            
        # 构建词汇表
        print("正在构建词汇表...")
        vocab = self.build_vocab(texts)
        print(f"词汇表大小: {len(vocab)}")
        
        # 准备标签
        label_columns = [
            'methods_of_building_qubits',
            'addressing_obstacles',
            'quantum_computing_model',
            'quantum_algorithms',
            'quantum_programming'
        ]
        labels = df[label_columns].values
        
        # 计算每个类别的样本数
        print("\n类别分布:")
        for i, col in enumerate(label_columns):
            count = np.sum(labels[:, i])
            print(f"{col}: {count} 样本")
        
        # 使用分层抽样划分数据集
        print("\n正在划分数据集...")
        # 首先划分出测试集
        train_val_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        
        # 从剩余数据中划分验证集
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.25,  # 0.25 * 0.8 = 0.2
            random_state=42,
            stratify=labels[train_val_idx]
        )
        
        # 创建数据集
        train_dataset = QuantumTextDataset(
            [texts[i] for i in train_idx],
            labels[train_idx],
            vocab,
            self.max_len
        )
        
        val_dataset = QuantumTextDataset(
            [texts[i] for i in val_idx],
            labels[val_idx],
            vocab,
            self.max_len
        )
        
        test_dataset = QuantumTextDataset(
            [texts[i] for i in test_idx],
            labels[test_idx],
            vocab,
            self.max_len
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"\n训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, vocab 