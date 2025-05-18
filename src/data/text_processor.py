"""
量子文本处理模块
"""

import pandas as pd
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from tqdm import tqdm
from typing import List, Dict, Union
import gensim
from gensim.utils import simple_preprocess

class QuantumTextProcessor:
    """量子文本处理器"""
    
    def __init__(self):
        """初始化文本处理器"""
        # 加载spaCy模型
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.nlp.tokenizer = self._custom_tokenizer(self.nlp)
        
        # 初始化词形还原器
        self.lemmatizer = WordNetLemmatizer()
        
        # 加载停用词
        self.stop_words = set(stopwords.words("english"))
        self.quantum_stop_words = {
            "present_invention", "invention", "claim_wherein", "wherein", "least_one",
            "et_al", "phys_rev", "allows_us", "also_discussed", "enables_us",
            "quantum", "non", "investigate", "result", "present", "give", "find",
            "paper", "new", "apply", "input", "mode", "property", "form",
            "implementation", "make", "achieve", "general", "low", "determine",
            "type", "correspond", "term", "realize", "compare", "configure"
        }
        
        # 初始化正则表达式模式
        self.patterns = {
            'quantum_terms': {
                'nisq': 'noisy_intermediate_scale_quantum',
                'squid': 'superconducting_quantum_interference_device',
                'qed': 'quantum_electrodynamics',
                '1WQC': 'one_way_quantum_computing'
            },
            'common_terms': {
                'computers': 'computer',
                'computations': 'computation',
                'computings': 'computing',
                'computation': 'computing',
                'computer': 'computing'
            }
        }
    
    def _custom_tokenizer(self, nlp):
        """创建自定义分词器"""
        infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        )
        infix_re = compile_infix_regex(infixes)
        return Tokenizer(
            nlp.vocab,
            prefix_search=nlp.tokenizer.prefix_search,
            suffix_search=nlp.tokenizer.suffix_search,
            infix_finditer=infix_re.finditer,
            token_match=nlp.tokenizer.token_match,
            rules=nlp.Defaults.tokenizer_exceptions
        )
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 基本清理
        text = text.replace("_", "@")
        text = re.sub(r"−|-", "_", text)
        text = re.sub(r"\r\n|\n", " ", text)
        text = re.sub(r"\\'|\\~|\\\"", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"_+", "_", text)
        text = re.sub(r" _ | _|_ ", "_", text)
        
        # 替换特定术语
        for term, replacement in self.patterns['quantum_terms'].items():
            text = text.replace(term, replacement)
        
        for term, replacement in self.patterns['common_terms'].items():
            text = text.replace(term, replacement)
        
        return text.lower().strip()
    
    def get_underscores(self, text: str) -> List[str]:
        """获取下划线连接的词"""
        return [i for i in text.replace(".", " ").replace(",", " ").split() if "_" in i]
    
    def get_lemma(self, phrase: str) -> str:
        """获取词形还原形式"""
        return "_".join(self.lemmatizer.lemmatize(word) for word in phrase.split("_"))
    
    def process_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理文档数据框"""
        # 合并标题和摘要
        df["text"] = df["title"] + ": " + df["abstract"]
        
        # 清理文本
        df["text_cleaned"] = df["text"].apply(self.clean_text)
        
        # 处理下划线词
        underscores = df["text_cleaned"].apply(self.get_underscores).values.tolist()
        underscores = pd.DataFrame(
            [item for sublist in underscores for item in sublist],
            columns=["underscore"]
        )
        
        # 统计频率
        underscores["frequency"] = 1
        underscores = underscores.groupby(["underscore"])["frequency"].sum().reset_index()
        underscores = underscores.sort_values(by="frequency", ascending=False)
        
        # 过滤低频词
        underscores = underscores[underscores["frequency"] >= 5]
        
        return df, underscores
    
    def tokenize_text(self, texts: List[str]) -> List[List[str]]:
        """分词"""
        return [simple_preprocess(str(text), deacc=True, min_len=2) for text in texts]
    
    def remove_stopwords(self, texts: List[List[str]]) -> List[List[str]]:
        """移除停用词"""
        return [[word for word in doc if word not in self.stop_words and word not in self.quantum_stop_words] 
                for doc in tqdm(texts)]
    
    def lemmatize_text(self, texts: List[List[str]]) -> List[List[str]]:
        """词形还原"""
        texts_out = []
        for sent in tqdm(texts):
            doc = self.nlp(" ".join(sent))
            doc_out = []
            for token in doc:
                if "_" in str(token):
                    doc_out.append(str(token))
                else:
                    doc_out.append(token.lemma_)
            texts_out.append(doc_out)
        return texts_out
    
    def process_for_training(self, df: pd.DataFrame) -> Dict:
        """处理数据用于训练"""
        # 处理文档
        df, underscores = self.process_document(df)
        
        # 分词
        data_words = self.tokenize_text(df["text_cleaned"].values.tolist())
        
        # 移除停用词
        data_words = self.remove_stopwords(data_words)
        
        # 词形还原
        data_words = self.lemmatize_text(data_words)
        
        return {
            "processed_texts": data_words,
            "underscores": underscores,
            "original_df": df
        } 