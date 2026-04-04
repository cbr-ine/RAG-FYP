"""向量存储模块"""
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.passages = []

    def add_passages(self, embeddings: np.ndarray, passages: List[Dict]):
        """添加段落到向量库"""
        logger.info(f"Adding {len(passages)} passages to vector store...")
        self.index.add(embeddings.astype('float32'))
        self.passages.extend(passages)
        logger.info(f"Vector store now contains {len(self.passages)} passages")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """搜索最相关的段落"""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.passages):
                # 转换距离为相似度分数 (0-1)
                similarity = 1 / (1 + distance)
                results.append((self.passages[idx], similarity))

        return results

    def save(self, filepath: str):
        """保存向量库"""
        logger.info(f"Saving vector store to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'passages': self.passages
            }, f)

    def load(self, filepath: str):
        """加载向量库"""
        logger.info(f"Loading vector store from {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index'])
            self.passages = data['passages']