"""向量存储模块"""
import logging
import pickle
import re
from typing import Dict, List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.passages = []
        self.metadata = {}

    def add_passages(self, embeddings: np.ndarray, passages: List[Dict]):
        """添加段落到向量库"""
        logger.info(f"Adding {len(passages)} passages to vector store...")
        self.index.add(embeddings.astype('float32'))
        self.passages.extend(passages)
        logger.info(f"Vector store now contains {len(self.passages)} passages")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower())

    def _keyword_search(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not query_text or not self.passages:
            return []

        query_text_lower = query_text.lower().strip()
        query_tokens = self._tokenize(query_text_lower)
        query_token_set = set(query_tokens)
        short_query = len(query_text_lower) <= 24 or len(query_tokens) <= 3

        scored = []
        for passage in self.passages:
            text = str(passage.get("text", ""))
            if not text:
                continue

            text_lower = text.lower()
            text_tokens = set(self._tokenize(text_lower))
            overlap_score = 0.0
            substring_bonus = 0.0

            if query_token_set:
                overlap_score = len(query_token_set.intersection(text_tokens)) / len(query_token_set)

            if query_text_lower and query_text_lower in text_lower:
                substring_bonus = 1.0 if short_query else 0.7

            keyword_score = max(overlap_score, substring_bonus)
            if keyword_score > 0:
                scored.append((passage, min(keyword_score, 1.0)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               query_text: str = "", keyword_top_k: int = 8) -> List[Tuple[Dict, float]]:
        """搜索最相关的段落，支持 dense + keyword fallback。"""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        dense_results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.passages):
                # 转换距离为相似度分数 (0-1)
                similarity = float(1 / (1 + distance))
                dense_results.append((self.passages[idx], similarity))

        keyword_results = []
        if keyword_top_k and keyword_top_k > 0:
            keyword_results = self._keyword_search(query_text, top_k=keyword_top_k)
        combined = {}

        for passage, similarity in dense_results:
            passage_key = passage.get("id", hash(passage.get("text", "")))
            combined[passage_key] = (passage, similarity)

        for passage, keyword_score in keyword_results:
            passage_key = passage.get("id", hash(passage.get("text", "")))
            existing = combined.get(passage_key)
            blended_score = keyword_score * 0.92
            if existing:
                passage_obj, existing_score = existing
                combined[passage_key] = (passage_obj, max(existing_score, blended_score))
            else:
                combined[passage_key] = (passage, blended_score)

        merged_results = list(combined.values())
        merged_results.sort(key=lambda item: item[1], reverse=True)
        limit = max(top_k, keyword_top_k or 0)
        return merged_results[:limit]

    def save(self, filepath: str, metadata: Dict = None):
        """保存向量库"""
        logger.info(f"Saving vector store to {filepath}")
        self.metadata = metadata or self.metadata or {}
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'passages': self.passages,
                'metadata': self.metadata
            }, f)

    def clone(self) -> "VectorStore":
        """Create an in-memory copy of the current vector store."""
        cloned = VectorStore(self.dimension)
        cloned.index = faiss.deserialize_index(faiss.serialize_index(self.index))
        cloned.passages = [passage.copy() if isinstance(passage, dict) else passage for passage in self.passages]
        cloned.metadata = dict(self.metadata)
        return cloned

    def load(self, filepath: str):
        """加载向量库"""
        logger.info(f"Loading vector store from {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index'])
            self.passages = data['passages']
            self.metadata = data.get('metadata', {})