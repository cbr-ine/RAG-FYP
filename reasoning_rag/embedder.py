"""文本嵌入模块"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """批量嵌入文本"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """嵌入单个文本"""
        return self.model.encode(text, convert_to_numpy=True)