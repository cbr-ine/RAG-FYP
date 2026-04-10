"""传统单轮 RAG 基线实现。"""
import logging
import time
from typing import Dict, List, Optional

from answer_generator import AnswerGenerator
from config import Config
from embedder import Embedder
from query_analyzer import QueryAnalyzer
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class TraditionalRAG:
    """不做分解、多跳和证据整合的传统 RAG 基线。"""

    def __init__(
        self,
        config: Config = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        system_name: str = "Traditional RAG",
        use_abstention: bool = False,
        keyword_fallback_top_k: Optional[int] = None
    ):
        self.config = config or Config()
        self.system_name = system_name
        self.use_abstention = use_abstention
        self.keyword_fallback_top_k = (
            self.config.KEYWORD_FALLBACK_TOP_K
            if keyword_fallback_top_k is None else keyword_fallback_top_k
        )

        logger.info(f"Initializing {self.system_name} baseline...")
        self.embedder = embedder or Embedder(self.config.EMBEDDING_MODEL)
        self.vector_store = vector_store or VectorStore(self.embedder.embedding_dim)
        self.query_analyzer = QueryAnalyzer(self.config.COMPLEXITY_THRESHOLD)
        self.answer_generator = AnswerGenerator(self.config.MIN_EVIDENCE_SIMILARITY)
        logger.info(f"{self.system_name} baseline initialized successfully")

    def get_experiment_config(self) -> Dict:
        return {
            'system_name': self.system_name,
            'use_decomposition': False,
            'use_multi_hop': False,
            'use_evidence_integration': False,
            'use_abstention': self.use_abstention,
            'keyword_fallback_top_k': self.keyword_fallback_top_k
        }

    def build_index(self, passages: List[Dict]):
        """构建向量索引。"""
        logger.info(f"Building index for {len(passages)} passages...")
        texts = [p['text'] for p in passages]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_passages(embeddings, passages)
        logger.info("Index built successfully")

    def _build_reasoning_path(self, analysis_result: Dict, evidence_count: int) -> List[Dict]:
        return [
            {
                'step': 1,
                'type': 'analysis',
                'description': (
                    f"Analyzed question as {analysis_result['query_type']} "
                    f"(complexity={analysis_result['complexity_score']:.2f})"
                )
            },
            {
                'step': 2,
                'type': 'retrieval',
                'description': (
                    f"Executed single dense retrieval over the original query and collected "
                    f"{evidence_count} evidence item(s)"
                )
            },
            {
                'step': 3,
                'type': 'generation',
                'description': (
                    f"Generated answer directly from top-k evidence "
                    f"with {'abstention enabled' if self.use_abstention else 'no abstention guard'}"
                )
            }
        ]

    def query(self, question: str, verbose: bool = True) -> Dict:
        """执行传统单轮 RAG 查询。"""
        start_time = time.perf_counter()
        if verbose:
            logger.info(f"Processing question with {self.system_name}: {question}")

        analysis_result = self.query_analyzer.analyze(question)
        subqueries = [{
            'subquery': question,
            'type': 'direct',
            'order': 1,
            'dependency': None
        }]

        query_embedding = self.embedder.embed_single(question)
        search_results = self.vector_store.search(
            query_embedding,
            self.config.TOP_K_RETRIEVAL,
            query_text=question,
            keyword_top_k=self.keyword_fallback_top_k
        )

        evidence = []
        for passage, similarity in search_results:
            evidence.append({
                'text': passage['text'],
                'similarity': similarity,
                'quality_score': similarity,
                'hop': 1,
                'source_query': question,
                'source_type': passage.get('source', 'unknown')
            })

        avg_similarity = (
            sum(item['similarity'] for item in evidence) / len(evidence)
            if evidence else 0.0
        )
        retrieval_results = {
            'subquery_results': [{
                'subquery': question,
                'order': 1,
                'type': 'direct',
                'hops': [{
                    'hop_number': 1,
                    'query': question,
                    'results_count': len(evidence)
                }],
                'evidence': evidence,
                'stats': {
                    'total_retrievals': 1,
                    'hops_performed': 1
                }
            }],
            'total_evidence': evidence,
            'retrieval_stats': {
                'total_retrievals': 1,
                'successful_retrievals': 1 if evidence else 0,
                'avg_similarity': avg_similarity,
                'high_quality_count': sum(
                    1 for item in evidence
                    if item['similarity'] >= self.config.SIMILARITY_THRESHOLD
                )
            }
        }

        integrated_evidence = {
            'evidence': evidence,
            'validation': {
                'is_consistent': True,
                'avg_confidence': avg_similarity,
                'coverage_score': 1.0 if evidence else 0.0,
                'contradictions': []
            },
            'integration_method': 'none',
            'stats': {
                'total_evidence_count': len(evidence),
                'selected_evidence_count': len(evidence),
                'avg_confidence': avg_similarity,
                'coverage_score': 1.0 if evidence else 0.0
            }
        }

        answer_result = self.answer_generator.generate(
            question,
            integrated_evidence,
            analysis_result,
            retrieval_results,
            use_abstention=self.use_abstention,
            reasoning_path=self._build_reasoning_path(analysis_result, len(evidence))
        )

        runtime_seconds = time.perf_counter() - start_time
        return {
            'question': question,
            'analysis': analysis_result,
            'subqueries': subqueries,
            'retrieval': retrieval_results,
            'evidence': integrated_evidence,
            'answer': answer_result,
            'metadata': {
                'system_name': self.system_name,
                'runtime_seconds': runtime_seconds,
                'llm_calls': answer_result.get('llm_calls', 0),
                'llm_provider': self.answer_generator.provider,
                'generation_model': self.answer_generator.model_name,
                'decomposition_model': None,
                'generation_llm_attempted': self.answer_generator.last_llm_attempted,
                'generation_llm_succeeded': self.answer_generator.last_llm_succeeded,
                'generation_llm_error': self.answer_generator.last_llm_error,
                'decomposition_llm_attempted': False,
                'decomposition_llm_succeeded': False,
                'decomposition_llm_error': None,
                'query_type': analysis_result.get('query_type', ''),
                'integration_method': 'none',
                'experiment_config': self.get_experiment_config()
            }
        }

    def save_index(self, filepath: str):
        """保存索引。"""
        self.vector_store.save(filepath, metadata={
            'embedding_model': self.embedder.model_name
        })

    def load_index(self, filepath: str):
        """加载索引。"""
        self.vector_store.load(filepath)
