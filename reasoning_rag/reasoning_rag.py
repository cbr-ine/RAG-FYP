"""主RAG系统集成"""
import logging
from typing import Dict, List
from config import Config
from embedder import Embedder
from vector_store import VectorStore
from query_analyzer import QueryAnalyzer
from query_decomposer import QueryDecomposer
from multi_hop_retriever import MultiHopRetriever
from evidence_integrator import EvidenceIntegrator
from answer_generator import AnswerGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReasoningRAG:
    def __init__(self, config: Config = None):
        self.config = config or Config()

        # 初始化所有组件
        logger.info("Initializing Reasoning RAG system...")

        self.embedder = Embedder(self.config.EMBEDDING_MODEL)
        self.vector_store = VectorStore(self.embedder.embedding_dim)
        self.query_analyzer = QueryAnalyzer(self.config.COMPLEXITY_THRESHOLD)
        self.query_decomposer = QueryDecomposer(self.config.MAX_SUBQUERIES)
        self.multi_hop_retriever = MultiHopRetriever(
            self.vector_store,
            self.embedder,
            self.config.TOP_K_RETRIEVAL,
            self.config.MAX_HOPS,
            self.config.SIMILARITY_THRESHOLD
        )
        self.evidence_integrator = EvidenceIntegrator(self.config.MAX_EVIDENCE_LENGTH)
        self.answer_generator = AnswerGenerator()

        logger.info("Reasoning RAG system initialized successfully")

    def build_index(self, passages: List[Dict]):
        """构建向量索引"""
        logger.info(f"Building index for {len(passages)} passages...")

        # 提取文本
        texts = [p['text'] for p in passages]

        # 生成嵌入
        embeddings = self.embedder.embed_texts(texts)

        # 添加到向量库
        self.vector_store.add_passages(embeddings, passages)

        logger.info("Index built successfully")

    def query(self, question: str, verbose: bool = True) -> Dict:
        """执行完整的RAG查询流程"""
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing question: {question}")
            logger.info(f"{'='*80}\n")

        # 步骤1: 问题分析
        analysis_result = self.query_analyzer.analyze(question)

        # 步骤2: 查询分解 (如果需要)
        subqueries = self.query_decomposer.decompose(analysis_result)

        # 步骤3: 多跳检索
        retrieval_results = self.multi_hop_retriever.retrieve_for_subqueries(subqueries)

        # 步骤4: 证据整合与验证
        integrated_evidence = self.evidence_integrator.integrate_and_validate(retrieval_results)

        # 步骤5: 答案生成
        answer_result = self.answer_generator.generate(
            question,
            integrated_evidence,
            analysis_result,
            retrieval_results
        )

        # 组合完整结果
        complete_result = {
            'question': question,
            'analysis': analysis_result,
            'subqueries': subqueries,
            'retrieval': retrieval_results,
            'evidence': integrated_evidence,
            'answer': answer_result
        }

        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"Query processing completed")
            logger.info(f"{'='*80}\n")

        return complete_result

    def save_index(self, filepath: str):
        """保存索引"""
        self.vector_store.save(filepath)

    def load_index(self, filepath: str):
        """加载索引"""
        self.vector_store.load(filepath)