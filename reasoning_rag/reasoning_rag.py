"""主RAG系统集成"""
import logging
import time
from typing import Dict, List, Optional

from answer_generator import AnswerGenerator
from config import Config
from embedder import Embedder
from evidence_integrator import EvidenceIntegrator
from multi_hop_retriever import MultiHopRetriever
from query_analyzer import QueryAnalyzer
from query_decomposer import QueryDecomposer
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReasoningRAG:
    def __init__(
        self,
        config: Config = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        system_name: str = "Reasoning RAG",
        use_decomposition: bool = True,
        use_multi_hop: bool = True,
        use_evidence_integration: bool = True,
        use_abstention: bool = True,
        keyword_fallback_top_k: Optional[int] = None
    ):
        self.config = config or Config()
        self.system_name = system_name
        self.use_decomposition = use_decomposition
        self.use_multi_hop = use_multi_hop
        self.use_evidence_integration = use_evidence_integration
        self.use_abstention = use_abstention
        self.keyword_fallback_top_k = (
            self.config.KEYWORD_FALLBACK_TOP_K
            if keyword_fallback_top_k is None else keyword_fallback_top_k
        )

        logger.info(f"Initializing {self.system_name} system...")

        self.embedder = embedder or Embedder(self.config.EMBEDDING_MODEL)
        self.vector_store = vector_store or VectorStore(self.embedder.embedding_dim)
        self.query_analyzer = QueryAnalyzer(self.config.COMPLEXITY_THRESHOLD)
        self.query_decomposer = QueryDecomposer(self.config.MAX_SUBQUERIES)
        self.multi_hop_retriever = MultiHopRetriever(
            self.vector_store,
            self.embedder,
            self.config.TOP_K_RETRIEVAL,
            self.config.MAX_HOPS if self.use_multi_hop else 1,
            self.config.SIMILARITY_THRESHOLD,
            self.keyword_fallback_top_k,
            self.config.SHORT_QUERY_KEYWORD_THRESHOLD
        )
        self.evidence_integrator = EvidenceIntegrator(self.config.MAX_EVIDENCE_LENGTH)
        self.answer_generator = AnswerGenerator(self.config.MIN_EVIDENCE_SIMILARITY)

        logger.info(f"{self.system_name} system initialized successfully")

    def get_experiment_config(self) -> Dict:
        """返回当前系统的实验配置，便于对比实验记录。"""
        return {
            'system_name': self.system_name,
            'use_decomposition': self.use_decomposition,
            'use_multi_hop': self.use_multi_hop,
            'use_evidence_integration': self.use_evidence_integration,
            'use_abstention': self.use_abstention,
            'keyword_fallback_top_k': self.keyword_fallback_top_k
        }

    def build_index(self, passages: List[Dict]):
        """构建向量索引"""
        logger.info(f"Building index for {len(passages)} passages...")
        texts = [p['text'] for p in passages]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_passages(embeddings, passages)
        logger.info("Index built successfully")

    def _build_direct_subquery(self, question: str) -> List[Dict]:
        return [{
            'subquery': question,
            'type': 'direct',
            'order': 1,
            'dependency': None
        }]

    def _retrieve(self, subqueries: List[Dict]) -> Dict:
        if self.use_multi_hop:
            return self.multi_hop_retriever.retrieve_for_subqueries(subqueries)
        return self._single_hop_retrieve(subqueries)

    def _single_hop_retrieve(self, subqueries: List[Dict]) -> Dict:
        """仅执行单次检索，用于 multi-hop 消融实验。"""
        all_results = {
            'subquery_results': [],
            'total_evidence': [],
            'retrieval_stats': {
                'total_retrievals': 0,
                'successful_retrievals': 0,
                'avg_similarity': 0.0,
                'high_quality_count': 0
            }
        }

        all_similarities = []
        for subquery_info in subqueries:
            query_embedding = self.embedder.embed_single(subquery_info['subquery'])
            search_results = self.vector_store.search(
                query_embedding,
                self.config.TOP_K_RETRIEVAL,
                query_text=subquery_info['subquery'],
                keyword_top_k=self.keyword_fallback_top_k
            )
            query_is_short = (
                len(subquery_info['subquery'].split()) <= 4
                or len(subquery_info['subquery'].strip()) <= 24
            )
            threshold = (
                self.config.SHORT_QUERY_KEYWORD_THRESHOLD
                if query_is_short else self.config.SIMILARITY_THRESHOLD
            )
            filtered_results = [
                (passage, similarity) for passage, similarity in search_results
                if similarity >= threshold
            ]

            hop_evidence = []
            for passage, similarity in filtered_results:
                evidence_item = {
                    'text': passage['text'],
                    'similarity': similarity,
                    'hop': 1,
                    'source_query': subquery_info['subquery'],
                    'source_type': passage.get('source', 'unknown')
                }
                hop_evidence.append(evidence_item)

            subquery_result = {
                'subquery': subquery_info['subquery'],
                'order': subquery_info['order'],
                'type': subquery_info['type'],
                'hops': [{
                    'hop_number': 1,
                    'query': subquery_info['subquery'],
                    'results_count': len(hop_evidence)
                }],
                'evidence': hop_evidence,
                'stats': {
                    'total_retrievals': 1,
                    'hops_performed': 1
                }
            }
            all_results['subquery_results'].append(subquery_result)
            all_results['total_evidence'].extend(hop_evidence)
            all_results['retrieval_stats']['total_retrievals'] += 1

            if hop_evidence:
                all_results['retrieval_stats']['successful_retrievals'] += 1
                all_similarities.extend(item['similarity'] for item in hop_evidence)

        all_results['total_evidence'] = self.multi_hop_retriever._deduplicate_evidence(
            all_results['total_evidence']
        )

        if all_similarities:
            all_results['retrieval_stats']['avg_similarity'] = (
                sum(all_similarities) / len(all_similarities)
            )

        all_results['retrieval_stats']['high_quality_count'] = sum(
            1 for item in all_results['total_evidence']
            if item['similarity'] >= self.config.SIMILARITY_THRESHOLD
        )
        return all_results

    def _integrate(self, retrieval_results: Dict) -> Dict:
        if self.use_evidence_integration:
            return self.evidence_integrator.integrate_and_validate(retrieval_results)
        return self._simple_integrate(retrieval_results)

    def _simple_integrate(self, retrieval_results: Dict) -> Dict:
        """简单整合检索结果，用于 evidence integration 消融实验。"""
        sorted_evidence = sorted(
            retrieval_results['total_evidence'],
            key=lambda item: item['similarity'],
            reverse=True
        )

        selected = []
        total_length = 0
        for evidence in sorted_evidence:
            evidence_copy = evidence.copy()
            evidence_copy['quality_score'] = evidence_copy['similarity']
            evidence_length = len(evidence_copy['text'])
            if total_length + evidence_length > self.config.MAX_EVIDENCE_LENGTH and selected:
                break
            selected.append(evidence_copy)
            total_length += evidence_length

        avg_similarity = (
            sum(item['similarity'] for item in selected) / len(selected)
            if selected else 0.0
        )

        return {
            'evidence': selected,
            'validation': {
                'is_consistent': True,
                'avg_confidence': avg_similarity,
                'coverage_score': 1.0 if selected else 0.0,
                'contradictions': []
            },
            'integration_method': 'top_k_only',
            'stats': {
                'total_evidence_count': len(retrieval_results['total_evidence']),
                'selected_evidence_count': len(selected),
                'avg_confidence': avg_similarity,
                'coverage_score': 1.0 if selected else 0.0
            }
        }

    def _build_reasoning_path(self, analysis_result: Dict, retrieval_results: Dict,
                              integrated_evidence: Dict) -> List[Dict]:
        total_hops = sum(
            sq['stats']['hops_performed']
            for sq in retrieval_results.get('subquery_results', [])
        )
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
                'type': 'decomposition',
                'description': (
                    f"Used {'query decomposition' if self.use_decomposition else 'direct single-query routing'} "
                    f"with {len(retrieval_results.get('subquery_results', []))} retrieval unit(s)"
                )
            },
            {
                'step': 3,
                'type': 'retrieval',
                'description': (
                    f"Executed {'multi-hop' if self.use_multi_hop else 'single-hop'} retrieval "
                    f"({total_hops} hop(s), {retrieval_results['retrieval_stats']['total_retrievals']} retrieval call(s))"
                )
            },
            {
                'step': 4,
                'type': 'integration',
                'description': (
                    f"Used {integrated_evidence.get('integration_method', 'unknown')} evidence integration "
                    f"over {len(integrated_evidence.get('evidence', []))} evidence item(s)"
                )
            },
            {
                'step': 5,
                'type': 'generation',
                'description': (
                    f"Generated answer with {'abstention enabled' if self.use_abstention else 'no abstention guard'}"
                )
            }
        ]

    def _estimate_decomposition_llm_calls(self, subqueries: List[Dict]) -> int:
        if not self.use_decomposition or not self.query_decomposer.client:
            return 0

        rule_based_types = {
            'direct', 'conjunction_split', 'punctuation_split',
            'main_query', 'mechanism_query'
        }
        if any(item.get('type') not in rule_based_types for item in subqueries):
            return 1
        return 0

    def query(self, question: str, verbose: bool = True) -> Dict:
        """执行完整的RAG查询流程"""
        start_time = time.perf_counter()
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing question with {self.system_name}: {question}")
            logger.info(f"{'='*80}\n")

        analysis_result = self.query_analyzer.analyze(question)
        if self.use_decomposition:
            subqueries = self.query_decomposer.decompose(analysis_result)
        else:
            subqueries = self._build_direct_subquery(question)

        retrieval_results = self._retrieve(subqueries)
        integrated_evidence = self._integrate(retrieval_results)
        reasoning_path = self._build_reasoning_path(
            analysis_result,
            retrieval_results,
            integrated_evidence
        )
        answer_result = self.answer_generator.generate(
            question,
            integrated_evidence,
            analysis_result,
            retrieval_results,
            use_abstention=self.use_abstention,
            reasoning_path=reasoning_path
        )

        runtime_seconds = time.perf_counter() - start_time
        total_llm_calls = (
            self._estimate_decomposition_llm_calls(subqueries)
            + answer_result.get('llm_calls', 0)
        )

        complete_result = {
            'question': question,
            'analysis': analysis_result,
            'subqueries': subqueries,
            'retrieval': retrieval_results,
            'evidence': integrated_evidence,
            'answer': answer_result,
            'metadata': {
                'system_name': self.system_name,
                'runtime_seconds': runtime_seconds,
                'llm_calls': total_llm_calls,
                'llm_provider': self.answer_generator.provider or self.query_decomposer.provider,
                'generation_model': self.answer_generator.model_name,
                'decomposition_model': self.query_decomposer.model_name if self.use_decomposition else None,
                'generation_llm_attempted': self.answer_generator.last_llm_attempted,
                'generation_llm_succeeded': self.answer_generator.last_llm_succeeded,
                'generation_llm_error': self.answer_generator.last_llm_error,
                'decomposition_llm_attempted': self.query_decomposer.last_llm_attempted if self.use_decomposition else False,
                'decomposition_llm_succeeded': self.query_decomposer.last_llm_succeeded if self.use_decomposition else False,
                'decomposition_llm_error': self.query_decomposer.last_llm_error if self.use_decomposition else None,
                'query_type': analysis_result.get('query_type', ''),
                'integration_method': integrated_evidence.get('integration_method', ''),
                'experiment_config': self.get_experiment_config()
            }
        }

        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(
                f"Query processing completed in {runtime_seconds:.3f}s "
                f"using {self.system_name}"
            )
            logger.info(f"{'='*80}\n")

        return complete_result

    def save_index(self, filepath: str):
        """保存索引"""
        self.vector_store.save(filepath, metadata={
            'embedding_model': self.embedder.model_name
        })

    def load_index(self, filepath: str):
        """加载索引"""
        self.vector_store.load(filepath)
        saved_model = self.vector_store.metadata.get('embedding_model')
        if saved_model and saved_model != self.embedder.model_name:
            logger.warning(
                "Index embedding model mismatch: index built with %s but current model is %s. "
                "Please rebuild or overwrite the index for best retrieval quality.",
                saved_model,
                self.embedder.model_name
            )