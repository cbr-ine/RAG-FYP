"""多跳检索模块"""
from typing import List, Dict, Tuple
import logging
from vector_store import VectorStore
from embedder import Embedder

logger = logging.getLogger(__name__)

class MultiHopRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder,
                 top_k: int = 5, max_hops: int = 3,
                 similarity_threshold: float = 0.5,
                 keyword_fallback_top_k: int = 8,
                 short_query_keyword_threshold: float = 0.25):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.max_hops = max_hops
        self.similarity_threshold = similarity_threshold
        self.keyword_fallback_top_k = keyword_fallback_top_k
        self.short_query_keyword_threshold = short_query_keyword_threshold

    def retrieve_for_subqueries(self, subqueries: List[Dict]) -> Dict:
        """为所有子查询执行检索"""
        logger.info(f"Retrieving for {len(subqueries)} subqueries...")

        all_results = {
            'subquery_results': [],
            'total_evidence': [],
            'retrieval_stats': {
                'total_retrievals': 0,
                'successful_retrievals': 0,
                'avg_similarity': 0.0,
                'high_quality_count': 0      # 新增：高质量证据数量
            }
        }

        all_similarities = []

        for subquery_info in subqueries:
            subquery = subquery_info['subquery']
            logger.info(f"\nProcessing subquery {subquery_info['order']}: {subquery}")

            hop_results = self._multi_hop_search(subquery, subquery_info)

            subquery_result = {
                'subquery': subquery,
                'order': subquery_info['order'],
                'type': subquery_info['type'],
                'hops': hop_results['hops'],
                'evidence': hop_results['evidence'],
                'stats': hop_results['stats']
            }

            all_results['subquery_results'].append(subquery_result)
            all_results['total_evidence'].extend(hop_results['evidence'])
            all_results['retrieval_stats']['total_retrievals'] += \
                hop_results['stats']['total_retrievals']

            if hop_results['evidence']:
                all_results['retrieval_stats']['successful_retrievals'] += 1
                all_similarities.extend([e['similarity'] for e in hop_results['evidence']])

        if all_similarities:
            all_results['retrieval_stats']['avg_similarity'] = (
                sum(all_similarities) / len(all_similarities)
            )

        # 去重
        all_results['total_evidence'] = self._deduplicate_evidence(
            all_results['total_evidence']
        )

        # 统计高质量证据数量（供 main.py 展示）
        high_quality = [
            e for e in all_results['total_evidence']
            if e['similarity'] >= self.similarity_threshold
        ]
        all_results['retrieval_stats']['high_quality_count'] = len(high_quality)

        logger.info(f"\nTotal unique evidence collected: {len(all_results['total_evidence'])}")
        logger.info(
            f"High quality (similarity >= {self.similarity_threshold}): {len(high_quality)}"
        )
        logger.info(
            f"Successful retrievals: "
            f"{all_results['retrieval_stats']['successful_retrievals']}/{len(subqueries)}"
        )

        return all_results

    def _multi_hop_search(self, query: str, subquery_info: Dict) -> Dict:
        """执行多跳检索"""
        result = {
            'hops': [],
            'evidence': [],
            'stats': {
                'total_retrievals': 0,
                'hops_performed': 0
            }
        }

        current_query = query
        retrieved_passages = set()

        for hop in range(self.max_hops):
            logger.info(f"  Hop {hop + 1}: {current_query[:100]}...")

            query_embedding = self.embedder.embed_single(current_query)
            search_results = self.vector_store.search(
                query_embedding,
                self.top_k,
                query_text=current_query,
                keyword_top_k=self.keyword_fallback_top_k
            )
            result['stats']['total_retrievals'] += 1

            # 过滤低相似度结果
            query_is_short = len(current_query.split()) <= 4 or len(current_query.strip()) <= 24
            threshold = (
                self.short_query_keyword_threshold
                if query_is_short else self.similarity_threshold
            )
            filtered_results = [
                (passage, sim) for passage, sim in search_results
                if sim >= threshold
            ]

            if not filtered_results:
                logger.info(
                    f"  No results above threshold ({threshold}) "
                    f"at hop {hop + 1}"
                )
                result['hops'].append({
                    'hop_number': hop + 1,
                    'query': current_query,
                    'results_count': 0
                })
                result['stats']['hops_performed'] += 1
                break

            hop_evidence = []
            for passage, similarity in filtered_results:
                passage_id = (
                    f"{passage.get('question_id', '')}_{passage.get('passage_id', '')}"
                    f"_{passage.get('source', '')}"
                )

                if passage_id not in retrieved_passages:
                    retrieved_passages.add(passage_id)
                    evidence_item = {
                        'text': passage['text'],
                        'similarity': similarity,
                        'hop': hop + 1,
                        'source_query': current_query,
                        'source_type': passage.get('source', 'unknown')
                    }
                    hop_evidence.append(evidence_item)
                    result['evidence'].append(evidence_item)

            result['hops'].append({
                'hop_number': hop + 1,
                'query': current_query,
                'results_count': len(hop_evidence)
            })
            result['stats']['hops_performed'] += 1

            logger.info(f"  Found {len(hop_evidence)} new passages at hop {hop + 1}")

            # 修复过早终止：只有找到足够"高置信度"证据才提前停止
            # 原来的 "hop==0 and len>=3" 逻辑会导致接受无关结果后就停止
            high_conf_count = sum(1 for e in hop_evidence if e['similarity'] >= 0.65)
            if high_conf_count >= 3:
                logger.info(
                    f"  Found {high_conf_count} high-confidence passages, stopping early."
                )
                break

            # 为下一跳扩展查询
            if hop < self.max_hops - 1 and hop_evidence:
                top_passage_text = hop_evidence[0]['text']
                words = top_passage_text.split()[:50]
                current_query = query + " " + " ".join(words)
            else:
                break

        return result

    def _deduplicate_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        """去重证据"""
        seen_texts = set()
        unique_evidence = []

        for evidence in evidence_list:
            text_hash = hash(evidence['text'])
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_evidence.append(evidence)

        return unique_evidence