"""评估指标计算模块"""
import logging
from typing import List, Dict
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.metrics = defaultdict(list)

    def evaluate_batch(self, results: List[Dict], ground_truth: List[Dict]) -> Dict:
        """批量评估RAG系统性能"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {len(results)} queries...")
        logger.info(f"{'='*80}\n")

        metrics = {
            'retrieval_success_rate': 0.0,
            'avg_evidence_count': 0.0,
            'avg_similarity_score': 0.0,
            'avg_confidence': 0.0,
            'avg_hops': 0.0,
            'complexity_distribution': {'simple': 0, 'complex': 0},
            'decomposition_rate': 0.0,
            'avg_subqueries': 0.0,
            'answer_coverage': 0.0,
            'total_queries': len(results)
        }

        successful_retrievals = 0
        total_evidence = 0
        total_similarity = 0
        total_confidence = 0
        total_hops = 0
        total_subqueries = 0
        decomposed_queries = 0
        answer_coverages = []

        for i, (result, gt) in enumerate(zip(results, ground_truth)):
            # 检索成功率
            evidence_count = len(result['evidence']['evidence'])
            if evidence_count > 0:
                successful_retrievals += 1

            total_evidence += evidence_count

            # 相似度分数
            if result['retrieval']['retrieval_stats']['avg_similarity'] > 0:
                total_similarity += result['retrieval']['retrieval_stats']['avg_similarity']

            # 置信度
            total_confidence += result['answer']['confidence']

            # 跳数统计
            for sq_result in result['retrieval']['subquery_results']:
                total_hops += sq_result['stats']['hops_performed']

            # 复杂度分布
            if result['analysis']['is_complex']:
                metrics['complexity_distribution']['complex'] += 1
            else:
                metrics['complexity_distribution']['simple'] += 1

            # 分解率
            if result['analysis']['requires_decomposition']:
                decomposed_queries += 1

            # 子查询数量
            total_subqueries += len(result['subqueries'])

            # 答案覆盖率 (简化计算)
            coverage = self._calculate_answer_coverage(result, gt)
            answer_coverages.append(coverage)

        # 计算平均值
        metrics['retrieval_success_rate'] = successful_retrievals / len(results)
        metrics['avg_evidence_count'] = total_evidence / len(results)
        metrics['avg_similarity_score'] = total_similarity / len(results)
        metrics['avg_confidence'] = total_confidence / len(results)
        metrics['avg_hops'] = total_hops / len(results)
        metrics['decomposition_rate'] = decomposed_queries / len(results)
        metrics['avg_subqueries'] = total_subqueries / len(results)
        metrics['answer_coverage'] = np.mean(answer_coverages) if answer_coverages else 0.0

        # 打印详细指标
        self._print_metrics(metrics)

        return metrics

    def _calculate_answer_coverage(self, result: Dict, ground_truth: Dict) -> float:
        """计算答案覆盖率"""
        # 简化实现:计算答案与ground truth的词汇重叠
        answer = result['answer']['answer'].lower()

        # 获取ground truth答案
        gt_answers = ground_truth.get('answers', [])
        if not gt_answers or not answer:
            return 0.0

        # 使用第一个ground truth答案
        gt_text = str(gt_answers[0]).lower() if gt_answers else ""

        # 计算词汇重叠
        answer_words = set(answer.split())
        gt_words = set(gt_text.split())

        if not gt_words:
            return 0.0

        intersection = answer_words.intersection(gt_words)
        coverage = len(intersection) / len(gt_words)

        return min(coverage, 1.0)

    def _print_metrics(self, metrics: Dict):
        """打印评估指标"""
        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION METRICS")
        logger.info(f"{'='*80}\n")

        logger.info(f"📊 Retrieval Performance:")
        logger.info(f"  ✓ Success Rate:           {metrics['retrieval_success_rate']:.2%}")
        logger.info(f"  ✓ Avg Evidence Count:     {metrics['avg_evidence_count']:.2f}")
        logger.info(f"  ✓ Avg Similarity Score:   {metrics['avg_similarity_score']:.3f}")

        logger.info(f"\n🎯 Query Analysis:")
        logger.info(f"  ✓ Simple Queries:         {metrics['complexity_distribution']['simple']} ({metrics['complexity_distribution']['simple']/metrics['total_queries']:.1%})")
        logger.info(f"  ✓ Complex Queries:        {metrics['complexity_distribution']['complex']} ({metrics['complexity_distribution']['complex']/metrics['total_queries']:.1%})")
        logger.info(f"  ✓ Decomposition Rate:     {metrics['decomposition_rate']:.2%}")
        logger.info(f"  ✓ Avg Subqueries:         {metrics['avg_subqueries']:.2f}")

        logger.info(f"\n🔍 Multi-Hop Retrieval:")
        logger.info(f"  ✓ Avg Hops per Query:     {metrics['avg_hops']:.2f}")

        logger.info(f"\n💡 Answer Quality:")
        logger.info(f"  ✓ Avg Confidence:         {metrics['avg_confidence']:.3f}")
        logger.info(f"  ✓ Answer Coverage:        {metrics['answer_coverage']:.2%}")

        logger.info(f"\n📈 Overall:")
        logger.info(f"  ✓ Total Queries:          {metrics['total_queries']}")

        logger.info(f"\n{'='*80}\n")