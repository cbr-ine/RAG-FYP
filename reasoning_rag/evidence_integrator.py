"""证据整合与验证模块"""
from typing import List, Dict
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class EvidenceIntegrator:
    def __init__(self, max_evidence_length: int = 2000):
        self.max_evidence_length = max_evidence_length

    def integrate_and_validate(self, retrieval_results: Dict) -> Dict:
        """整合并验证证据"""
        logger.info("Integrating and validating evidence...")

        evidence_list = retrieval_results['total_evidence']

        # 按相似度排序
        sorted_evidence = sorted(evidence_list, key=lambda x: x['similarity'], reverse=True)

        # 证据质量评分
        scored_evidence = self._score_evidence(sorted_evidence, retrieval_results['subquery_results'])

        # 交叉验证
        validation_result = self._cross_validate(scored_evidence)

        # 选择最佳证据
        selected_evidence = self._select_best_evidence(scored_evidence, validation_result)

        # 推断整合方法
        integration_method = self._infer_integration_method(
            evidence_list, selected_evidence, validation_result
        )

        integration_result = {
            'evidence': selected_evidence,
            'validation': validation_result,
            'integration_method': integration_method,          # ← 新增字段
            'stats': {
                'total_evidence_count': len(evidence_list),
                'selected_evidence_count': len(selected_evidence),
                'avg_confidence': validation_result['avg_confidence'],
                'coverage_score': validation_result['coverage_score']
            }
        }

        logger.info(f"Selected {len(selected_evidence)} high-quality evidence items")
        logger.info(f"Average confidence: {validation_result['avg_confidence']:.3f}")
        logger.info(f"Integration method: {integration_method}")

        return integration_result

    def _infer_integration_method(self, evidence_list: List[Dict],
                                   selected_evidence: List[Dict],
                                   validation_result: Dict) -> str:
        """根据证据情况推断整合方法"""
        if not evidence_list:
            return "none"
        if len(evidence_list) == 1:
            return "direct"
        if validation_result.get('is_consistent', False) and len(selected_evidence) >= 3:
            return "weighted_fusion"
        if len(selected_evidence) >= 2:
            return "validated_selection"
        return "top_k_selection"

    def _score_evidence(self, evidence_list: List[Dict], subquery_results: List[Dict]) -> List[Dict]:
        """为证据打分"""
        scored_evidence = []

        for evidence in evidence_list:
            score = 0.0

            # 相似度分数 (权重0.4)
            score += evidence['similarity'] * 0.4

            # 跳数惩罚 (第一跳最重要,权重0.3)
            hop_score = max(0, 1.0 - (evidence['hop'] - 1) * 0.2)
            score += hop_score * 0.3

            # 文本长度分数 (适中长度更好,权重0.2)
            text_length = len(evidence['text'].split())
            if 50 <= text_length <= 200:
                length_score = 1.0
            elif text_length < 50:
                length_score = text_length / 50
            else:
                length_score = max(0.5, 1.0 - (text_length - 200) / 400)
            score += length_score * 0.2

            # 覆盖多个子查询加分 (权重0.1)
            coverage_score = min(1.0, sum(
                1 for sq in subquery_results
                if sq['subquery'][:30] in evidence['source_query'][:30]
            ) / len(subquery_results))
            score += coverage_score * 0.1

            evidence_with_score = evidence.copy()
            evidence_with_score['quality_score'] = score
            scored_evidence.append(evidence_with_score)

        # 重新排序
        scored_evidence.sort(key=lambda x: x['quality_score'], reverse=True)

        return scored_evidence

    def _cross_validate(self, evidence_list: List[Dict]) -> Dict:
        """交叉验证证据一致性"""
        if len(evidence_list) < 2:
            return {
                'is_consistent': True,
                'avg_confidence': evidence_list[0]['quality_score'] if evidence_list else 0.0,
                'coverage_score': 1.0 if evidence_list else 0.0,
                'contradictions': []
            }

        # 简化实现:基于重叠词汇评估一致性
        all_words = defaultdict(int)
        evidence_words = []

        for evidence in evidence_list[:10]:  # 只检查前10个证据
            words = set(evidence['text'].lower().split())
            evidence_words.append(words)
            for word in words:
                if len(word) > 3:  # 忽略短词
                    all_words[word] += 1

        # 计算词汇重叠度
        common_words = {word: count for word, count in all_words.items() if count >= 2}
        coverage_score = len(common_words) / max(len(all_words), 1)

        # 计算平均置信度
        avg_confidence = sum(e['quality_score'] for e in evidence_list[:5]) / min(len(evidence_list), 5)

        return {
            'is_consistent': coverage_score > 0.2,
            'avg_confidence': avg_confidence,
            'coverage_score': coverage_score,
            'contradictions': [],  # 简化实现,不检测矛盾
            'common_concepts_count': len(common_words)
        }

    def _select_best_evidence(self, scored_evidence: List[Dict], validation_result: Dict) -> List[Dict]:
        """选择最佳证据"""
        selected = []
        total_length = 0

        for evidence in scored_evidence:
            # 检查长度限制
            evidence_length = len(evidence['text'])
            if total_length + evidence_length > self.max_evidence_length:
                # 如果已经有一些证据了就停止,否则至少包含第一个
                if selected:
                    break
                else:
                    # 截断文本
                    remaining = self.max_evidence_length - total_length
                    evidence = evidence.copy()
                    evidence['text'] = evidence['text'][:remaining] + "..."
                    evidence_length = remaining

            selected.append(evidence)
            total_length += evidence_length

            # 至少选择3个高质量证据
            if len(selected) >= 3 and evidence['quality_score'] < 0.5:
                break

        return selected