"""问题分析与复杂度评估模块"""
from typing import Dict
import re
import logging

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(self, complexity_threshold: float = 0.2):
        self.complexity_threshold = complexity_threshold

        # 复杂问题的特征关键词
        self.complex_keywords = [
            'how', 'why', 'explain', 'describe', 'compare', 'contrast',
            'relationship', 'mechanism', 'process', 'multiple', 'various',
            'different', 'affect', 'influence', 'cause', 'effect'
        ]

        # 连接词
        self.conjunctions = ['and', 'or', 'but', 'as well as', 'in addition to']

    def _infer_query_type(self, question: str, features: Dict) -> str:
        """根据问题特征推断查询类型"""
        question_lower = question.lower()

        if features['is_multi_part']:
            return 'multi-part'
        if any(w in question_lower for w in ['compare', 'contrast', 'difference', 'versus', 'vs']):
            return 'comparative'
        if any(w in question_lower for w in ['why', 'cause', 'reason', 'effect', 'affect', 'influence']):
            return 'causal'
        if any(w in question_lower for w in ['how', 'mechanism', 'process', 'pathway']):
            return 'procedural'
        if any(w in question_lower for w in ['explain', 'describe', 'what is', 'define']):
            return 'descriptive'
        if features['has_conjunctions']:
            return 'compound'
        return 'factual'

    def analyze(self, question: str) -> Dict:
        """分析问题并评估复杂度"""
        question_lower = question.lower()

        # 计算复杂度分数
        complexity_score = 0.0
        features = {
            'has_complex_keywords': False,
            'has_multiple_clauses': False,
            'has_conjunctions': False,
            'question_length': len(question.split()),
            'is_multi_part': False
        }

        # 检查复杂关键词
        for keyword in self.complex_keywords:
            if keyword in question_lower:
                features['has_complex_keywords'] = True
                complexity_score += 0.2
                break

        # 检查连接词
        for conj in self.conjunctions:
            if conj in question_lower:
                features['has_conjunctions'] = True
                complexity_score += 0.15
                break

        # 检查问题长度
        word_count = len(question.split())
        if word_count > 15:
            complexity_score += 0.2
        elif word_count > 10:
            complexity_score += 0.1

        # 检查多个子句
        clause_count = len(re.split(r'[,;]', question))
        if clause_count > 2:
            features['has_multiple_clauses'] = True
            complexity_score += 0.15

        # 检查是否为多部分问题 (包含多个问号或明确的部分)
        if question.count('?') > 1 or any(marker in question_lower for marker in ['first', 'second', 'also', 'additionally']):
            features['is_multi_part'] = True
            complexity_score += 0.2

        # 限制最大分数为1.0
        complexity_score = min(complexity_score, 1.0)

        is_complex = complexity_score >= self.complexity_threshold

        # 根据复杂度分数推断推理深度
        if complexity_score < 0.2:
            reasoning_depth = "shallow"
        elif complexity_score < 0.5:
            reasoning_depth = "moderate"
        else:
            reasoning_depth = "deep"

        # 推断查询类型
        query_type = self._infer_query_type(question, features)

        result = {
            'question': question,
            'complexity_score': complexity_score,
            'is_complex': is_complex,
            'features': features,
            'requires_decomposition': is_complex,
            'reasoning_depth': reasoning_depth,
            'query_type': query_type           # ← 新增字段
        }

        logger.info(f"Question complexity: {complexity_score:.2f} ({'complex' if is_complex else 'simple'}), "
                    f"type: {query_type}, depth: {reasoning_depth}")

        return result