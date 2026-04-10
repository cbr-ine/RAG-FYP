"""子查询分解模块 - 使用 OpenAI LLM."""
from typing import List, Dict
import logging
from llm_provider import get_llm_client, get_model_name, get_token_limit_kwargs

logger = logging.getLogger(__name__)

class QueryDecomposer:
    def __init__(self, max_subqueries: int = 4):
        self.max_subqueries = max_subqueries
        self.client, self.provider = get_llm_client()
        self.model_name = get_model_name("decomposition", self.provider)
        self.last_llm_attempted = False
        self.last_llm_succeeded = False
        self.last_llm_error = None
        if not self.client or not self.model_name:
            logger.warning(
                "OPENAI_API_KEY not found in environment. "
                "Using rule-based decomposition."
            )
        else:
            logger.info(
                f"Query decomposition LLM enabled: provider={self.provider}, model={self.model_name}"
            )

    def decompose(self, analysis_result: Dict) -> List[Dict]:
        """将复杂问题分解为子查询"""
        self.last_llm_attempted = False
        self.last_llm_succeeded = False
        self.last_llm_error = None
        question = analysis_result['question']

        if not analysis_result['requires_decomposition']:
            # 简单问题不需要分解
            return [{
                'subquery': question,
                'type': 'direct',
                'order': 1,
                'dependency': None
            }]

        logger.info(f"Decomposing complex question: {question}")

        # 尝试使用 LLM 分解
        if self.client:
            try:
                self.last_llm_attempted = True
                subqueries = self._decompose_with_llm(question, analysis_result)
                if subqueries:
                    self.last_llm_succeeded = True
                    logger.info(f"LLM generated {len(subqueries)} subqueries")
                    return subqueries
            except Exception as e:
                self.last_llm_error = str(e)
                logger.error(f"LLM decomposition failed: {e}. Falling back to rule-based.")

        # 回退到基于规则的分解
        return self._decompose_with_rules(question, analysis_result)

    def _decompose_with_llm(self, question: str, analysis_result: Dict) -> List[Dict]:
        """使用 OpenAI LLM 分解问题"""

        prompt = f"""You are an expert at breaking down complex questions into simpler sub-questions for a retrieval system.

Given the following question, decompose it into 2-4 logical sub-questions that can be answered independently and then combined.

Original Question: {question}

Question Complexity Score: {analysis_result['complexity_score']:.2f}
Question Features:
- Has complex keywords: {analysis_result['features']['has_complex_keywords']}
- Multiple clauses: {analysis_result['features']['has_multiple_clauses']}
- Has conjunctions: {analysis_result['features']['has_conjunctions']}

Instructions:
1. Create 2-4 sub-questions that are simpler and more focused
2. Each sub-question should be self-contained and answerable
3. The sub-questions should logically cover the original question
4. Order them from foundational to more specific
5. Return ONLY a JSON array of sub-questions in this exact format:

[
  {{"subquery": "first sub-question?", "type": "foundational", "order": 1}},
  {{"subquery": "second sub-question?", "type": "specific", "order": 2}}
]

Do not include any explanation or additional text. Only return the JSON array."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a precise question decomposition assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            **get_token_limit_kwargs(self.model_name, 500)
        )

        # 解析响应
        result_text = response.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块标记
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        # 解析 JSON
        import json
        subqueries_raw = json.loads(result_text)

        # 转换为标准格式
        subqueries = []
        for i, sq in enumerate(subqueries_raw[:self.max_subqueries], 1):
            subqueries.append({
                'subquery': sq['subquery'],
                'type': sq.get('type', 'llm_generated'),
                'order': i,
                'dependency': None if i == 1 else i - 1
            })

        # 打印生成的子查询
        for i, sq in enumerate(subqueries, 1):
            logger.info(f"  Subquery {i}: {sq['subquery']}")

        return subqueries

    def _decompose_with_rules(self, question: str, analysis_result: Dict) -> List[Dict]:
        """基于规则的分解 (回退方案)"""
        import re

        subqueries = []

        # 策略1: 基于连接词分割
        if analysis_result['features']['has_conjunctions']:
            subqueries.extend(self._split_by_conjunctions(question))

        # 策略2: 基于逗号和分号分割
        if analysis_result['features']['has_multiple_clauses'] and len(subqueries) == 0:
            subqueries.extend(self._split_by_punctuation(question))

        # 策略3: 提取核心概念查询
        if len(subqueries) == 0:
            subqueries.extend(self._extract_core_concepts(question))

        # 如果所有策略都失败,返回原问题
        if len(subqueries) == 0:
            subqueries = [{
                'subquery': question,
                'type': 'direct',
                'order': 1,
                'dependency': None
            }]

        # 限制子查询数量
        subqueries = subqueries[:self.max_subqueries]

        logger.info(f"Generated {len(subqueries)} subqueries using rules")
        for i, sq in enumerate(subqueries, 1):
            logger.info(f"  Subquery {i}: {sq['subquery']}")

        return subqueries

    def _split_by_conjunctions(self, question: str) -> List[Dict]:
        """基于连接词分割问题"""
        import re
        subqueries = []
        patterns = [r'\band\b', r'\bor\b', r'\bas well as\b']

        for pattern in patterns:
            parts = re.split(pattern, question, flags=re.IGNORECASE)
            if len(parts) > 1:
                for i, part in enumerate(parts, 1):
                    part = part.strip()
                    if part and len(part) > 5:
                        if not part.endswith('?'):
                            part += '?'
                        subqueries.append({
                            'subquery': part,
                            'type': 'conjunction_split',
                            'order': i,
                            'dependency': None if i == 1 else i - 1
                        })
                break

        return subqueries

    def _split_by_punctuation(self, question: str) -> List[Dict]:
        """基于标点符号分割问题"""
        import re
        subqueries = []
        parts = re.split(r'[,;]', question)

        if len(parts) > 1:
            for i, part in enumerate(parts, 1):
                part = part.strip()
                if part and len(part) > 5:
                    if not part.endswith('?'):
                        part += '?'
                    subqueries.append({
                        'subquery': part,
                        'type': 'punctuation_split',
                        'order': i,
                        'dependency': None if i == 1 else i - 1
                    })

        return subqueries

    def _extract_core_concepts(self, question: str) -> List[Dict]:
        """提取核心概念生成子查询"""
        subqueries = []

        # 主查询
        subqueries.append({
            'subquery': question,
            'type': 'main_query',
            'order': 1,
            'dependency': None
        })

        # 如果问题包含"how"或"why",添加机制查询
        if any(word in question.lower() for word in ['how', 'why', 'mechanism']):
            mechanism_query = question.replace('?', ' mechanism?')
            subqueries.append({
                'subquery': mechanism_query,
                'type': 'mechanism_query',
                'order': 2,
                'dependency': 1
            })

        return subqueries