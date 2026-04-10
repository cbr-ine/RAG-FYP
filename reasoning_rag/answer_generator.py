"""答案生成模块 - 使用 OpenAI LLM."""
from typing import Dict, List, Optional
import logging
from llm_provider import get_llm_client, get_model_name, get_token_limit_kwargs

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, min_answer_similarity: float = 0.50):
        self.client, self.provider = get_llm_client()
        self.model_name = get_model_name("generation", self.provider)
        self.min_answer_similarity = min_answer_similarity
        self.last_llm_attempted = False
        self.last_llm_succeeded = False
        self.last_llm_error = None
        if not self.client or not self.model_name:
            logger.warning(
                "OPENAI_API_KEY not found in environment. "
                "Using simple synthesis."
            )
        else:
            logger.info(
                f"Answer generation LLM enabled: provider={self.provider}, model={self.model_name}"
            )

    def generate(self, question: str, integrated_evidence: Dict,
                 analysis_result: Dict, retrieval_results: Dict,
                 use_abstention: bool = True,
                 reasoning_path: Optional[List[Dict]] = None) -> Dict:
        """基于证据生成答案"""
        self.last_llm_attempted = False
        self.last_llm_succeeded = False
        self.last_llm_error = None
        logger.info("Generating answer...")

        evidence_list = integrated_evidence['evidence']

        # --- 无证据处理 ---
        if not evidence_list:
            return self._no_evidence_result(
                question, analysis_result, retrieval_results, reasoning_path
            )

        # --- 相关性检测：最高相似度低于阈值时拒绝生成 ---
        max_similarity = max(e['similarity'] for e in evidence_list)
        if use_abstention and max_similarity < self.min_answer_similarity:
            logger.warning(
                f"All evidence below similarity threshold "
                f"(max={max_similarity:.3f} < {self.min_answer_similarity}). "
                f"Relevant passage may not be in the index."
            )
            reasoning_path = reasoning_path or self._build_reasoning_path(
                question, analysis_result,
                retrieval_results['subquery_results'], evidence_list
            )
            return {
                'answer': (
                    f"The retrieved evidence is not sufficiently relevant to answer this question "
                    f"(best similarity: {max_similarity:.3f}, required: {self.min_answer_similarity}).\n\n"
                    f"Suggestion: rebuild the index with more passages using:\n"
                    f"  python main.py --mode build --full-index --rebuild-index"
                ),
                'confidence': 0.0,
                'reasoning_path': reasoning_path,
                'sources': self._extract_sources(evidence_list),
                'evidence_count': len(evidence_list),
                'abstained': True,
                'generation_method': 'abstained',
                'llm_calls': 0,
                'llm_error': None
            }

        # --- 正常流程 ---
        reasoning_path = reasoning_path or self._build_reasoning_path(
            question, analysis_result,
            retrieval_results['subquery_results'], evidence_list
        )

        generation_method = 'simple'
        llm_calls = 0
        if self.client:
            try:
                self.last_llm_attempted = True
                answer = self._synthesize_answer_with_llm(
                    question, evidence_list, analysis_result
                )
                llm_calls = 1
                if self._is_effective_answer(answer):
                    generation_method = 'llm'
                    self.last_llm_succeeded = True
                else:
                    self.last_llm_error = "LLM returned an empty or low-information answer."
                    logger.warning(
                        "LLM returned an empty or low-information answer. "
                        "Falling back to simple evidence synthesis."
                    )
                    answer = self._synthesize_answer_simple(question, evidence_list)
                    generation_method = 'simple_fallback'
            except Exception as e:
                self.last_llm_error = str(e)
                logger.error(f"LLM answer generation failed: {e}. Using simple synthesis.")
                answer = self._synthesize_answer_simple(question, evidence_list)
                generation_method = 'simple_fallback'
        else:
            answer = self._synthesize_answer_simple(question, evidence_list)

        sources = self._extract_sources(evidence_list)
        confidence = integrated_evidence['validation']['avg_confidence']

        result = {
            'answer': answer,
            'confidence': confidence,
            'reasoning_path': reasoning_path,
            'sources': sources,
            'evidence_count': len(evidence_list),
            'abstained': False,
            'generation_method': generation_method,
            'llm_calls': llm_calls,
            'llm_error': self.last_llm_error
        }

        logger.info(f"Answer generated with confidence: {confidence:.3f}")
        return result

    def _no_evidence_result(self, question: str, analysis_result: Dict,
                             retrieval_results: Dict,
                             reasoning_path: Optional[List[Dict]] = None) -> Dict:
        """无证据时的返回结构"""
        return {
            'answer': "No sufficient evidence found to answer this question.",
            'confidence': 0.0,
            'reasoning_path': reasoning_path or self._build_reasoning_path(
                question, analysis_result,
                retrieval_results.get('subquery_results', []), []
            ),
            'sources': [],
            'evidence_count': 0,
            'abstained': True,
            'generation_method': 'no_evidence',
            'llm_calls': 0,
            'llm_error': None
        }

    def _synthesize_answer_with_llm(self, question: str, evidence_list: List[Dict],
                                     analysis_result: Dict) -> str:
        """使用 OpenAI LLM 合成答案"""
        evidence_texts = []
        for i, evidence in enumerate(evidence_list[:5], 1):
            evidence_texts.append(
                f"Evidence {i} (Quality: {evidence['quality_score']:.2f}, "
                f"Similarity: {evidence['similarity']:.2f}):\n{evidence['text']}\n"
            )

        evidence_context = "\n".join(evidence_texts)

        prompt = f"""You are an expert assistant that synthesizes information from multiple sources to answer questions accurately.

Question: {question}

Available Evidence:
{evidence_context}

Instructions:
1. Synthesize a comprehensive answer based on the provided evidence
2. Prioritize evidence with higher quality and similarity scores
3. Be concise but complete
4. If the evidence is insufficient or contradictory, acknowledge this
5. Use factual language and avoid speculation
6. Generate your answer after careful analysis of the evidence

Provide a clear, well-structured answer:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise and factual answer synthesis assistant."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            **get_token_limit_kwargs(self.model_name, 300)
        )

        answer = response.choices[0].message.content.strip()
        logger.info("LLM generated answer preview: %r", answer[:100])
        return answer

    def _is_effective_answer(self, answer: str) -> bool:
        """Reject empty or placeholder-only model outputs before showing them to users."""
        if not answer:
            return False

        normalized = " ".join(answer.split()).strip().lower()
        if not normalized:
            return False

        low_information_answers = {
            ".", "..", "...", "n/a", "na", "none", "unknown",
            "no answer", "no answer available"
        }
        if normalized in low_information_answers:
            return False

        content_chars = [ch for ch in normalized if ch.isalnum()]
        if len(content_chars) < 8:
            return False

        return True

    def _synthesize_answer_simple(self, question: str, evidence_list: List[Dict]) -> str:
        """简单的基于证据的答案生成（回退方案）"""
        if not evidence_list:
            return "No answer available."

        top_evidence = evidence_list[0]['text']
        sentences = top_evidence.split('.')
        answer_sentences = sentences[:2]

        answer = '. '.join(s.strip() for s in answer_sentences if s.strip())
        if not answer:
            answer = top_evidence.strip()
        if answer and not answer.endswith('.'):
            answer += '.'

        if len(evidence_list) > 1 and evidence_list[1]['quality_score'] > 0.6:
            additional = evidence_list[1]['text'].split('.')[0].strip()
            if additional and additional not in answer:
                answer += f" Additionally, {additional.lower()}."

        return answer

    def _build_reasoning_path(self, question: str, analysis: Dict,
                               subquery_results: List[Dict],
                               evidence: List[Dict]) -> List[Dict]:
        """构建推理路径"""
        path = []

        path.append({
            'step': 1,
            'type': 'analysis',
            'description': f"Analyzed question complexity: {analysis['complexity_score']:.2f}",
            'is_complex': analysis['is_complex']
        })

        if analysis['requires_decomposition']:
            path.append({
                'step': 2,
                'type': 'decomposition',
                'description': f"Decomposed into {len(subquery_results)} sub-queries",
                'subqueries': [sq['subquery'] for sq in subquery_results]
            })

        total_hops = sum(sq['stats']['hops_performed'] for sq in subquery_results)
        path.append({
            'step': 3,
            'type': 'retrieval',
            'description': f"Performed multi-hop retrieval ({total_hops} total hops)",
            'evidence_found': len(evidence)
        })

        path.append({
            'step': 4,
            'type': 'integration',
            'description': f"Integrated and validated {len(evidence)} evidence pieces",
            'top_evidence_scores': [f"{e['quality_score']:.3f}" for e in evidence[:3]]
        })

        path.append({
            'step': 5,
            'type': 'generation',
            'description': "Synthesized final answer from evidence"
        })

        return path

    def _extract_sources(self, evidence_list: List[Dict]) -> List[Dict]:
        """提取证据来源"""
        sources = []
        for i, evidence in enumerate(evidence_list, 1):
            text = evidence['text']
            sources.append({
                'rank': i,
                'text': text[:200] + '...' if len(text) > 200 else text,
                'similarity': evidence['similarity'],
                'quality_score': evidence['quality_score'],
                'hop': evidence['hop'],
                'source_type': evidence.get('source_type', 'unknown')
            })
        return sources