"""答案生成模块 - 使用 DeepSeek LLM"""
from typing import Dict, List
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

# 低于此相似度则认为证据与问题无关，拒绝生成答案
MIN_ANSWER_SIMILARITY = 0.55

class AnswerGenerator:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY not found in environment. Using simple synthesis.")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

    def generate(self, question: str, integrated_evidence: Dict,
                 analysis_result: Dict, retrieval_results: Dict) -> Dict:
        """基于证据生成答案"""
        logger.info("Generating answer...")

        evidence_list = integrated_evidence['evidence']

        # --- 无证据处理 ---
        if not evidence_list:
            return self._no_evidence_result(
                question, analysis_result, retrieval_results
            )

        # --- 相关性检测：最高相似度低于阈值时拒绝生成 ---
        max_similarity = max(e['similarity'] for e in evidence_list)
        if max_similarity < MIN_ANSWER_SIMILARITY:
            logger.warning(
                f"All evidence below similarity threshold "
                f"(max={max_similarity:.3f} < {MIN_ANSWER_SIMILARITY}). "
                f"Relevant passage may not be in the index."
            )
            reasoning_path = self._build_reasoning_path(
                question, analysis_result,
                retrieval_results['subquery_results'], evidence_list
            )
            return {
                'answer': (
                    f"The retrieved evidence is not sufficiently relevant to answer this question "
                    f"(best similarity: {max_similarity:.3f}, required: {MIN_ANSWER_SIMILARITY}).\n\n"
                    f"Suggestion: rebuild the index with more passages using:\n"
                    f"  python main.py --mode build --full-index --rebuild-index"
                ),
                'confidence': 0.0,
                'reasoning_path': reasoning_path,
                'sources': self._extract_sources(evidence_list),
                'evidence_count': len(evidence_list)
            }

        # --- 正常流程 ---
        reasoning_path = self._build_reasoning_path(
            question, analysis_result,
            retrieval_results['subquery_results'], evidence_list
        )

        if self.client:
            try:
                answer = self._synthesize_answer_with_llm(
                    question, evidence_list, analysis_result
                )
            except Exception as e:
                logger.error(f"LLM answer generation failed: {e}. Using simple synthesis.")
                answer = self._synthesize_answer_simple(question, evidence_list)
        else:
            answer = self._synthesize_answer_simple(question, evidence_list)

        sources = self._extract_sources(evidence_list)
        confidence = integrated_evidence['validation']['avg_confidence']

        result = {
            'answer': answer,
            'confidence': confidence,
            'reasoning_path': reasoning_path,
            'sources': sources,
            'evidence_count': len(evidence_list)
        }

        logger.info(f"Answer generated with confidence: {confidence:.3f}")
        return result

    def _no_evidence_result(self, question: str, analysis_result: Dict,
                             retrieval_results: Dict) -> Dict:
        """无证据时的返回结构"""
        return {
            'answer': "No sufficient evidence found to answer this question.",
            'confidence': 0.0,
            'reasoning_path': self._build_reasoning_path(
                question, analysis_result,
                retrieval_results.get('subquery_results', []), []
            ),
            'sources': [],
            'evidence_count': 0
        }

    def _synthesize_answer_with_llm(self, question: str, evidence_list: List[Dict],
                                     analysis_result: Dict) -> str:
        """使用 DeepSeek LLM 合成答案"""
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
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise and factual answer synthesis assistant."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"LLM generated answer: {answer[:100]}...")
        return answer

    def _synthesize_answer_simple(self, question: str, evidence_list: List[Dict]) -> str:
        """简单的基于证据的答案生成（回退方案）"""
        if not evidence_list:
            return "No answer available."

        top_evidence = evidence_list[0]['text']
        sentences = top_evidence.split('.')
        answer_sentences = sentences[:2]

        answer = '. '.join(s.strip() for s in answer_sentences if s.strip())
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