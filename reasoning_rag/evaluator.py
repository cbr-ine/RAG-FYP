"""评估指标计算模块"""
import logging
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self):
        self.metrics = defaultdict(list)

    def evaluate_batch(self, results: List[Dict], ground_truth: List[Dict],
                       system_name: str = "Unknown System") -> Dict:
        """批量评估 RAG 系统性能。"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {len(results)} queries for {system_name}...")
        logger.info(f"{'='*80}\n")

        if not results:
            return {
                'system_name': system_name,
                'total_queries': 0
            }

        metrics = {
            'system_name': system_name,
            'retrieval_success_rate': 0.0,
            'avg_evidence_count': 0.0,
            'avg_similarity_score': 0.0,
            'avg_confidence': 0.0,
            'avg_hops': 0.0,
            'complexity_distribution': {'simple': 0, 'complex': 0},
            'decomposition_rate': 0.0,
            'avg_subqueries': 0.0,
            'answer_coverage': 0.0,
            'exact_match': 0.0,
            'token_f1': 0.0,
            'abstention_rate': 0.0,
            'avg_latency_seconds': 0.0,
            'avg_total_retrievals': 0.0,
            'avg_llm_calls': 0.0,
            'llm_usage_rate': 0.0,
            'llm_provider_distribution': {},
            'query_analysis_visibility_rate': 0.0,
            'reasoning_path_availability_rate': 0.0,
            'evidence_traceability_rate': 0.0,
            'provenance_completeness_score': 0.0,
            'decomposition_visibility_rate': 0.0,
            'retrieval_path_visibility_rate': 0.0,
            'complex_query_transparency_score': 0.0,
            'v_score': 0.0,
            'avg_high_quality_evidence': 0.0,
            'integration_method_distribution': {},
            'query_type_breakdown': {},
            'total_queries': len(results)
        }

        successful_retrievals = 0
        total_evidence = 0
        total_similarity = 0.0
        total_confidence = 0.0
        total_hops = 0
        total_subqueries = 0
        decomposed_queries = 0
        abstained_queries = 0
        total_latency = 0.0
        total_retrieval_calls = 0
        total_llm_calls = 0
        llm_used_queries = 0
        total_high_quality_evidence = 0
        total_query_analysis_visibility = 0.0
        total_reasoning_path_availability = 0.0
        total_evidence_traceability = 0.0
        total_provenance_completeness = 0.0
        total_decomposition_visibility = 0.0
        total_retrieval_path_visibility = 0.0
        complex_query_transparency_scores = []
        answer_coverages = []
        exact_matches = []
        f1_scores = []
        integration_methods = Counter()
        llm_provider_counts = Counter()

        query_type_stats = defaultdict(lambda: {
            'count': 0,
            'exact_match': [],
            'token_f1': [],
            'answer_coverage': [],
            'v_score': [],
            'reasoning_path_availability_rate': [],
            'evidence_traceability_rate': []
        })
        complexity_stats = defaultdict(lambda: {
            'count': 0,
            'exact_match': [],
            'token_f1': [],
            'answer_coverage': [],
            'v_score': []
        })

        for result, gt in zip(results, ground_truth):
            evidence_count = len(result['evidence']['evidence'])
            if evidence_count > 0:
                successful_retrievals += 1
            total_evidence += evidence_count

            retrieval_stats = result['retrieval']['retrieval_stats']
            total_similarity += retrieval_stats.get('avg_similarity', 0.0)
            total_retrieval_calls += retrieval_stats.get('total_retrievals', 0)
            total_high_quality_evidence += retrieval_stats.get('high_quality_count', 0)

            total_confidence += result['answer'].get('confidence', 0.0)
            query_llm_calls = result.get('metadata', {}).get('llm_calls', 0)
            total_llm_calls += query_llm_calls
            if query_llm_calls > 0:
                llm_used_queries += 1

            llm_provider = result.get('metadata', {}).get('llm_provider')
            if llm_provider:
                llm_provider_counts[llm_provider] += 1
            total_latency += result.get('metadata', {}).get('runtime_seconds', 0.0)

            total_hops += sum(
                sq_result['stats'].get('hops_performed', 0)
                for sq_result in result['retrieval']['subquery_results']
            )

            if result['analysis']['is_complex']:
                metrics['complexity_distribution']['complex'] += 1
            else:
                metrics['complexity_distribution']['simple'] += 1

            if len(result.get('subqueries', [])) > 1:
                decomposed_queries += 1

            total_subqueries += len(result.get('subqueries', []))

            query_analysis_visibility = self._calculate_query_analysis_visibility(result)
            reasoning_path_availability = self._calculate_reasoning_path_availability(result)
            evidence_traceability = self._calculate_evidence_traceability(result)
            provenance_completeness = self._calculate_provenance_completeness(result)
            decomposition_visibility = self._calculate_decomposition_visibility(result)
            retrieval_path_visibility = self._calculate_retrieval_path_visibility(result)

            total_query_analysis_visibility += query_analysis_visibility
            total_reasoning_path_availability += reasoning_path_availability
            total_evidence_traceability += evidence_traceability
            total_provenance_completeness += provenance_completeness
            total_decomposition_visibility += decomposition_visibility
            total_retrieval_path_visibility += retrieval_path_visibility
            transparency_score = float(np.mean([
                query_analysis_visibility,
                reasoning_path_availability,
                evidence_traceability,
                provenance_completeness,
                decomposition_visibility,
                retrieval_path_visibility,
            ]))

            if result.get('analysis', {}).get('is_complex', False):
                complex_query_transparency_scores.append(
                    np.mean([
                        query_analysis_visibility,
                        reasoning_path_availability,
                        decomposition_visibility,
                        retrieval_path_visibility,
                        evidence_traceability
                    ])
                )

            coverage = self._calculate_answer_coverage(result, gt)
            em = self._calculate_exact_match(result, gt)
            f1 = self._calculate_token_f1(result, gt)

            answer_coverages.append(coverage)
            exact_matches.append(em)
            f1_scores.append(f1)

            if result['answer'].get('abstained', False):
                abstained_queries += 1

            query_type = result.get('analysis', {}).get('query_type', 'unknown') or 'unknown'
            query_type_stats[query_type]['count'] += 1
            query_type_stats[query_type]['exact_match'].append(em)
            query_type_stats[query_type]['token_f1'].append(f1)
            query_type_stats[query_type]['answer_coverage'].append(coverage)
            query_type_stats[query_type]['v_score'].append(transparency_score)
            query_type_stats[query_type]['reasoning_path_availability_rate'].append(reasoning_path_availability)
            query_type_stats[query_type]['evidence_traceability_rate'].append(evidence_traceability)

            complexity_bucket = 'complex' if result.get('analysis', {}).get('is_complex', False) else 'simple'
            complexity_stats[complexity_bucket]['count'] += 1
            complexity_stats[complexity_bucket]['exact_match'].append(em)
            complexity_stats[complexity_bucket]['token_f1'].append(f1)
            complexity_stats[complexity_bucket]['answer_coverage'].append(coverage)
            complexity_stats[complexity_bucket]['v_score'].append(transparency_score)

            integration_method = (
                result.get('metadata', {}).get('integration_method')
                or result.get('evidence', {}).get('integration_method')
                or 'unknown'
            )
            integration_methods[integration_method] += 1

        metrics['retrieval_success_rate'] = successful_retrievals / len(results)
        metrics['avg_evidence_count'] = total_evidence / len(results)
        metrics['avg_similarity_score'] = total_similarity / len(results)
        metrics['avg_confidence'] = total_confidence / len(results)
        metrics['avg_hops'] = total_hops / len(results)
        metrics['decomposition_rate'] = decomposed_queries / len(results)
        metrics['avg_subqueries'] = total_subqueries / len(results)
        metrics['answer_coverage'] = np.mean(answer_coverages) if answer_coverages else 0.0
        metrics['exact_match'] = np.mean(exact_matches) if exact_matches else 0.0
        metrics['token_f1'] = np.mean(f1_scores) if f1_scores else 0.0
        metrics['abstention_rate'] = abstained_queries / len(results)
        metrics['avg_latency_seconds'] = total_latency / len(results)
        metrics['avg_total_retrievals'] = total_retrieval_calls / len(results)
        metrics['avg_llm_calls'] = total_llm_calls / len(results)
        metrics['llm_usage_rate'] = llm_used_queries / len(results)
        metrics['llm_provider_distribution'] = dict(llm_provider_counts)
        metrics['query_analysis_visibility_rate'] = total_query_analysis_visibility / len(results)
        metrics['reasoning_path_availability_rate'] = total_reasoning_path_availability / len(results)
        metrics['evidence_traceability_rate'] = total_evidence_traceability / len(results)
        metrics['provenance_completeness_score'] = total_provenance_completeness / len(results)
        metrics['decomposition_visibility_rate'] = total_decomposition_visibility / len(results)
        metrics['retrieval_path_visibility_rate'] = total_retrieval_path_visibility / len(results)
        metrics['complex_query_transparency_score'] = (
            float(np.mean(complex_query_transparency_scores))
            if complex_query_transparency_scores else 0.0
        )
        metrics['v_score'] = float(np.mean([
            metrics['query_analysis_visibility_rate'],
            metrics['reasoning_path_availability_rate'],
            metrics['evidence_traceability_rate'],
            metrics['provenance_completeness_score'],
            metrics['decomposition_visibility_rate'],
            metrics['retrieval_path_visibility_rate'],
        ]))
        metrics['avg_high_quality_evidence'] = total_high_quality_evidence / len(results)
        metrics['integration_method_distribution'] = dict(integration_methods)
        metrics['query_type_breakdown'] = {
            query_type: {
                'count': values['count'],
                'exact_match': float(np.mean(values['exact_match'])) if values['exact_match'] else 0.0,
                'token_f1': float(np.mean(values['token_f1'])) if values['token_f1'] else 0.0,
                'answer_coverage': float(np.mean(values['answer_coverage'])) if values['answer_coverage'] else 0.0,
                'v_score': float(np.mean(values['v_score'])) if values['v_score'] else 0.0,
                'reasoning_path_availability_rate': float(np.mean(values['reasoning_path_availability_rate'])) if values['reasoning_path_availability_rate'] else 0.0,
                'evidence_traceability_rate': float(np.mean(values['evidence_traceability_rate'])) if values['evidence_traceability_rate'] else 0.0
            }
            for query_type, values in sorted(query_type_stats.items())
        }
        metrics['complexity_stratification'] = {
            bucket: {
                'count': values['count'],
                'exact_match': float(np.mean(values['exact_match'])) if values['exact_match'] else 0.0,
                'token_f1': float(np.mean(values['token_f1'])) if values['token_f1'] else 0.0,
                'answer_coverage': float(np.mean(values['answer_coverage'])) if values['answer_coverage'] else 0.0,
                'v_score': float(np.mean(values['v_score'])) if values['v_score'] else 0.0
            }
            for bucket, values in sorted(complexity_stats.items())
        }
        metrics['main_metrics'] = {
            'exact_match': metrics['exact_match'],
            'token_f1': metrics['token_f1'],
            'answer_coverage': metrics['answer_coverage'],
            'retrieval_success_rate': metrics['retrieval_success_rate'],
            'avg_similarity_score': metrics['avg_similarity_score']
        }
        metrics['transparency_metrics'] = {
            'v_score': metrics['v_score'],
            'query_analysis_visibility_rate': metrics['query_analysis_visibility_rate'],
            'reasoning_path_availability_rate': metrics['reasoning_path_availability_rate'],
            'evidence_traceability_rate': metrics['evidence_traceability_rate'],
            'provenance_completeness_score': metrics['provenance_completeness_score'],
            'decomposition_visibility_rate': metrics['decomposition_visibility_rate'],
            'retrieval_path_visibility_rate': metrics['retrieval_path_visibility_rate'],
            'complex_query_transparency_score': metrics['complex_query_transparency_score']
        }
        metrics['efficiency_metrics'] = {
            'avg_latency_seconds': metrics['avg_latency_seconds'],
            'avg_total_retrievals': metrics['avg_total_retrievals'],
            'avg_hops': metrics['avg_hops'],
            'avg_llm_calls': metrics['avg_llm_calls']
        }

        self._print_metrics(metrics)
        return metrics

    def _normalize_text(self, text: str) -> str:
        """归一化文本，便于 EM/F1 计算。"""
        if not text:
            return ""

        def remove_articles(value: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', value)

        def white_space_fix(value: str) -> str:
            return ' '.join(value.split())

        def remove_punc(value: str) -> str:
            exclude = set(string.punctuation)
            return ''.join(ch for ch in value if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(text.lower())))

    def _best_score_over_ground_truths(self, prediction: str, gt_answers: List[str], scorer) -> float:
        if not gt_answers:
            return 0.0
        return max(scorer(prediction, str(answer)) for answer in gt_answers)

    def _calculate_answer_coverage(self, result: Dict, ground_truth: Dict) -> float:
        """计算答案覆盖率。"""
        prediction = self._normalize_text(result['answer'].get('answer', ''))
        gt_answers = ground_truth.get('answers', [])
        if not prediction or not gt_answers:
            return 0.0

        def coverage_score(pred_text: str, gt_text: str) -> float:
            pred_words = set(pred_text.split())
            gt_words = set(self._normalize_text(gt_text).split())
            if not gt_words:
                return 0.0
            return min(len(pred_words.intersection(gt_words)) / len(gt_words), 1.0)

        return self._best_score_over_ground_truths(prediction, gt_answers, coverage_score)

    def _calculate_exact_match(self, result: Dict, ground_truth: Dict) -> float:
        prediction = self._normalize_text(result['answer'].get('answer', ''))
        gt_answers = ground_truth.get('answers', [])
        if not prediction or not gt_answers:
            return 0.0

        return self._best_score_over_ground_truths(
            prediction,
            gt_answers,
            lambda pred, gt: 1.0 if pred == self._normalize_text(gt) else 0.0
        )

    def _calculate_token_f1(self, result: Dict, ground_truth: Dict) -> float:
        prediction = self._normalize_text(result['answer'].get('answer', ''))
        gt_answers = ground_truth.get('answers', [])
        if not prediction or not gt_answers:
            return 0.0

        def f1_score(pred_text: str, gt_text: str) -> float:
            pred_tokens = pred_text.split()
            gt_tokens = self._normalize_text(gt_text).split()
            if not pred_tokens or not gt_tokens:
                return 0.0

            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0

            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            return (2 * precision * recall) / (precision + recall)

        return self._best_score_over_ground_truths(prediction, gt_answers, f1_score)

    def _calculate_query_analysis_visibility(self, result: Dict) -> float:
        analysis = result.get('analysis', {})
        required_fields = [
            'complexity_score',
            'is_complex',
            'requires_decomposition',
            'reasoning_depth',
            'query_type'
        ]
        return sum(1 for field in required_fields if field in analysis) / len(required_fields)

    def _calculate_reasoning_path_availability(self, result: Dict) -> float:
        path = result.get('answer', {}).get('reasoning_path', [])
        if not path:
            return 0.0
        visible_steps = 0
        for step in path:
            if step.get('type') and step.get('description'):
                visible_steps += 1
        return visible_steps / len(path)

    def _calculate_evidence_traceability(self, result: Dict) -> float:
        sources = result.get('answer', {}).get('sources', [])
        if not sources:
            return 0.0

        traceable_sources = 0
        for source in sources:
            if (
                source.get('text')
                and source.get('source_type') not in (None, '', 'unknown')
                and source.get('hop') is not None
            ):
                traceable_sources += 1
        return traceable_sources / len(sources)

    def _calculate_provenance_completeness(self, result: Dict) -> float:
        sources = result.get('answer', {}).get('sources', [])
        if not sources:
            return 0.0

        required_fields = ['text', 'similarity', 'quality_score', 'hop', 'source_type']
        source_scores = []
        for source in sources:
            present_fields = 0
            for field in required_fields:
                value = source.get(field)
                if value not in (None, '', 'unknown'):
                    present_fields += 1
            source_scores.append(present_fields / len(required_fields))
        return float(np.mean(source_scores))

    def _calculate_decomposition_visibility(self, result: Dict) -> float:
        analysis = result.get('analysis', {})
        subqueries = result.get('subqueries', [])
        if analysis.get('requires_decomposition', False):
            meaningful_subqueries = [sq for sq in subqueries if sq.get('subquery')]
            return min(1.0, len(meaningful_subqueries) / 2) if meaningful_subqueries else 0.0
        return 1.0 if subqueries else 0.0

    def _calculate_retrieval_path_visibility(self, result: Dict) -> float:
        subquery_results = result.get('retrieval', {}).get('subquery_results', [])
        if not subquery_results:
            return 0.0

        visible_paths = 0
        for subquery_result in subquery_results:
            hops = subquery_result.get('hops', [])
            if hops and all('hop_number' in hop and 'query' in hop for hop in hops):
                visible_paths += 1
        return visible_paths / len(subquery_results)

    @staticmethod
    def _aggregate_scalar(values: List[float]) -> Dict:
        if not values:
            return {'mean': 0.0, 'std': 0.0}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }

    @classmethod
    def aggregate_metrics(cls, metrics_runs: List[Dict], system_name: str = "") -> Dict:
        """Aggregate metrics across multiple runs/seeds."""
        if not metrics_runs:
            return {'system_name': system_name, 'num_runs': 0}

        scalar_keys = [
            'exact_match', 'token_f1', 'answer_coverage', 'retrieval_success_rate',
            'avg_similarity_score', 'avg_latency_seconds', 'avg_total_retrievals',
            'avg_hops', 'avg_llm_calls', 'v_score',
            'query_analysis_visibility_rate', 'reasoning_path_availability_rate',
            'evidence_traceability_rate', 'provenance_completeness_score',
            'decomposition_visibility_rate', 'retrieval_path_visibility_rate',
            'complex_query_transparency_score', 'abstention_rate'
        ]

        aggregate = {
            'system_name': system_name or metrics_runs[0].get('system_name', ''),
            'num_runs': len(metrics_runs),
            'main_metrics': {},
            'transparency_metrics': {},
            'efficiency_metrics': {},
            'query_type_breakdown': {},
            'complexity_stratification': {}
        }

        scalar_summary = {
            key: cls._aggregate_scalar([run.get(key, 0.0) for run in metrics_runs])
            for key in scalar_keys
        }
        aggregate.update(scalar_summary)
        aggregate['main_metrics'] = {
            key: scalar_summary[key]
            for key in ['exact_match', 'token_f1', 'answer_coverage', 'retrieval_success_rate', 'avg_similarity_score']
        }
        aggregate['transparency_metrics'] = {
            key: scalar_summary[key]
            for key in [
                'v_score', 'query_analysis_visibility_rate',
                'reasoning_path_availability_rate', 'evidence_traceability_rate',
                'provenance_completeness_score', 'decomposition_visibility_rate',
                'retrieval_path_visibility_rate', 'complex_query_transparency_score'
            ]
        }
        aggregate['efficiency_metrics'] = {
            key: scalar_summary[key]
            for key in ['avg_latency_seconds', 'avg_total_retrievals', 'avg_hops', 'avg_llm_calls', 'abstention_rate']
        }

        query_types = sorted({
            query_type
            for run in metrics_runs
            for query_type in run.get('query_type_breakdown', {}).keys()
        })
        for query_type in query_types:
            aggregate['query_type_breakdown'][query_type] = {
                'token_f1': cls._aggregate_scalar([
                    run.get('query_type_breakdown', {}).get(query_type, {}).get('token_f1', 0.0)
                    for run in metrics_runs
                ]),
                'answer_coverage': cls._aggregate_scalar([
                    run.get('query_type_breakdown', {}).get(query_type, {}).get('answer_coverage', 0.0)
                    for run in metrics_runs
                ]),
                'v_score': cls._aggregate_scalar([
                    run.get('query_type_breakdown', {}).get(query_type, {}).get('v_score', 0.0)
                    for run in metrics_runs
                ]),
                'count': int(round(np.mean([
                    run.get('query_type_breakdown', {}).get(query_type, {}).get('count', 0)
                    for run in metrics_runs
                ])))
            }

        for bucket in ['simple', 'complex']:
            aggregate['complexity_stratification'][bucket] = {
                'token_f1': cls._aggregate_scalar([
                    run.get('complexity_stratification', {}).get(bucket, {}).get('token_f1', 0.0)
                    for run in metrics_runs
                ]),
                'answer_coverage': cls._aggregate_scalar([
                    run.get('complexity_stratification', {}).get(bucket, {}).get('answer_coverage', 0.0)
                    for run in metrics_runs
                ]),
                'v_score': cls._aggregate_scalar([
                    run.get('complexity_stratification', {}).get(bucket, {}).get('v_score', 0.0)
                    for run in metrics_runs
                ]),
                'count': int(round(np.mean([
                    run.get('complexity_stratification', {}).get(bucket, {}).get('count', 0)
                    for run in metrics_runs
                ])))
            }

        return aggregate

    def _print_metrics(self, metrics: Dict):
        """打印评估指标。"""
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION METRICS - {metrics['system_name']}")
        logger.info(f"{'='*80}\n")

        logger.info("Retrieval Performance:")
        logger.info(f"  Success Rate:             {metrics['retrieval_success_rate']:.2%}")
        logger.info(f"  Avg Evidence Count:       {metrics['avg_evidence_count']:.2f}")
        logger.info(f"  Avg Similarity Score:     {metrics['avg_similarity_score']:.3f}")
        logger.info(f"  Avg High-Quality Evidence:{metrics['avg_high_quality_evidence']:.2f}")

        logger.info("\nQuery Analysis:")
        logger.info(
            f"  Simple Queries:           {metrics['complexity_distribution']['simple']} "
            f"({metrics['complexity_distribution']['simple']/metrics['total_queries']:.1%})"
        )
        logger.info(
            f"  Complex Queries:          {metrics['complexity_distribution']['complex']} "
            f"({metrics['complexity_distribution']['complex']/metrics['total_queries']:.1%})"
        )
        logger.info(f"  Decomposition Rate:       {metrics['decomposition_rate']:.2%}")
        logger.info(f"  Avg Subqueries:           {metrics['avg_subqueries']:.2f}")
        logger.info(f"  Avg Hops per Query:       {metrics['avg_hops']:.2f}")

        logger.info("\nAnswer Quality:")
        logger.info(f"  Exact Match:              {metrics['exact_match']:.2%}")
        logger.info(f"  Token F1:                 {metrics['token_f1']:.2%}")
        logger.info(f"  Answer Coverage:          {metrics['answer_coverage']:.2%}")
        logger.info(f"  Avg Confidence:           {metrics['avg_confidence']:.3f}")
        logger.info(f"  Abstention Rate:          {metrics['abstention_rate']:.2%}")

        logger.info("\nEfficiency:")
        logger.info(f"  Avg Latency (s):          {metrics['avg_latency_seconds']:.3f}")
        logger.info(f"  Avg Retrieval Calls:      {metrics['avg_total_retrievals']:.2f}")
        logger.info(f"  Avg LLM Calls:            {metrics['avg_llm_calls']:.2f}")
        logger.info(f"  LLM Usage Rate:           {metrics['llm_usage_rate']:.2%}")
        if metrics['llm_provider_distribution']:
            logger.info(f"  LLM Providers:            {metrics['llm_provider_distribution']}")
        if metrics['avg_llm_calls'] == 0:
            logger.warning(
                "No LLM calls were recorded for this evaluation. "
                "Check OPENAI_API_KEY and model configuration."
            )

        logger.info("\nTransparency / Traceability:")
        logger.info(f"  V-Score:                  {metrics['v_score']:.2%}")
        logger.info(f"  Analysis Visibility:      {metrics['query_analysis_visibility_rate']:.2%}")
        logger.info(f"  Reasoning Path Visible:   {metrics['reasoning_path_availability_rate']:.2%}")
        logger.info(f"  Evidence Traceability:    {metrics['evidence_traceability_rate']:.2%}")
        logger.info(f"  Provenance Completeness:  {metrics['provenance_completeness_score']:.2%}")
        logger.info(f"  Decomposition Visibility: {metrics['decomposition_visibility_rate']:.2%}")
        logger.info(f"  Retrieval Path Visibility:{metrics['retrieval_path_visibility_rate']:.2%}")
        logger.info(f"  Complex Query Transparency:{metrics['complex_query_transparency_score']:.2%}")

        logger.info("\nIntegration Methods:")
        for method, count in metrics['integration_method_distribution'].items():
            logger.info(f"  {method}: {count}")

        logger.info("\nQuery Type Breakdown:")
        for query_type, values in metrics['query_type_breakdown'].items():
            logger.info(
                f"  {query_type}: count={values['count']}, "
                f"EM={values['exact_match']:.2%}, "
                f"F1={values['token_f1']:.2%}, "
                f"Coverage={values['answer_coverage']:.2%}"
            )

        logger.info(f"\nOverall:")
        logger.info(f"  Total Queries:            {metrics['total_queries']}")
        logger.info(f"\n{'='*80}\n")