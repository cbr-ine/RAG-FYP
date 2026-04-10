"""主运行文件"""
import argparse
import copy
import json
import logging
import os
import random
from html import escape
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from config import Config
from data_loader import DataLoader
from evaluator import RAGEvaluator
from llm_provider import get_llm_status
from reasoning_rag import ReasoningRAG
from traditional_rag import TraditionalRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset_if_needed(data_loader: DataLoader):
    """确保 BioASQ 数据已经载入。"""
    if data_loader.train_data is None or data_loader.test_data is None:
        data_loader.load_bioasq_dataset(train_ratio=0.8)


def log_llm_status():
    """Log current LLM availability before running experiments."""
    decomposition_status = get_llm_status("decomposition")
    generation_status = get_llm_status("generation")
    logger.info(
        "LLM status | decomposition: enabled=%s provider=%s model=%s | generation: enabled=%s provider=%s model=%s",
        decomposition_status["enabled"],
        decomposition_status["provider"],
        decomposition_status["model_name"],
        generation_status["enabled"],
        generation_status["provider"],
        generation_status["model_name"],
    )


def build_index(rag_system: ReasoningRAG, data_loader: DataLoader,
                index_path: str, max_passages: int = 1000,
                use_full_dataset: bool = False):
    """构建并保存索引。"""
    logger.info("Building vector index from dataset...")
    load_dataset_if_needed(data_loader)

    if use_full_dataset:
        train_passages = data_loader.get_passages(split='train')
        test_passages = data_loader.get_passages(split='test')
        passages = train_passages + test_passages
        logger.info(
            f"Full dataset mode: {len(train_passages)} train + "
            f"{len(test_passages)} test = {len(passages)} total passages"
        )
    else:
        passages = data_loader.get_passages(split='train', max_passages=max_passages)
        logger.info(f"Train-only mode: {len(passages)} passages (max={max_passages})")

    if not passages:
        logger.error("No passages found! Cannot build index.")
        return

    rag_system.build_index(passages)
    rag_system.save_index(index_path)
    logger.info(f"Index saved to {index_path}")


def ensure_index(rag_system: ReasoningRAG, data_loader: DataLoader, args):
    """构建或加载索引。"""
    need_build = (
        args.mode == 'build'
        or args.rebuild_index
        or not os.path.exists(args.index_path)
    )

    if need_build:
        build_index(
            rag_system,
            data_loader,
            args.index_path,
            args.max_passages,
            use_full_dataset=args.full_index
        )
        return

    logger.info(f"Loading existing index from {args.index_path}")
    try:
        rag_system.load_index(args.index_path)
    except Exception as exc:
        logger.error(f"Failed to load index: {exc}")
        logger.info("Building new index...")
        build_index(
            rag_system,
            data_loader,
            args.index_path,
            args.max_passages,
            use_full_dataset=args.full_index
        )


def select_questions(data_loader: DataLoader, split: str, sample_size: int, seed: int):
    """使用固定随机种子抽取问题。"""
    load_dataset_if_needed(data_loader)
    questions = data_loader.get_questions(split=split)
    if len(questions) <= sample_size:
        return questions

    rng = random.Random(seed)
    indices = rng.sample(range(len(questions)), sample_size)
    return [questions[i] for i in indices]


def run_evaluation(rag_system, test_questions, system_name: str):
    """运行单个系统的评测。"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"# RUNNING EVALUATION FOR {system_name} (Sample size: {len(test_questions)})")
    logger.info(f"{'#'*80}\n")

    results = []
    for i, question_data in enumerate(test_questions, 1):
        logger.info(f"\n[{system_name}] Processing {i}/{len(test_questions)}...")
        result = rag_system.query(question_data['question'], verbose=False)
        results.append(result)

    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_batch(results, test_questions, system_name=system_name)
    return metrics, results


def run_demo(rag_system: ReasoningRAG, demo_questions):
    """运行演示对话。"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"# RUNNING DEMO CONVERSATIONS (Sample size: {len(demo_questions)})")
    logger.info(f"{'#'*80}\n")

    for i, question_data in enumerate(demo_questions, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"DEMO {i}/{len(demo_questions)}")
        logger.info(f"{'='*80}\n")

        question = question_data['question']
        ground_truth_answers = question_data['answers']

        print(f"\nQUESTION:\n   {question}\n")
        result = rag_system.query(question, verbose=True)

        print(f"\nANALYSIS:")
        print(f"   Complexity Score: {result['analysis']['complexity_score']:.2f}")
        print(f"   Is Complex: {'Yes' if result['analysis']['is_complex'] else 'No'}")
        print(
            f"   Requires Decomposition: "
            f"{'Yes' if result['analysis']['requires_decomposition'] else 'No'}"
        )

        if len(result['subqueries']) > 1:
            print(f"\nSUB-QUERIES ({len(result['subqueries'])}):")
            for j, sq in enumerate(result['subqueries'], 1):
                print(f"   {j}. [{sq['type']}] {sq['subquery']}")

        print(f"\nRETRIEVAL STATS:")
        print(f"   Total Retrievals: {result['retrieval']['retrieval_stats']['total_retrievals']}")
        print(
            f"   Successful: "
            f"{result['retrieval']['retrieval_stats']['successful_retrievals']}"
            f"/{len(result['subqueries'])}"
        )
        print(f"   Avg Similarity: {result['retrieval']['retrieval_stats']['avg_similarity']:.3f}")
        print(f"   Total Evidence: {len(result['evidence']['evidence'])}")

        print(f"\nREASONING PATH:")
        for step in result['answer']['reasoning_path']:
            print(f"   Step {step['step']} [{step['type']}]: {step['description']}")

        print(f"\nGENERATED ANSWER:")
        print(f"   Confidence: {result['answer']['confidence']:.3f}")
        print(f"   {result['answer']['answer']}\n")

        if result['answer']['sources']:
            num_show = min(3, len(result['answer']['sources']))
            print(f"\nEVIDENCE SOURCES (Top {num_show}):")
            for source in result['answer']['sources'][:num_show]:
                print(
                    f"\n   [{source['rank']}] Quality: {source['quality_score']:.3f} "
                    f"| Similarity: {source['similarity']:.3f} "
                    f"| Hop: {source['hop']} "
                    f"| Type: {source.get('source_type', 'N/A')}"
                )
                print(f"       {source['text']}")

        print(f"\nGROUND TRUTH ANSWERS (for reference):")
        for j, ans in enumerate(ground_truth_answers[:2], 1):
            print(f"   {j}. {ans}")

        print(f"\n{'='*80}\n")
        if i < len(demo_questions):
            input("Press Enter to continue to next demo...\n")


def run_interactive(rag_system: ReasoningRAG):
    """运行交互式问答模式。"""
    logger.info(f"\n{'#' * 80}")
    logger.info("# INTERACTIVE QUESTION ANSWERING MODE")
    logger.info("# Type 'quit' or 'exit' to stop")
    logger.info(f"{'#' * 80}\n")

    print("\nWelcome to Reasoning RAG Interactive Mode!")
    print("You can ask any question and see the complete reasoning process.\n")

    question_count = 0
    while True:
        try:
            print(f"\n{'=' * 80}")
            user_input = input("\nYour Question: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Thanks for using Reasoning RAG!\n")
                break

            if not user_input:
                print("Please enter a question.")
                continue

            question_count += 1
            result = rag_system.query(user_input, verbose=True)

            print(f"\n{'=' * 80}")
            print("ANALYSIS RESULTS")
            print(f"{'=' * 80}\n")
            print(f"Complexity Score: {result['analysis']['complexity_score']:.2f}")
            print(f"Classification: {'Complex' if result['analysis']['is_complex'] else 'Simple'}")
            print(f"Needs Decomposition: {'Yes' if result['analysis']['requires_decomposition'] else 'No'}")
            print(f"Reasoning Depth: {result['analysis']['reasoning_depth']}")

            if len(result['subqueries']) > 1:
                print(f"\nQuery Decomposition ({len(result['subqueries'])} sub-queries):")
                for i, sq in enumerate(result['subqueries'], 1):
                    print(f"   {i}. [{sq['type'].upper()}] {sq['subquery']}")
            else:
                print("\nQuery Type: Single query (no decomposition needed)")

            stats = result['retrieval']['retrieval_stats']
            print(f"\nRetrieval Statistics:")
            print(f"   Total Retrievals: {stats['total_retrievals']}")
            print(f"   Successful: {stats['successful_retrievals']}/{len(result['subqueries'])}")
            print(f"   Average Similarity: {stats['avg_similarity']:.3f}")
            print(f"   Total Evidence Collected: {len(result['evidence']['evidence'])}")
            print(f"   High Quality Evidence: {stats.get('high_quality_count', 0)}")

            print(f"\nReasoning Path:")
            for step in result['answer']['reasoning_path']:
                print(f"   Step {step['step']} - {step['type'].upper()}: {step['description']}")

            print(f"\nGenerated Answer")
            print(f"Confidence Score: {result['answer']['confidence']:.3f}")
            print(f"\n{result['answer']['answer']}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!\n")
            break
        except Exception as exc:
            logger.error(f"Error processing question: {exc}", exc_info=True)
            print(f"\nError: {exc}")
            print("Please try another question.\n")

    if question_count > 0:
        print(f"\n{'=' * 80}")
        print(f"Session Statistics:")
        print(f"   Total Questions Processed: {question_count}")
        print(f"{'=' * 80}\n")


SYSTEM_PROFILES = {
    'core': [
        'traditional_dense',
        'traditional_dense_abstention',
        'transparent_single_hop',
        'reasoning_full',
        'reasoning_wo_decomposition',
        'reasoning_wo_multihop',
        'reasoning_wo_integration',
        'reasoning_wo_abstention',
    ],
    'extended': [
        'traditional_dense',
        'traditional_dense_abstention',
        'hybrid_retrieval_baseline',
        'transparent_single_hop',
        'reasoning_full',
        'reasoning_wo_decomposition',
        'reasoning_wo_multihop',
        'reasoning_wo_integration',
        'reasoning_wo_abstention',
    ],
    'maximal': [
        'traditional_dense',
        'traditional_dense_abstention',
        'hybrid_retrieval_baseline',
        'transparent_single_hop',
        'reasoning_full',
        'reasoning_wo_decomposition',
        'reasoning_wo_multihop',
        'reasoning_wo_integration',
        'reasoning_wo_abstention',
    ],
}


def create_config_instance(base_config: Config, embedding_model: str = None) -> Config:
    """Clone a config object and optionally override the embedding model."""
    config = copy.deepcopy(base_config)
    if embedding_model:
        config.EMBEDDING_MODEL = embedding_model
    return config


def sanitize_slug(text: str) -> str:
    slug = ''.join(ch.lower() if ch.isalnum() else '_' for ch in text)
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug.strip('_')


def get_profile_system_ids(compare_profile: str, skip_ablations: bool) -> list:
    profile_ids = list(SYSTEM_PROFILES.get(compare_profile, SYSTEM_PROFILES['core']))
    if skip_ablations:
        profile_ids = [
            system_id for system_id in profile_ids
            if not system_id.startswith('reasoning_wo_')
        ]
    return profile_ids


def instantiate_system(system_id: str, config: Config, reference_system: ReasoningRAG):
    shared_kwargs = {
        'config': config,
        'embedder': reference_system.embedder,
        'vector_store': reference_system.vector_store,
    }

    if system_id == 'traditional_dense':
        return "Traditional RAG", TraditionalRAG(
            **shared_kwargs,
            system_name="Traditional RAG",
            use_abstention=False,
            keyword_fallback_top_k=0
        )
    if system_id == 'traditional_dense_abstention':
        return "Traditional RAG + Abstention", TraditionalRAG(
            **shared_kwargs,
            system_name="Traditional RAG + Abstention",
            use_abstention=True,
            keyword_fallback_top_k=0
        )
    if system_id == 'hybrid_retrieval_baseline':
        return "Hybrid Retrieval Baseline", TraditionalRAG(
            **shared_kwargs,
            system_name="Hybrid Retrieval Baseline",
            use_abstention=False,
            keyword_fallback_top_k=config.KEYWORD_FALLBACK_TOP_K
        )
    if system_id == 'transparent_single_hop':
        return "Transparent Single-Hop RAG", ReasoningRAG(
            **shared_kwargs,
            system_name="Transparent Single-Hop RAG",
            use_decomposition=False,
            use_multi_hop=False,
            use_evidence_integration=False,
            use_abstention=True,
        )
    if system_id == 'reasoning_full':
        return "Reasoning RAG", reference_system
    if system_id == 'reasoning_wo_decomposition':
        return "Reasoning RAG w/o Decomposition", ReasoningRAG(
            **shared_kwargs,
            system_name="Reasoning RAG w/o Decomposition",
            use_decomposition=False,
            use_multi_hop=True,
            use_evidence_integration=True,
            use_abstention=True
        )
    if system_id == 'reasoning_wo_multihop':
        return "Reasoning RAG w/o Multi-Hop", ReasoningRAG(
            **shared_kwargs,
            system_name="Reasoning RAG w/o Multi-Hop",
            use_decomposition=True,
            use_multi_hop=False,
            use_evidence_integration=True,
            use_abstention=True
        )
    if system_id == 'reasoning_wo_integration':
        return "Reasoning RAG w/o Integration", ReasoningRAG(
            **shared_kwargs,
            system_name="Reasoning RAG w/o Integration",
            use_decomposition=True,
            use_multi_hop=True,
            use_evidence_integration=False,
            use_abstention=True
        )
    if system_id == 'reasoning_wo_abstention':
        return "Reasoning RAG w/o Abstention", ReasoningRAG(
            **shared_kwargs,
            system_name="Reasoning RAG w/o Abstention",
            use_decomposition=True,
            use_multi_hop=True,
            use_evidence_integration=True,
            use_abstention=False
        )
    raise ValueError(f"Unknown system id: {system_id}")


def build_comparison_systems(config: Config, reference_system: ReasoningRAG,
                             compare_profile: str, skip_ablations: bool):
    """构建对比实验中的所有系统，并共享同一向量索引。"""
    systems = []
    for system_id in get_profile_system_ids(compare_profile, skip_ablations):
        systems.append(instantiate_system(system_id, config, reference_system))
    return systems


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown_table(headers, rows):
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row] + body_rows)


def to_serializable(value):
    """将 numpy/Path 等对象转换为 JSON 可序列化格式。"""
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return value
    return value


def parse_seed_values(args, config: Config) -> list:
    if getattr(args, 'seeds', None):
        return [int(seed.strip()) for seed in args.seeds.split(',') if seed.strip()]

    if getattr(args, 'num_seeds', 1) and args.num_seeds > 1:
        default_pool = [config.RANDOM_SEED, 123, 456, 789, 2024]
        return default_pool[:args.num_seeds]

    return [args.seed]


def resolve_embedding_models(args, config: Config) -> list:
    if getattr(args, 'embedding_models', None):
        models = [model.strip() for model in args.embedding_models.split(',') if model.strip()]
        return models or [config.EMBEDDING_MODEL]

    if getattr(args, 'compare_profile', 'core') == 'maximal':
        candidates = [
            config.EMBEDDING_MODEL,
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
        resolved = []
        for model in candidates:
            if model not in resolved:
                resolved.append(model)
        return resolved

    return [config.EMBEDDING_MODEL]


def resolve_index_path(base_index_path: str, embedding_model: str, multi_embedding: bool) -> str:
    if not multi_embedding:
        return base_index_path

    path = Path(base_index_path)
    suffix = path.suffix or ".pkl"
    stem = path.stem or "bioasq_index"
    embedding_slug = sanitize_slug(embedding_model)
    return str(path.with_name(f"{stem}_{embedding_slug}{suffix}"))


def metric_value(entry):
    if isinstance(entry, dict) and 'mean' in entry:
        return entry['mean']
    return entry


def format_chart_value(value: float, percentage: bool) -> str:
    if percentage:
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def save_grouped_bar_chart(output_path: Path, title: str, x_labels, series_map: dict,
                           y_label: str, percentage: bool = False):
    """保存带数值标注的论文风格 SVG 分组柱状图。"""
    width = 1120
    height = 620
    margin_left = 95
    margin_right = 40
    margin_top = 75
    margin_bottom = 145
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    palette = ["#4C78A8", "#E45756", "#54A24B", "#B279A2", "#F58518"]

    series_items = list(series_map.items())
    series_count = max(len(series_items), 1)
    max_value = max(
        max(values) for _, values in series_items if values
    ) if series_items else 1.0
    if percentage:
        max_value = max(max_value, 1.0)
    max_value *= 1.18
    if max_value <= 0:
        max_value = 1.0

    group_width = chart_width / max(len(x_labels), 1)
    bar_width = min(44, (group_width * 0.72) / series_count)
    group_inner_width = bar_width * series_count

    def y_coord(value: float) -> float:
        return margin_top + chart_height - (max(0.0, value) / max_value) * chart_height

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="24" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="24" y="{margin_top + chart_height / 2}" text-anchor="middle" transform="rotate(-90 24 {margin_top + chart_height / 2})" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#374151">{escape(y_label)}</text>',
    ]

    tick_count = 5
    for tick in range(tick_count + 1):
        value = max_value * tick / tick_count
        y = y_coord(value)
        label = f"{value * 100:.0f}%" if percentage else f"{value:.2f}"
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 14}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#6B7280">{escape(label)}</text>'
        )

    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#111827" stroke-width="1.5"/>'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" stroke="#111827" stroke-width="1.5"/>'
    )

    for group_index, group_label in enumerate(x_labels):
        group_start_x = margin_left + group_index * group_width + (group_width - group_inner_width) / 2
        group_center_x = margin_left + group_index * group_width + group_width / 2
        for series_index, (_, values) in enumerate(series_items):
            value = values[group_index]
            x = group_start_x + series_index * bar_width
            y = y_coord(value)
            height_value = margin_top + chart_height - y
            color = palette[series_index % len(palette)]
            svg_parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 4:.2f}" height="{height_value:.2f}" fill="{color}" rx="3"/>'
            )
            svg_parts.append(
                f'<text x="{x + (bar_width - 4) / 2:.2f}" y="{max(y - 8, margin_top - 2):.2f}" text-anchor="middle" font-size="11" font-family="Arial, Helvetica, sans-serif" fill="#111827">{escape(format_chart_value(value, percentage))}</text>'
            )

        svg_parts.append(
            f'<text x="{group_center_x:.2f}" y="{height - 88}" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#374151">{escape(str(group_label))}</text>'
        )

    legend_x = margin_left
    legend_y = height - 38
    legend_gap = 220
    for series_index, (series_label, _) in enumerate(series_items):
        color = palette[series_index % len(palette)]
        item_x = legend_x + series_index * legend_gap
        svg_parts.append(
            f'<rect x="{item_x}" y="{legend_y - 12}" width="20" height="12" fill="{color}" rx="2"/>'
        )
        svg_parts.append(
            f'<text x="{item_x + 28}" y="{legend_y - 1}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#111827">{escape(series_label)}</text>'
        )

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def generate_visualizations(run_dir: Path, metrics_by_system: dict, include_ablations: bool):
    """为实验结果生成论文风格图表。"""
    figures = {}
    main_systems = [
        system_name for system_name in [
            "Traditional RAG",
            "Traditional RAG + Abstention",
            "Hybrid Retrieval Baseline",
            "Transparent Single-Hop RAG",
            "Reasoning RAG",
        ]
        if system_name in metrics_by_system
    ]

    if main_systems:
        save_grouped_bar_chart(
            run_dir / "main_results_line.svg",
            "Main Results Comparison",
            ["Exact Match", "Token F1", "Answer Coverage", "Retrieval Success"],
            {
                system_name: [
                    metric_value(metrics_by_system[system_name].get("exact_match", 0.0)),
                    metric_value(metrics_by_system[system_name].get("token_f1", 0.0)),
                    metric_value(metrics_by_system[system_name].get("answer_coverage", 0.0)),
                    metric_value(metrics_by_system[system_name].get("retrieval_success_rate", 0.0)),
                ]
                for system_name in main_systems
            },
            "Score",
            percentage=True
        )
        figures["main_results_line"] = "main_results_line.svg"

        save_grouped_bar_chart(
            run_dir / "transparency_score.svg",
            "Transparency and Traceability",
            ["V-Score", "Reasoning Path", "Traceability", "Provenance", "Decomp Visibility", "Retrieval Visibility"],
            {
                system_name: [
                    metric_value(metrics_by_system[system_name].get("v_score", 0.0)),
                    metric_value(metrics_by_system[system_name].get("reasoning_path_availability_rate", 0.0)),
                    metric_value(metrics_by_system[system_name].get("evidence_traceability_rate", 0.0)),
                    metric_value(metrics_by_system[system_name].get("provenance_completeness_score", 0.0)),
                    metric_value(metrics_by_system[system_name].get("decomposition_visibility_rate", 0.0)),
                    metric_value(metrics_by_system[system_name].get("retrieval_path_visibility_rate", 0.0)),
                ]
                for system_name in main_systems
            },
            "Score",
            percentage=True
        )
        figures["transparency_score"] = "transparency_score.svg"

        save_grouped_bar_chart(
            run_dir / "efficiency_score.svg",
            "Efficiency and Cost",
            ["Latency", "Retrieval Calls", "Avg Hops", "LLM Calls"],
            {
                system_name: [
                    metric_value(metrics_by_system[system_name].get("avg_latency_seconds", 0.0)),
                    metric_value(metrics_by_system[system_name].get("avg_total_retrievals", 0.0)),
                    metric_value(metrics_by_system[system_name].get("avg_hops", 0.0)),
                    metric_value(metrics_by_system[system_name].get("avg_llm_calls", 0.0)),
                ]
                for system_name in main_systems
            },
            "Value",
            percentage=False
        )
        figures["efficiency_score"] = "efficiency_score.svg"

        save_grouped_bar_chart(
            run_dir / "complex_query_score.svg",
            "Complex Query Analysis",
            ["Complex F1", "Complex Coverage", "Complex V-Score"],
            {
                system_name: [
                    metric_value(metrics_by_system[system_name].get("complexity_stratification", {}).get("complex", {}).get("token_f1", 0.0)),
                    metric_value(metrics_by_system[system_name].get("complexity_stratification", {}).get("complex", {}).get("answer_coverage", 0.0)),
                    metric_value(metrics_by_system[system_name].get("complexity_stratification", {}).get("complex", {}).get("v_score", 0.0)),
                ]
                for system_name in main_systems
            },
            "Score",
            percentage=True
        )
        figures["complex_query_score"] = "complex_query_score.svg"

        query_types = sorted({
            query_type
            for system_name in main_systems
            for query_type in metrics_by_system[system_name].get("query_type_breakdown", {}).keys()
        })
        if query_types:
            save_grouped_bar_chart(
                run_dir / "query_type_f1_line.svg",
                "Token F1 by Query Type",
                query_types,
                {
                    system_name: [
                        metric_value(metrics_by_system[system_name].get("query_type_breakdown", {}).get(query_type, {}).get("token_f1", 0.0))
                        for query_type in query_types
                    ]
                    for system_name in main_systems
                },
                "Token F1",
                percentage=True
            )
            figures["query_type_f1_line"] = "query_type_f1_line.svg"

    if include_ablations:
        ablation_systems = [
            "Reasoning RAG",
            "Reasoning RAG w/o Decomposition",
            "Reasoning RAG w/o Multi-Hop",
            "Reasoning RAG w/o Integration",
            "Reasoning RAG w/o Abstention"
        ]
        available_ablation_systems = [
            system_name for system_name in ablation_systems
            if system_name in metrics_by_system
        ]
        if available_ablation_systems:
            ablation_labels = {
                "Reasoning RAG": "Full",
                "Reasoning RAG w/o Decomposition": "w/o Decomp",
                "Reasoning RAG w/o Multi-Hop": "w/o Multi-Hop",
                "Reasoning RAG w/o Integration": "w/o Integration",
                "Reasoning RAG w/o Abstention": "w/o Abstention"
            }
            ablation_series = {
                "Token F1": [
                    metrics_by_system[system_name].get("token_f1", 0.0)
                    for system_name in available_ablation_systems
                ],
                "Coverage": [
                    metrics_by_system[system_name].get("answer_coverage", 0.0)
                    for system_name in available_ablation_systems
                ],
                "Retrieval Success": [
                    metrics_by_system[system_name].get("retrieval_success_rate", 0.0)
                    for system_name in available_ablation_systems
                ]
            }
            ablation_path = run_dir / "ablation_line.svg"
            save_grouped_bar_chart(
                ablation_path,
                "Ablation Study",
                [ablation_labels.get(name, name) for name in available_ablation_systems],
                ablation_series,
                "Score",
                percentage=True
            )
            figures["ablation_line"] = ablation_path.name

    return figures


def build_report_markdown(metadata: dict, metrics_by_system: dict,
                          include_ablations: bool, figures: dict = None) -> str:
    """生成对比实验 markdown 报告。"""
    figures = figures or {}
    main_systems = [
        system_name for system_name in [
            "Traditional RAG",
            "Traditional RAG + Abstention",
            "Hybrid Retrieval Baseline",
            "Transparent Single-Hop RAG",
            "Reasoning RAG",
        ]
        if system_name in metrics_by_system
    ]

    main_rows = []
    for system_name in main_systems:
        metrics = metrics_by_system.get(system_name, {})
        if not metrics:
            continue
        main_rows.append([
            system_name,
            format_percent(metric_value(metrics.get('exact_match', 0.0))),
            format_percent(metric_value(metrics.get('token_f1', 0.0))),
            format_percent(metric_value(metrics.get('answer_coverage', 0.0))),
            format_percent(metric_value(metrics.get('retrieval_success_rate', 0.0))),
            f"{metric_value(metrics.get('avg_similarity_score', 0.0)):.3f}",
            f"{metric_value(metrics.get('avg_latency_seconds', 0.0)):.3f}"
        ])

    ablation_rows = []
    if include_ablations:
        for system_name in [
            "Reasoning RAG",
            "Reasoning RAG w/o Decomposition",
            "Reasoning RAG w/o Multi-Hop",
            "Reasoning RAG w/o Integration",
            "Reasoning RAG w/o Abstention"
        ]:
            metrics = metrics_by_system.get(system_name, {})
            if not metrics:
                continue
            ablation_rows.append([
                system_name,
                format_percent(metrics.get('exact_match', 0.0)),
                format_percent(metrics.get('token_f1', 0.0)),
                format_percent(metrics.get('answer_coverage', 0.0)),
                f"{metrics.get('avg_hops', 0.0):.2f}",
                f"{metrics.get('avg_total_retrievals', 0.0):.2f}",
                format_percent(metrics.get('abstention_rate', 0.0))
            ])

    query_types = sorted({
        query_type
        for system_name in main_systems
        for query_type in metrics_by_system.get(system_name, {}).get('query_type_breakdown', {}).keys()
    })
    query_type_rows = []
    for query_type in query_types:
        counts = [
            metrics_by_system.get(system_name, {}).get('query_type_breakdown', {}).get(query_type, {}).get('count', 0)
            for system_name in main_systems
        ]
        row = [query_type, str(max(counts) if counts else 0)]
        for system_name in main_systems:
            breakdown = metrics_by_system.get(system_name, {}).get('query_type_breakdown', {}).get(query_type, {})
            row.append(format_percent(metric_value(breakdown.get('token_f1', 0.0))))
            row.append(format_percent(metric_value(breakdown.get('v_score', 0.0))))
        query_type_rows.append(row)

    report_lines = [
        "# Reasoning RAG Comparison Report",
        "",
        "## Experiment Setup",
        f"- Dataset: {metadata['dataset']}",
        f"- Split policy: {metadata['split_policy']}",
        f"- Index path: `{metadata['index_path']}`",
        f"- Seed: {metadata['seed']}",
        f"- Eval size: {metadata['eval_size']}",
        f"- Max passages: {metadata['max_passages']}",
        f"- Compare profile: {metadata.get('compare_profile', 'core')}",
        f"- Embedding model: `{metadata.get('embedding_model', 'n/a')}`",
        f"- Aggregate mode: {metadata.get('aggregate_mode', 'single-run')}",
        "",
        "## Main Performance Comparison",
        build_markdown_table(
            ["System", "EM", "F1", "Coverage", "Retrieval Success", "Avg Similarity", "Latency (s)"],
            main_rows
        ),
    ]

    if figures.get("main_results_line"):
        report_lines.extend([
            "",
            "## Main Results Figure",
            f"![Main Results Chart](./{figures['main_results_line']})"
        ])

    transparency_rows = []
    for system_name in main_systems:
        metrics = metrics_by_system.get(system_name, {})
        if not metrics:
            continue
        transparency_rows.append([
            system_name,
            format_percent(metric_value(metrics.get('v_score', 0.0))),
            format_percent(metric_value(metrics.get('reasoning_path_availability_rate', 0.0))),
            format_percent(metric_value(metrics.get('evidence_traceability_rate', 0.0))),
            format_percent(metric_value(metrics.get('provenance_completeness_score', 0.0))),
            format_percent(metric_value(metrics.get('decomposition_visibility_rate', 0.0))),
            format_percent(metric_value(metrics.get('retrieval_path_visibility_rate', 0.0))),
            format_percent(metric_value(metrics.get('complex_query_transparency_score', 0.0))),
        ])
    if transparency_rows:
        report_lines.extend([
            "",
            "## Transparency Metrics",
            build_markdown_table(
                [
                    "System",
                    "V-Score",
                    "Reasoning Path",
                    "Traceability",
                    "Provenance",
                    "Decomp Visibility",
                    "Retrieval Visibility",
                    "Complex Transparency"
                ],
                transparency_rows
            )
        ])
        if figures.get("transparency_score"):
            report_lines.extend([
                "",
                "## Transparency Figure",
                f"![Transparency Score Chart](./{figures['transparency_score']})"
            ])

    efficiency_rows = []
    for system_name in main_systems:
        metrics = metrics_by_system.get(system_name, {})
        if not metrics:
            continue
        efficiency_rows.append([
            system_name,
            f"{metric_value(metrics.get('avg_latency_seconds', 0.0)):.3f}",
            f"{metric_value(metrics.get('avg_total_retrievals', 0.0)):.2f}",
            f"{metric_value(metrics.get('avg_hops', 0.0)):.2f}",
            f"{metric_value(metrics.get('avg_llm_calls', 0.0)):.2f}",
        ])
    if efficiency_rows:
        report_lines.extend([
            "",
            "## Cost And Efficiency",
            build_markdown_table(
                ["System", "Latency (s)", "Retrieval Calls", "Avg Hops", "LLM Calls"],
                efficiency_rows
            )
        ])
        if figures.get("efficiency_score"):
            report_lines.extend([
                "",
                "## Efficiency Figure",
                f"![Efficiency Score Chart](./{figures['efficiency_score']})"
            ])

    if ablation_rows:
        report_lines.extend([
            "",
            "## Ablation Study",
            build_markdown_table(
                ["System", "EM", "F1", "Coverage", "Avg Hops", "Avg Retrievals", "Abstention Rate"],
                ablation_rows
            )
        ])
        if figures.get("ablation_line"):
            report_lines.extend([
                "",
                "## Ablation Figure",
                f"![Ablation Chart](./{figures['ablation_line']})"
            ])

    if query_type_rows:
        report_lines.extend([
            "",
            "## Query Type And Complexity Analysis",
            build_markdown_table(
                ["Query Type", "Count"] + [
                    label
                    for system_name in main_systems
                    for label in (f"{system_name} F1", f"{system_name} V-Score")
                ],
                query_type_rows
            )
        ])
        if figures.get("query_type_f1_line"):
            report_lines.extend([
                "",
                "## Query Type Figure",
                f"![Query Type Token F1 Chart](./{figures['query_type_f1_line']})"
            ])

    complex_rows = []
    for system_name in main_systems:
        complexity_metrics = metrics_by_system.get(system_name, {}).get('complexity_stratification', {}).get('complex', {})
        if not complexity_metrics:
            continue
        complex_rows.append([
            system_name,
            str(complexity_metrics.get('count', 0)),
            format_percent(metric_value(complexity_metrics.get('token_f1', 0.0))),
            format_percent(metric_value(complexity_metrics.get('answer_coverage', 0.0))),
            format_percent(metric_value(complexity_metrics.get('v_score', 0.0))),
        ])
    if complex_rows:
        report_lines.extend([
            "",
            "## Complex Query Results",
            build_markdown_table(
                ["System", "Count", "Complex F1", "Complex Coverage", "Complex V-Score"],
                complex_rows
            )
        ])
        if figures.get("complex_query_score"):
            report_lines.extend([
                "",
                "## Complex Query Figure",
                f"![Complex Query Score Chart](./{figures['complex_query_score']})"
            ])

    report_lines.extend([
        "",
        "## Discussion",
        "- Transparency should be interpreted separately from raw answer quality; a higher V-Score does not automatically imply higher reliability.",
        "- The comparison between `Transparent Single-Hop RAG` and full `Reasoning RAG` helps isolate the value of visible process traces versus full multi-hop reasoning.",
        "- Use the ablation results to identify which modules contribute to quality gains and which modules currently trade performance for interpretability."
    ])

    return "\n".join(report_lines) + "\n"


def build_tables_payload(metrics_by_system: dict) -> dict:
    main_systems = [
        system_name for system_name in [
            "Traditional RAG",
            "Traditional RAG + Abstention",
            "Hybrid Retrieval Baseline",
            "Transparent Single-Hop RAG",
            "Reasoning RAG",
        ]
        if system_name in metrics_by_system
    ]

    tables = {
        'main_comparison': [],
        'transparency': [],
        'efficiency': [],
        'complex_queries': [],
        'ablations': [],
    }

    for system_name in main_systems:
        metrics = metrics_by_system[system_name]
        tables['main_comparison'].append({
            'system': system_name,
            'exact_match': metric_value(metrics.get('exact_match', 0.0)),
            'token_f1': metric_value(metrics.get('token_f1', 0.0)),
            'answer_coverage': metric_value(metrics.get('answer_coverage', 0.0)),
            'retrieval_success_rate': metric_value(metrics.get('retrieval_success_rate', 0.0)),
            'avg_similarity_score': metric_value(metrics.get('avg_similarity_score', 0.0)),
        })
        tables['transparency'].append({
            'system': system_name,
            'v_score': metric_value(metrics.get('v_score', 0.0)),
            'reasoning_path_availability_rate': metric_value(metrics.get('reasoning_path_availability_rate', 0.0)),
            'evidence_traceability_rate': metric_value(metrics.get('evidence_traceability_rate', 0.0)),
            'provenance_completeness_score': metric_value(metrics.get('provenance_completeness_score', 0.0)),
            'retrieval_path_visibility_rate': metric_value(metrics.get('retrieval_path_visibility_rate', 0.0)),
        })
        tables['efficiency'].append({
            'system': system_name,
            'avg_latency_seconds': metric_value(metrics.get('avg_latency_seconds', 0.0)),
            'avg_total_retrievals': metric_value(metrics.get('avg_total_retrievals', 0.0)),
            'avg_hops': metric_value(metrics.get('avg_hops', 0.0)),
            'avg_llm_calls': metric_value(metrics.get('avg_llm_calls', 0.0)),
        })

        complex_metrics = metrics.get('complexity_stratification', {}).get('complex', {})
        if complex_metrics:
            tables['complex_queries'].append({
                'system': system_name,
                'count': complex_metrics.get('count', 0),
                'token_f1': metric_value(complex_metrics.get('token_f1', 0.0)),
                'answer_coverage': metric_value(complex_metrics.get('answer_coverage', 0.0)),
                'v_score': metric_value(complex_metrics.get('v_score', 0.0)),
            })

    for system_name in [
        "Reasoning RAG",
        "Reasoning RAG w/o Decomposition",
        "Reasoning RAG w/o Multi-Hop",
        "Reasoning RAG w/o Integration",
        "Reasoning RAG w/o Abstention",
    ]:
        metrics = metrics_by_system.get(system_name)
        if not metrics:
            continue
        tables['ablations'].append({
            'system': system_name,
            'token_f1': metric_value(metrics.get('token_f1', 0.0)),
            'answer_coverage': metric_value(metrics.get('answer_coverage', 0.0)),
            'avg_hops': metric_value(metrics.get('avg_hops', 0.0)),
            'avg_total_retrievals': metric_value(metrics.get('avg_total_retrievals', 0.0)),
        })

    return tables


def aggregate_experiment_runs(run_summaries: list) -> dict:
    grouped = {}
    for run_summary in run_summaries:
        embedding_model = run_summary['metadata']['embedding_model']
        embedding_group = grouped.setdefault(embedding_model, {})
        for system_name, metrics in run_summary['metrics_by_system'].items():
            embedding_group.setdefault(system_name, []).append(metrics)

    aggregate_by_embedding = {}
    for embedding_model, systems in grouped.items():
        aggregate_by_embedding[embedding_model] = {
            system_name: RAGEvaluator.aggregate_metrics(metric_runs, system_name=system_name)
            for system_name, metric_runs in systems.items()
        }
    return aggregate_by_embedding


def save_comparison_outputs(output_dir: str, metadata: dict, metrics_by_system: dict,
                            results_by_system: dict, include_ablations: bool,
                            aggregate_summary: dict = None, tables_payload: dict = None):
    """保存对比实验的 JSON 与 Markdown 报告。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"compare_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        'metadata': metadata,
        'metrics_by_system': metrics_by_system
    }
    detailed_payload = {
        'metadata': metadata,
        'metrics_by_system': metrics_by_system,
        'results_by_system': results_by_system
    }

    summary_path = run_dir / "summary.json"
    details_path = run_dir / "detailed_results.json"
    report_path = run_dir / "report.md"
    aggregate_path = run_dir / "aggregate_summary.json"
    tables_path = run_dir / "tables.json"
    figures = generate_visualizations(run_dir, metrics_by_system, include_ablations)

    with open(summary_path, 'w', encoding='utf-8') as file_obj:
        json.dump(to_serializable(summary_payload), file_obj, ensure_ascii=False, indent=2)
    with open(details_path, 'w', encoding='utf-8') as file_obj:
        json.dump(to_serializable(detailed_payload), file_obj, ensure_ascii=False, indent=2)
    with open(report_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(build_report_markdown(metadata, metrics_by_system, include_ablations, figures))
    if aggregate_summary is not None:
        with open(aggregate_path, 'w', encoding='utf-8') as file_obj:
            json.dump(to_serializable(aggregate_summary), file_obj, ensure_ascii=False, indent=2)
    if tables_payload is not None:
        with open(tables_path, 'w', encoding='utf-8') as file_obj:
            json.dump(to_serializable(tables_payload), file_obj, ensure_ascii=False, indent=2)

    logger.info(f"Saved comparison summary to {summary_path}")
    logger.info(f"Saved detailed results to {details_path}")
    logger.info(f"Saved markdown report to {report_path}")
    if aggregate_summary is not None:
        logger.info(f"Saved aggregate summary to {aggregate_path}")
    if tables_payload is not None:
        logger.info(f"Saved table payloads to {tables_path}")


def run_single_comparison(config: Config, data_loader: DataLoader, args,
                          embedding_model: str, seed: int):
    local_config = create_config_instance(config, embedding_model=embedding_model)
    local_args = SimpleNamespace(**vars(args))
    local_args.seed = seed
    local_args.index_path = resolve_index_path(
        args.index_path,
        embedding_model,
        multi_embedding=len(resolve_embedding_models(args, config)) > 1
    )

    reference_system = ReasoningRAG(local_config)
    ensure_index(reference_system, data_loader, local_args)
    test_questions = select_questions(data_loader, 'test', args.eval_size, seed)

    systems = build_comparison_systems(
        local_config,
        reference_system,
        args.compare_profile,
        args.skip_ablations
    )
    metrics_by_system = {}
    results_by_system = {}
    for system_name, system in systems:
        metrics, results = run_evaluation(system, test_questions, system_name)
        metrics_by_system[system_name] = metrics
        results_by_system[system_name] = results

    return {
        'metadata': {
            'dataset': 'BioASQ (enelpol/rag-mini-bioasq)',
            'split_policy': 'Train passages for indexing, test questions for evaluation',
            'index_path': local_args.index_path,
            'seed': seed,
            'eval_size': len(test_questions),
            'max_passages': args.max_passages,
            'ablations_included': not args.skip_ablations,
            'compare_profile': args.compare_profile,
            'embedding_model': embedding_model,
            'aggregate_mode': 'single-run',
        },
        'metrics_by_system': metrics_by_system,
        'results_by_system': results_by_system,
    }


def run_comparison(config: Config, data_loader: DataLoader, args):
    """统一运行 baseline、full model、消融、多 seed 与多 embedding 比较实验。"""
    logger.info(f"\n{'#'*80}")
    logger.info("# RUNNING COMPARISON EXPERIMENT")
    logger.info(f"{'#'*80}\n")

    embedding_models = resolve_embedding_models(args, config)
    seed_values = parse_seed_values(args, config)
    run_summaries = []

    for embedding_model in embedding_models:
        logger.info("Embedding profile: %s", embedding_model)
        for seed in seed_values:
            logger.info("Running comparison for seed=%s", seed)
            run_summaries.append(
                run_single_comparison(config, data_loader, args, embedding_model, seed)
            )

    aggregate_by_embedding = aggregate_experiment_runs(run_summaries)
    primary_embedding = embedding_models[0]
    primary_aggregate = aggregate_by_embedding[primary_embedding]
    tables_payload = build_tables_payload(primary_aggregate)
    aggregate_summary = {
        'metadata': {
            'dataset': 'BioASQ (enelpol/rag-mini-bioasq)',
            'split_policy': 'Train passages for indexing, test questions for evaluation',
            'compare_profile': args.compare_profile,
            'embedding_models': embedding_models,
            'seeds': seed_values,
            'eval_size': args.eval_size,
            'max_passages': args.max_passages,
        },
        'aggregate_by_embedding': aggregate_by_embedding,
        'run_summaries': run_summaries,
    }
    primary_metadata = {
        'dataset': 'BioASQ (enelpol/rag-mini-bioasq)',
        'split_policy': 'Train passages for indexing, test questions for evaluation',
        'index_path': resolve_index_path(args.index_path, primary_embedding, len(embedding_models) > 1),
        'seed': ','.join(str(seed) for seed in seed_values),
        'eval_size': args.eval_size,
        'max_passages': args.max_passages,
        'ablations_included': not args.skip_ablations,
        'compare_profile': args.compare_profile,
        'embedding_model': primary_embedding,
        'aggregate_mode': f'{len(seed_values)} seed(s) mean/std',
    }
    save_comparison_outputs(
        args.output_dir,
        primary_metadata,
        primary_aggregate,
        {'runs': run_summaries},
        include_ablations=not args.skip_ablations,
        aggregate_summary=aggregate_summary,
        tables_payload=tables_payload
    )
    return primary_aggregate, {'runs': run_summaries}


def main():
    config = Config()
    parser = argparse.ArgumentParser(description='Reasoning RAG System')
    parser.add_argument(
        '--mode', type=str, default='demo',
        choices=['build', 'eval', 'compare', 'demo', 'interactive', 'all'],
        help='Run mode'
    )
    parser.add_argument(
        '--index-path', type=str, default='./bioasq_index.pkl',
        help='Path to save/load the vector index'
    )
    parser.add_argument(
        '--eval-size', type=int, default=config.TEST_SAMPLE_SIZE,
        help='Number of samples for evaluation/comparison'
    )
    parser.add_argument(
        '--demo-size', type=int, default=config.DEMO_SAMPLE_SIZE,
        help='Number of samples for demo'
    )
    parser.add_argument(
        '--seed', type=int, default=config.RANDOM_SEED,
        help='Random seed for reproducible sampling'
    )
    parser.add_argument(
        '--num-seeds', type=int, default=1,
        help='Number of seeds to run for comparison aggregation'
    )
    parser.add_argument(
        '--seeds', type=str, default='',
        help='Comma-separated explicit seed list (overrides --num-seeds)'
    )
    parser.add_argument(
        '--rebuild-index', action='store_true',
        help='Force rebuild the index even if it exists'
    )
    parser.add_argument(
        '--max-passages', type=int, default=2000,
        help='Max passages for train-only index (doubled due to Q+A combined passages)'
    )
    parser.add_argument(
        '--full-index', action='store_true',
        help=(
            'Index ALL passages (train + test) for interactive/production use. '
            'Do NOT use this flag when running evaluation or comparison.'
        )
    )
    parser.add_argument(
        '--output-dir', type=str, default=config.EXPERIMENT_OUTPUT_DIR,
        help='Directory for comparison outputs'
    )
    parser.add_argument(
        '--skip-ablations', action='store_true',
        help='Only compare Traditional RAG and Reasoning RAG'
    )
    parser.add_argument(
        '--compare-profile', type=str, default='core',
        choices=['core', 'extended', 'maximal'],
        help='Comparison profile defining which RAG variants to include'
    )
    parser.add_argument(
        '--embedding-models', type=str, default='',
        help='Comma-separated embedding model list for comparison runs'
    )

    args = parser.parse_args()

    if args.mode in {'eval', 'compare', 'all'} and args.full_index:
        raise ValueError(
            "--full-index should not be used for evaluation/comparison because it causes test leakage."
        )

    data_loader = DataLoader(random_seed=config.RANDOM_SEED)
    log_llm_status()

    if args.mode == 'build':
        rag_system = ReasoningRAG(config)
        ensure_index(rag_system, data_loader, args)
    elif args.mode == 'eval':
        rag_system = ReasoningRAG(config)
        ensure_index(rag_system, data_loader, args)
        test_questions = select_questions(data_loader, 'test', args.eval_size, args.seed)
        run_evaluation(rag_system, test_questions, "Reasoning RAG")
    elif args.mode == 'compare':
        run_comparison(config, data_loader, args)
    elif args.mode == 'demo':
        rag_system = ReasoningRAG(config)
        ensure_index(rag_system, data_loader, args)
        demo_questions = select_questions(data_loader, 'test', args.demo_size, args.seed)
        run_demo(rag_system, demo_questions)
    elif args.mode == 'interactive':
        rag_system = ReasoningRAG(config)
        ensure_index(rag_system, data_loader, args)
        run_interactive(rag_system)
    elif args.mode == 'all':
        rag_system = ReasoningRAG(config)
        ensure_index(rag_system, data_loader, args)
        test_questions = select_questions(data_loader, 'test', args.eval_size, args.seed)
        run_evaluation(rag_system, test_questions, "Reasoning RAG")
        demo_questions = select_questions(data_loader, 'test', args.demo_size, args.seed)
        run_demo(rag_system, demo_questions)

    logger.info("\nAll tasks completed!")


if __name__ == '__main__':
    main()