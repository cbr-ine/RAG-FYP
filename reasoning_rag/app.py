#!/usr/bin/env python3
"""
Reasoning RAG — Web Chat Interface
────────────────────────────────────
Run:   python app.py
Open:  http://localhost:5000   (or set PORT, e.g. PORT=5001 if 5000 is in use — common on macOS / AirPlay)
"""

from flask import Flask, request, jsonify, render_template
import logging
import traceback
import os
from pathlib import Path

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  索引文件路径
#  与 main.py / interactive.py 保持一致，也可通过环境变量覆盖：
#    set RAG_INDEX_PATH=C:\path\to\your_index.pkl   (Windows)
#    export RAG_INDEX_PATH=/path/to/your_index.pkl  (Linux/Mac)
# ═══════════════════════════════════════════════════════════════════════
INDEX_PATH = os.getenv("RAG_INDEX_PATH", "./bioasq_index.pkl")

_rag = None
_conversation_rags = {}


def _create_rag():
    from reasoning_rag import ReasoningRAG
    from config import Config
    return ReasoningRAG(Config())


def _build_base_rag():
    rag = _create_rag()
    if os.path.exists(INDEX_PATH):
        logger.info(f"Loading index from {INDEX_PATH} …")
        rag.load_index(INDEX_PATH)
        logger.info(
            f"Index loaded ✓  "
            f"({rag.vector_store.index.ntotal} vectors, "
            f"{len(rag.vector_store.passages)} passages)"
        )
    else:
        logger.error(
            f"\n{'!'*60}\n"
            f"  Index file NOT found: {INDEX_PATH}\n"
            f"  Please build it first:\n"
            f"    python main.py --mode build --full-index\n"
            f"  Then restart app.py\n"
            f"{'!'*60}\n"
        )
    return rag


def _clone_base_rag():
    base_rag = get_rag()
    from reasoning_rag import ReasoningRAG
    cloned_rag = ReasoningRAG(
        base_rag.config,
        embedder=base_rag.embedder,
        vector_store=base_rag.vector_store.clone(),
        system_name=base_rag.system_name,
        use_decomposition=base_rag.use_decomposition,
        use_multi_hop=base_rag.use_multi_hop,
        use_evidence_integration=base_rag.use_evidence_integration,
        use_abstention=base_rag.use_abstention
    )
    return cloned_rag


def get_rag(conversation_id: str = None):
    global _rag
    global _conversation_rags
    if _rag is None:
        logger.info("Initialising Reasoning RAG system …")
        _rag = _build_base_rag()
        logger.info("Reasoning RAG ready ✓")

    if conversation_id:
        return _conversation_rags.get(conversation_id, _rag)
    return _rag


def reset_conversation_rag(conversation_id: str):
    if conversation_id in _conversation_rags:
        del _conversation_rags[conversation_id]


# ═══════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════
def _get(obj, *keys, default=None):
    """安全地从 dict 或对象中取值，支持链式 key。"""
    for k in keys:
        if obj is None:
            return default
        try:
            obj = obj[k] if isinstance(obj, dict) else getattr(obj, k)
        except (KeyError, AttributeError, TypeError):
            return default
    return default if obj is None else obj


def _query_analysis_visibility(analysis: dict) -> float:
    required_fields = [
        "complexity_score",
        "is_complex",
        "requires_decomposition",
        "reasoning_depth",
        "query_type",
    ]
    return sum(1 for field in required_fields if field in analysis) / len(required_fields)


def _reasoning_path_visibility(path: list) -> float:
    if not path:
        return 0.0
    visible_steps = sum(1 for step in path if _get(step, "type", default="") and _get(step, "description", default=""))
    return visible_steps / len(path)


def _evidence_traceability(evidence_list: list) -> float:
    if not evidence_list:
        return 0.0
    traceable = 0
    for evidence in evidence_list:
        if (
            _get(evidence, "text", default="")
            and _get(evidence, "source_type", default="") not in ("", "unknown")
            and _get(evidence, "hop", default=None) is not None
        ):
            traceable += 1
    return traceable / len(evidence_list)


def _provenance_completeness(evidence_list: list) -> float:
    if not evidence_list:
        return 0.0
    required_fields = ["text", "similarity", "quality_score", "hop", "source_type"]
    per_evidence_scores = []
    for evidence in evidence_list:
        present_fields = 0
        for field in required_fields:
            value = _get(evidence, field, default=None)
            if value not in (None, "", "unknown"):
                present_fields += 1
        per_evidence_scores.append(present_fields / len(required_fields))
    return sum(per_evidence_scores) / len(per_evidence_scores)


def _decomposition_visibility(analysis: dict, subqueries: list) -> float:
    requires_decomposition = _get(analysis, "requires_decomposition", default=False)
    if requires_decomposition:
        meaningful_subqueries = [sq for sq in subqueries if _get(sq, "subquery", default="")]
        return min(1.0, len(meaningful_subqueries) / 2) if meaningful_subqueries else 0.0
    return 1.0


def _retrieval_path_visibility(subquery_results: list) -> float:
    if not subquery_results:
        return 0.0
    visible_paths = 0
    for subquery_result in subquery_results:
        hops = _get(subquery_result, "hops", default=[]) or []
        if hops and all(("hop_number" in hop and "query" in hop) for hop in hops):
            visible_paths += 1
    return visible_paths / len(subquery_results)


def _compute_v_score(result: dict) -> float:
    analysis = _get(result, "analysis", default={}) or {}
    path = _get(result, "answer", "reasoning_path", default=[]) or []
    evidence_list = _get(result, "answer", "sources", default=[]) or []
    subqueries = _get(result, "subqueries", default=[]) or []
    subquery_results = _get(result, "retrieval", "subquery_results", default=[]) or []

    components = [
        _query_analysis_visibility(analysis),
        _reasoning_path_visibility(path),
        _evidence_traceability(evidence_list),
        _provenance_completeness(evidence_list),
        _decomposition_visibility(analysis, subqueries),
        _retrieval_path_visibility(subquery_results),
    ]
    return sum(components) / len(components)


def _make_offset_passages(passages: list, start_id: int) -> list:
    adjusted = []
    for offset, passage in enumerate(passages):
        item = passage.copy()
        item["id"] = start_id + offset
        adjusted.append(item)
    return adjusted


def to_json(result, question: str) -> dict:
    analysis      = _get(result, "analysis",  default={}) or {}
    retrieval     = _get(result, "retrieval", default={}) or {}
    answer_r      = _get(result, "answer",    default={}) or {}
    integrated_ev = _get(result, "evidence",  default={}) or {}
    subqueries    = _get(result, "subqueries", default=[]) or []

    ev_list   = _get(integrated_ev, "evidence", default=[]) or []
    ev_stats  = _get(integrated_ev, "stats",    default={}) or {}
    ret_stats = _get(retrieval, "retrieval_stats", default={}) or {}
    path      = _get(answer_r, "reasoning_path", default=[]) or []
    v_score   = _compute_v_score(result)

    return {
        "question":   question,

        "answer":     _get(answer_r, "answer",     default=""),
        "confidence": float(_get(answer_r, "confidence", default=0.0)),
        "v_score":    round(float(v_score), 3),

        "analysis": {
            "complexity_score":    float(_get(analysis, "complexity_score",       default=0.0)),
            "classification":            _get(analysis, "classification",         default=""),
            "needs_decomposition": bool( _get(analysis, "requires_decomposition", default=False)),
            "reasoning_depth":           _get(analysis, "reasoning_depth",        default=""),
        },

        "stats": {
            "total_retrievals": _get(ret_stats, "total_retrievals",      default=0),
            "successful":       _get(ret_stats, "successful_retrievals", default="—"),
            "avg_similarity":   round(float(_get(ret_stats, "avg_similarity", default=0.0)), 3),
            "total_evidence":        _get(ev_stats, "total_evidence_count",    default=len(ev_list)),
            "high_quality_evidence": _get(ev_stats, "selected_evidence_count", default=0),
        },

        "transparency": {
            "query_analysis_visibility": round(_query_analysis_visibility(analysis), 3),
            "reasoning_path_visibility": round(_reasoning_path_visibility(path), 3),
            "evidence_traceability": round(_evidence_traceability(_get(answer_r, "sources", default=[]) or []), 3),
            "provenance_completeness": round(_provenance_completeness(_get(answer_r, "sources", default=[]) or []), 3),
            "decomposition_visibility": round(_decomposition_visibility(analysis, subqueries), 3),
            "retrieval_path_visibility": round(_retrieval_path_visibility(_get(retrieval, "subquery_results", default=[]) or []), 3),
        },

        "reasoning_path": [
            {
                "step": _get(s, "step", default=i + 1),
                "type": _get(s, "type", default=""),
                "desc": _get(s, "description", default=""),
            }
            for i, s in enumerate(path)
        ],

        "evidence": [
            {
                "rank":       i + 1,
                "relevance":  round(float(_get(e, "quality_score", default=0.0)), 3),
                "similarity": round(float(_get(e, "similarity",    default=0.0)), 3),
                "hop":        _get(e, "hop",         default=1),
                "type":       _get(e, "source_type", default=""),
                "text":       _get(e, "text",        default=""),
            }
            for i, e in enumerate(ev_list[:5])
        ],

        "metadata": {
            "query_type":         _get(analysis,      "query_type",         default=""),
            "integration_method": _get(integrated_ev, "integration_method", default=""),
            "total_steps":        len(path),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    try:
        body     = request.get_json(force=True) or {}
        question = body.get("question", "").strip()
        conversation_id = str(body.get("conversation_id", "")).strip() or None

        if not question:
            return jsonify({"ok": False, "error": "Question cannot be empty."}), 400

        logger.info(f"Received query: {question!r}")
        rag = get_rag(conversation_id)

        # 检查索引是否已加载
        if rag.vector_store.index.ntotal == 0:
            return jsonify({
                "ok": False,
                "error": (
                    f"Index is empty. Please build it first:\n"
                    f"  python main.py --mode build --full-index\n"
                    f"Then restart app.py. (Looking for: {INDEX_PATH})"
                )
            }), 503

        result = rag.query(question)
        payload = to_json(result, question)
        return jsonify({"ok": True, "data": payload})

    except Exception:
        logger.error(traceback.format_exc())
        return jsonify({"ok": False, "error": traceback.format_exc().splitlines()[-1]}), 500


@app.route("/api/import-documents", methods=["POST"])
def api_import_documents():
    global _conversation_rags
    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"ok": False, "error": "Please upload at least one document."}), 400

        conversation_id = str(request.form.get("conversation_id", "")).strip()
        if not conversation_id:
            return jsonify({"ok": False, "error": "A conversation must be active before importing documents."}), 400

        mode = (request.form.get("mode") or "append").strip().lower()
        if mode not in {"append", "overwrite"}:
            return jsonify({"ok": False, "error": "Unsupported import mode."}), 400

        chunk_size = int(request.form.get("chunk_size", "180"))
        text_field = (request.form.get("text_field") or "").strip() or None

        from document_ingestor import DocumentIngestor
        ingestor = DocumentIngestor(chunk_size=chunk_size, text_field=text_field)
        imported_passages, import_summary = ingestor.ingest_files(files, save_dir=None)

        if not imported_passages:
            return jsonify({"ok": False, "error": "No usable text content was found in the uploaded documents."}), 400

        if mode == "overwrite":
            rag = _clone_base_rag()
            import_passages = _make_offset_passages(imported_passages, len(rag.vector_store.passages))
        else:
            rag = _conversation_rags.get(conversation_id)
            if rag is None:
                rag = _clone_base_rag()
            saved_model = rag.vector_store.metadata.get("embedding_model")
            current_model = rag.embedder.model_name
            if not saved_model:
                return jsonify({
                    "ok": False,
                    "error": (
                        "The current index does not record its embedding model. "
                        "Please use overwrite mode once to rebuild the index with the new multilingual model."
                    )
                }), 400
            if saved_model != current_model:
                return jsonify({
                    "ok": False,
                    "error": (
                        f"Embedding model mismatch: current index uses '{saved_model}' but the app is using "
                        f"'{current_model}'. Please use overwrite mode or rebuild the index first."
                    )
                }), 400
            existing_texts = {passage.get("text", "") for passage in rag.vector_store.passages}
            unique_passages = [
                passage for passage in imported_passages
                if passage.get("text", "") not in existing_texts
            ]
            if not unique_passages:
                return jsonify({
                    "ok": False,
                    "error": "All uploaded chunks already exist in the current knowledge base."
                }), 400
            import_passages = _make_offset_passages(unique_passages, len(rag.vector_store.passages))

        rag.build_index(import_passages)
        _conversation_rags[conversation_id] = rag

        return jsonify({
            "ok": True,
            "data": {
                "mode": mode,
                "conversation_id": conversation_id,
                "imported_files": import_summary["files"],
                "file_count": import_summary["file_count"],
                "chunk_count": len(import_passages),
                "index_size": rag.vector_store.index.ntotal,
                "temporary": True,
                "message": (
                    f"Imported {import_summary['file_count']} file(s) and indexed "
                    f"{len(import_passages)} chunk(s) for the current conversation only."
                )
            }
        })
    except Exception as exc:
        logger.error(traceback.format_exc())
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/conversation-kb", methods=["DELETE"])
def api_clear_conversation_kb():
    body = request.get_json(force=True) or {}
    conversation_id = str(body.get("conversation_id", "")).strip()
    if not conversation_id:
        return jsonify({"ok": False, "error": "conversation_id is required."}), 400
    reset_conversation_rag(conversation_id)
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("\n🧬  Reasoning RAG  ·  Web Interface")
    print("─" * 42)
    print(f"   Index : {INDEX_PATH}")
    print(f"   URL   : http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)