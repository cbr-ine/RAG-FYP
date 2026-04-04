#!/usr/bin/env python3
"""
Reasoning RAG — Web Chat Interface
────────────────────────────────────
Run:   python app.py
Open:  http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import logging
import traceback
import os

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


def get_rag():
    global _rag
    if _rag is None:
        logger.info("Initialising Reasoning RAG system …")

        from reasoning_rag import ReasoningRAG
        from config import Config
        _rag = ReasoningRAG(Config())

        # ── 加载已有索引 ────────────────────────────────────────────
        if os.path.exists(INDEX_PATH):
            logger.info(f"Loading index from {INDEX_PATH} …")
            _rag.load_index(INDEX_PATH)
            logger.info(
                f"Index loaded ✓  "
                f"({_rag.vector_store.index.ntotal} vectors, "
                f"{len(_rag.vector_store.passages)} passages)"
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
        # ────────────────────────────────────────────────────────────

        logger.info("Reasoning RAG ready ✓")
    return _rag


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


def to_json(result, question: str) -> dict:
    analysis      = _get(result, "analysis",  default={}) or {}
    retrieval     = _get(result, "retrieval", default={}) or {}
    answer_r      = _get(result, "answer",    default={}) or {}
    integrated_ev = _get(result, "evidence",  default={}) or {}

    ev_list   = _get(integrated_ev, "evidence", default=[]) or []
    ev_stats  = _get(integrated_ev, "stats",    default={}) or {}
    ret_stats = _get(retrieval, "retrieval_stats", default={}) or {}
    path      = _get(answer_r, "reasoning_path", default=[]) or []

    return {
        "question":   question,

        "answer":     _get(answer_r, "answer",     default=""),
        "confidence": float(_get(answer_r, "confidence", default=0.0)),

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

        if not question:
            return jsonify({"ok": False, "error": "Question cannot be empty."}), 400

        logger.info(f"Received query: {question!r}")
        rag = get_rag()

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


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🧬  Reasoning RAG  ·  Web Interface")
    print("─" * 42)
    print(f"   Index : {INDEX_PATH}")
    print(f"   URL   : http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)