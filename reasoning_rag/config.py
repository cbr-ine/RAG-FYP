"""配置文件"""
import os
from env_utils import load_project_env

load_project_env(override=True)

class Config:
    # 模型配置
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    RANDOM_SEED = 42

    # 检索配置
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.42
    KEYWORD_FALLBACK_TOP_K = 8
    SHORT_QUERY_KEYWORD_THRESHOLD = 0.25

    # 查询分解配置
    MAX_SUBQUERIES = 4
    COMPLEXITY_THRESHOLD = 0.2

    # 多跳检索配置
    MAX_HOPS = 3

    # 答案生成配置
    MAX_EVIDENCE_LENGTH = 2000

    # 证据质量控制（新增）
    # 低于此值时拒绝生成答案，提示用户重建索引
    MIN_EVIDENCE_SIMILARITY = 0.50

    # 评估配置
    TEST_SAMPLE_SIZE = 50
    DEMO_SAMPLE_SIZE = 3
    EXPERIMENT_OUTPUT_DIR = "./experiment_outputs"