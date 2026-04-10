#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/reasoning_rag"

EVAL_SIZE="${EVAL_SIZE:-50}"
SEED="${SEED:-42}"

cd "${PROJECT_DIR}"
python main.py --mode compare --eval-size "${EVAL_SIZE}" --seed "${SEED}" "$@"
