#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${SCRIPT_DIR}/run_compare.sh" "$@"

echo
echo "Finished. Press Enter to close..."
read -r
