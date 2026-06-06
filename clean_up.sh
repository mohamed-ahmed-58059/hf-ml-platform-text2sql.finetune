#!/usr/bin/env bash
# Remove generated artifacts, but KEEP the expensive LLM-generated table names
# (data/wikisql/table_names/) so naming isn't recomputed. After this,
# ./download_data.sh re-fetches the raw data + dbs. Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Downloaded WikiSQL data + dbs (re-fetchable) — preserve the table_names/ dir.
if [ -d "$ROOT/data/wikisql" ]; then
  find "$ROOT/data/wikisql" -mindepth 1 -maxdepth 1 -not -name table_names \
    -exec rm -rf {} + 2>/dev/null || true
fi

# vLLM serve log + tooling caches.
rm -f "$ROOT/vllm.log"
rm -rf "$ROOT/.ruff_cache"
find "$ROOT" -type d -name "__pycache__" -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

echo "Cleaned (kept table_names). Re-run ./download_data.sh to re-fetch data."
