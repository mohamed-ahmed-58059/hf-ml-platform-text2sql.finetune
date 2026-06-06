#!/usr/bin/env bash
# Download the WikiSQL dataset into data/wikisql/.
#
# WikiSQL's HuggingFace loader is script-based and broken on datasets >= 3.0, so we
# pull the original tarball from the Salesforce repo. It contains, per split:
#   {split}.jsonl         - questions + gold SQL (as logical forms)
#   {split}.tables.jsonl  - table schemas (+ rows)
#   {split}.db            - SQLite database for execution-accuracy evaluation
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/data/wikisql"
URL="https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

mkdir -p "$DEST"
echo "Downloading WikiSQL -> $DEST"
curl -L --fail -o "$DEST/data.tar.bz2" "$URL"

echo "Extracting..."
tar -xjf "$DEST/data.tar.bz2" -C "$DEST"

# The tarball extracts into a nested data/ dir; flatten it.
if [ -d "$DEST/data" ]; then
  mv "$DEST"/data/* "$DEST"/
  rmdir "$DEST/data"
fi
rm -f "$DEST/data.tar.bz2"

echo "Done. Contents of $DEST:"
ls -lh "$DEST"
