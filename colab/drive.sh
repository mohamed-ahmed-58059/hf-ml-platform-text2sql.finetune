#!/usr/bin/env bash
#
# Drive a free Colab T4 from YOUR machine to run the QLoRA text2sql fine-tune.
#
# Why your machine and not the cloud agent: the Colab CLI authenticates with a
# one-time browser-based Google login (OAuth). That needs a real browser, so it
# has to run somewhere you can click "Allow" — i.e. your laptop. After the first
# `colab new`, the login is cached and the rest is non-interactive.
#
# Prereqs (one time):
#   uv tool install google-colab-cli      # or: pip install google-colab-cli  (Python 3.13+)
#   cp .env.example .env && edit .env      # set HF_TOKEN (gated Llama access)
#
# Usage:
#   ./colab/drive.sh smoke     # ≈ minutes: prove the pipeline works on the T4
#   ./colab/drive.sh full      # the real run (set SMOKE=False in run_on_vm.py too)
#
# Notes on the FREE tier:
#   * Free T4 sessions are time-limited and can disconnect; a full 1-epoch 8B
#     QLoRA likely won't finish in one sitting. FULL mode pushes checkpoints to
#     the HF Hub every save, so you lose nothing — resume by re-running train.py
#     with T2S_RESUME_FROM pointing at the last checkpoint.
#   * Always stop the VM when done (this script does) — idle VMs burn compute.

set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

MODE="${1:-smoke}"
SESSION="text2sql"

command -v colab >/dev/null || { echo "colab CLI not found. Run: uv tool install google-colab-cli"; exit 1; }
[ -f .env ] || { echo "No .env found. Run: cp .env.example .env  and set HF_TOKEN"; exit 1; }

echo ">> [1/6] provisioning a T4 session (first run opens a browser for Google login)"
colab new -s "$SESSION" --gpu T4
colab status -s "$SESSION"

echo ">> [2/6] packing repo (+ .env) and uploading to the VM"
TARBALL="$(mktemp -t repo-XXXX.tgz)"
# Ship the code and your .env; skip the heavy / irrelevant dirs.
tar czf "$TARBALL" \
  --exclude='.git' --exclude='out' --exclude='data' \
  --exclude='__pycache__' --exclude='.venv' --exclude='wandb' \
  src train.py pyproject.toml colab .env
colab upload "$TARBALL" /content/repo.tgz -s "$SESSION"
rm -f "$TARBALL"

echo ">> [3/6] launching training in the background on the VM (mode: $MODE)"
echo "         (edit SMOKE in colab/run_on_vm.py: smoke=True, full=False)"
colab exec -s "$SESSION" -f colab/run_on_vm.py --timeout 600

echo ">> [4/6] tailing progress. Ctrl-C to detach — training keeps running on the VM."
echo "         re-attach any time with:  echo 'tail -n 40 /content/train.log' | colab console -s $SESSION"
for _ in $(seq 1 60); do
  echo "tail -n 15 /content/train.log" | colab console -s "$SESSION" || true
  sleep 30
done

echo ">> [5/6] (when training has finished) get the adapter off the VM"
echo "   FULL mode already pushed checkpoints to the HF Hub — easiest is to pull from there."
echo "   To grab them from the VM instead (download handles files, so tar the dir first):"
echo "     echo 'tar czf /content/out.tgz -C /content/text2sql.finetune out' | colab console -s $SESSION"
echo "     colab download /content/out.tgz ./out-colab.tgz -s $SESSION"
echo "     colab log -s $SESSION -o colab/run.ipynb     # save a replayable notebook of the session"

echo ">> [6/6] stopping the VM (free the compute)"
colab stop -s "$SESSION"
echo "done."
