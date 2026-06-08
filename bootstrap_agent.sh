#!/usr/bin/env bash
#
# Teleport Claude Code onto any Linux GPU box (Colab terminal, Lightning AI,
# RunPod, a cloud VM, ...). Run this in that machine's terminal, then `claude`.
# The agent then lives ON the GPU — drive it from your phone's browser terminal.
#
# One-liner (public repo):
#   curl -fsSL https://raw.githubusercontent.com/mohamed-ahmed-58059/hf-ml-platform-text2sql.finetune/claude/colab-cli-t4-gpu-8z1v8i/bootstrap_agent.sh | bash
# Private repo: GH_TOKEN=ghp_xxx bash bootstrap_agent.sh   (or just paste this file)

set -euo pipefail

REPO_PATH="mohamed-ahmed-58059/hf-ml-platform-text2sql.finetune"
BRANCH="claude/colab-cli-t4-gpu-8z1v8i"
DIR="$HOME/text2sql.finetune"
GH_TOKEN="${GH_TOKEN:-}"

echo ">> [1/4] GPU check"
nvidia-smi -L || echo "WARNING: no GPU visible on this machine"

echo ">> [2/4] Node 20 (Claude Code needs Node 18+)"
if ! command -v node >/dev/null 2>&1 || [ "$(node -v | sed 's/v//;s/\..*//')" -lt 18 ]; then
  if command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then SUDO=sudo; else SUDO=; fi
  curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO bash - && $SUDO apt-get install -y nodejs \
   || { curl -fsSL https://fnm.vercel.app/install | bash \
        && export PATH="$HOME/.local/share/fnm:$PATH" && eval "$(fnm env)" \
        && fnm install 20 && fnm use 20; }
fi
node -v

echo ">> [3/4] Claude Code"
npm i -g @anthropic-ai/claude-code

echo ">> [4/4] Repo"
URL="https://github.com/${REPO_PATH}"
[ -n "$GH_TOKEN" ] && URL="https://${GH_TOKEN}@github.com/${REPO_PATH}"
[ -d "$DIR" ] || git clone --branch "$BRANCH" --depth 1 "$URL" "$DIR"

cat <<EOF

================================================================
Claude Code is installed on this GPU machine. Now:

  export ANTHROPIC_API_KEY=sk-ant-...   # your Anthropic key (the agent runs on YOUR key here)
  export HF_TOKEN=hf_...                # gated Llama-3.1 access
  cd $DIR
  claude

Then tell the agent:
  "Run the smoke fine-tune (T2S_SMOKE=true), then the full run. See colab/README.md."
================================================================
EOF
