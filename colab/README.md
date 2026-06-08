# Run the fine-tune on a free Colab T4

Drive a Colab **T4 GPU** from your terminal (or a local Claude Code) using the
[Google Colab CLI](https://github.com/googlecolab/google-colab-cli) to run this
repo's QLoRA text2sql fine-tune — no browser notebook, no local GPU.

## Why this runs on *your* machine, not the cloud agent

The Colab CLI authenticates with a **one-time browser-based Google login (OAuth)**.
That needs a real browser to click "Allow", so it has to run somewhere you can do
that — your laptop. A headless cloud sandbox (where the agent that wrote these
files lives) has no browser and no access to your Google account, so it gets as
far as `colab new` and then hits:

```
Error: could not locate runnable browser
```

After the first login the token is cached and everything else is non-interactive.
The T4 also bills against *your* Colab plan's compute (the free tier includes T4,
with session-time and availability limits).

## One-time setup

```bash
uv tool install google-colab-cli      # or: pip install google-colab-cli  (needs Python 3.13+)
cp .env.example .env                   # then set HF_TOKEN (gated Llama-3.1 access)
```

## Run it

```bash
./colab/drive.sh smoke     # ≈ minutes — validates the whole pipeline on the T4 first
./colab/drive.sh full      # the real run (also set SMOKE = False in colab/run_on_vm.py)
```

`drive.sh` provisions a T4, uploads this repo (+ your `.env`) as a tarball,
launches `train.py` **in the background on the VM**, tails progress, and stops the
VM at the end. Detach from the tail with Ctrl-C any time — training keeps running.
Re-attach with:

```bash
echo "tail -n 40 /content/train.log" | colab console -s text2sql
```

## What runs on the VM

`colab/run_on_vm.py` (sent via `colab exec -f`) unpacks the repo, installs
`colab/requirements-train.txt` (training deps only — torch ships with the T4), and
launches `train.py`. It sets T4-safe overrides through `T2S_*` env vars that
`train.py` now reads:

| env var | smoke | full | effect |
|---|---|---|---|
| `T2S_SMOKE` | `true` | `false` | tiny subset + 10 steps for a fast correctness check |
| `T2S_RESUME_FROM` | `""` | `""` | fresh run (the repo's local checkpoint path doesn't exist on the VM) |
| `T2S_PUSH_TO_HUB` | `false` | `true` | full run streams checkpoints to the HF Hub every save |
| `T2S_LOG_TO_WANDB` | `false` | `false` | no W&B key needed on the VM |

The model code already adapts to the T4 (Turing, compute capability 7.5): it
auto-selects **fp16** compute instead of bf16 and uses default attention (no
FlashAttention-2 dependency). The 8B 4-bit QLoRA at batch 8 / seq 256 (~12 GB)
fits the T4's 16 GB.

## Free-tier reality + resuming

A full 1-epoch fine-tune of an 8B model on a *free* T4 is slow and likely **won't
finish in a single session** (free runtimes are time-limited and can disconnect).
That's why FULL mode pushes a checkpoint to the HF Hub on every save — you lose
nothing on a disconnect. To continue:

1. Pull the latest `checkpoint-N` (from the Hub or `colab download`).
2. Re-run with `T2S_RESUME_FROM=out/<run>/checkpoint-N` — it continues at the same
   step, LR, and data position (optimizer/scheduler state is in the checkpoint).

For an uninterrupted full run, a paid Colab tier (L4/A100) or the `--gpu A100`
flag finishes far faster.

## Local Claude Code can drive this for you

The Colab CLI ships a `COLAB_SKILL.md` skill file. If you run **Claude Code on
your own machine**, it can operate the CLI end-to-end (provision → exec → log →
download → stop) — point it at this folder. See the skill with `colab readme`.

## Files

| file | runs where | purpose |
|---|---|---|
| `drive.sh` | your machine | provision T4, upload repo, launch, tail, download, stop |
| `run_on_vm.py` | Colab VM | unpack repo, install deps, launch `train.py` in background |
| `requirements-train.txt` | Colab VM | training-only deps (no vLLM/serving deps) |
