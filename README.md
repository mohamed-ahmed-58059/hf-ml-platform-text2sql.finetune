# text2sql.finetune

Fine-tuning a small open LLM (QLoRA) for **text-to-SQL** — chasing frontier-model accuracy
at a fraction of the per-query cost. Built on **WikiSQL**, with a clean Stage/Pipeline
architecture for data prep, training, and execution-accuracy evaluation.

**🤗 Model:** [Llama-3.1-8B-text2sql-wikisql](https://huggingface.co/mohamed-ahmed-58059/Llama-3.1-8B-text2sql-wikisql)
 · **Dataset:** [wikisql-text2sql](https://huggingface.co/datasets/mohamed-ahmed-58059/wikisql-text2sql)

## Headline result

A QLoRA fine-tune of **Llama-3.1-8B** beats the frontier-cheap baseline on WikiSQL dev
execution accuracy — at effectively zero per-query cost (local serving).

| Model | exact | $/1k queries |
|---|---|---|
| Llama-3.1-8B (base, zero-shot) | 0.222 | ~0 |
| Hermes-3-8B (instruct, zero-shot) | 0.668 | ~0 |
| GPT-5.4-nano (frontier-cheap, zero-shot) | 0.721 | $0.083 |
| **Llama-3.1-8B + QLoRA** | **0.915** | **~0** |

`exact` = execution accuracy (run the predicted SQL against the real DB, compare result
sets to the gold query). Full numbers in [`SCORECARD.md`](SCORECARD.md).

## Approach

- **Metric = execution accuracy.** We don't string-match SQL; we *run* it and compare the
  returned rows. Cosmetic differences (quoting, casing, column aliases, trailing `;`) don't
  count against the model — only wrong answers do.
- **Clean eval DBs.** Each WikiSQL split is migrated into a single SQLite database with real
  table/column names, case-insensitive (`COLLATE U_NOCASE`) text columns, and numeric columns
  cast to `REAL`, so both gold and predicted real-name SQL execute directly.
- **QLoRA, 4-bit.** bitsandbytes NF4 + LoRA (r=16) via TRL's `SFTTrainer`, completion-only
  loss, `max_seq_len=256`. Fits a 12 GB consumer GPU.
- **Honest splits.** Validation is carved from `train` (never `dev`); `dev` is the held-out
  number we report; `test` stays sealed.

## Architecture

Everything is a `Stage` (`run(ctx) -> ctx`) threaded by a `Pipeline`, with a typed context
per phase. Three pipelines:

- **Prepare** (`prepare_wikisql.py`) — ingest → name tables (LLM) → normalize columns →
  assemble examples → build the SQLite DB → drop invalid examples → push to HuggingFace.
- **Train** (`train.py`) — auth → load dataset → val holdout → load 4-bit model →
  length-filter → LoRA → SFT config → trainer → train. Has a `SMOKE` knob and
  `RESUME_FROM` for exact checkpoint resume.
- **Evaluate** (`evaluate.py`) — fetch split + DB from HF → predict (threaded) → score
  (execution accuracy + row-level F1). One `BASELINE` switch picks the model under test.

## Setup

```bash
uv sync                      # install deps (Python 3.12, torch +cu130)
cp .env.example .env         # then add HF_TOKEN, OPENAI_API_KEY, WANDB_API_KEY
```

`.env` (gitignored) holds the secrets:

```
HF_TOKEN=...           # HuggingFace (gated Llama access + dataset push)
OPENAI_API_KEY=...     # only for the GPT baseline
WANDB_API_KEY=...      # only if LOG_TO_WANDB
```

## Usage

```bash
# 1. Data
./download_data.sh                 # WikiSQL -> data/wikisql/
python prepare_wikisql.py          # build examples + DBs, push to HF

# 2. Serve a model on vLLM (separate terminal; one model holds the 12 GB GPU)
./serve_hermes.sh                  # instruct baseline
./serve_llama_base.sh              # base baseline
./serve_finetuned.sh out/<run>/checkpoint-<n>   # the fine-tuned adapter (base + LoRA)

# 3. Train
python train.py                    # QLoRA SFT; checkpoints -> out/<run>/, optional hub push

# 4. Evaluate (set BASELINE in evaluate.py: llama-base | hermes | gpt-nano | finetuned)
python evaluate.py                 # full-dev execution accuracy -> scorecard
```

## Layout

```
src/
  pipeline/    Stage / Pipeline / Context base
  model/       Example, Split, Table, Question, ... (the canonical record)
  wikisql/     WikiSQL-specific prep stages + SQL decoder
  train/       TrainContext + SFT/QLoRA stages
  eval/        EvalContext + fetch/predict/score stages, scoring, SqlGenerator
  generation/  vLLM client + threaded batch runner
  utils/       sqlite (custom collation), hf dataset push/fetch, text, dedupe
prepare_wikisql.py · train.py · evaluate.py · serve_*.sh
```

## Hardware

Developed on an RTX 3080 Ti (12 GB). 8B QLoRA trains at batch 8 / seq 256 (~11.9 GB),
served 4-bit via vLLM. The fine-tune's win is accuracy *and* cost: it runs locally at
~zero marginal cost vs paying per token for a frontier API.
