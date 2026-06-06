# Scorecard

Execution accuracy on the held-out eval set. `exact` = result-set equality vs gold;
`f1` = row-level F1 (partial credit). `$/1k` = USD per 1,000 queries.

## WikiSQL — dev (n=8263)

| Model | Kind | exact | F1 | $/1k | Notes |
|---|---|---|---|---|---|
| Llama-3.1-8B (base) | local base, 0-shot completion | 0.222 | 0.223 | ~0 | floor; explicit-instruction prompt (see note) |
| Hermes-3-Llama-3.1-8B | local instruct, 0-shot | 0.668 | 0.670 | ~0 | served via vLLM, chat + structured output |
| GPT-5.4-nano | frontier-cheap, 0-shot | 0.721 | 0.725 | 0.083 | reasoning_effort=none; $0.68 for full dev |
| **Llama-3.1-8B + QLoRA (ckpt-5800)** | **local finetuned, completion** | **0.9148** | **0.9155** | **~0** | **FINAL MODEL — lowest eval loss; r=16, max_len=256** |
