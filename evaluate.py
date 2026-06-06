import os

from dotenv import load_dotenv

from src.eval.context import EvalContext
from src.eval.sql_generator import SqlGenerator
from src.eval.stages.fetch import FetchStage
from src.eval.stages.predict import PredictStage
from src.eval.stages.score import ScoreStage
from src.pipeline.pipeline import Pipeline


HF_REPO = "mohamed-ahmed-58059/wikisql-text2sql"
SPLITS = ["dev"]
LIMIT = None

BASELINE = "finetuned"
CONCURRENCY = {"gpt-nano": 8, "llama-base": 64, "hermes": 16, "finetuned": 64}

DEV_SIZE = 8263
PRICING = {"gpt-nano": (0.20, 1.25)}  # $ / 1M tokens (in, out) -- frontier API models only


def report(generator, prices=None):
    u = generator.usage
    if u["failures"]:
        print(f"\n!! {u['failures']} prediction(s) FAILED after retries — "
              f"metric is unreliable, do not trust this number")
    if not prices or not u["calls"]:
        return
    price_in, price_out = prices
    cost = (u["prompt_tokens"] * price_in + u["completion_tokens"] * price_out) / 1e6
    per = cost / u["calls"]
    print(f"\ntokens: in={u['prompt_tokens']} (cached={u['cached_tokens']}) "
          f"out={u['completion_tokens']} (reasoning={u['reasoning_tokens']})")
    print(f"cost: ${cost:.4f} over {u['calls']} calls  (${per * 1000:.4f}/1k examples)")
    print(f"projected full dev ({DEV_SIZE}): ${per * DEV_SIZE:.2f}")


def make_generator(name) -> SqlGenerator:
    if name == "gpt-nano":
        return SqlGenerator(
            base_url=None,
            model="gpt-5.4-nano-2026-03-17",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=None,
            token_param="max_completion_tokens",
            max_tokens=2000,
            reasoning_effort="none",
        )
    if name == "llama-base":
        return SqlGenerator(
            "http://localhost:8000/v1", "meta-llama/Llama-3.1-8B", mode="completion",
        )
    if name == "finetuned":
        # "text2sql" = the vLLM --lora-modules name from serve_finetuned.sh
        return SqlGenerator(
            "http://localhost:8000/v1", "text2sql", mode="completion",
        )
    return SqlGenerator("http://localhost:8000/v1", "NousResearch/Hermes-3-Llama-3.1-8B")


def build_pipeline(token) -> Pipeline:
    return Pipeline([
        FetchStage(HF_REPO, token, limit=LIMIT),
        PredictStage(concurrency=CONCURRENCY[BASELINE]),
        ScoreStage(),
    ])


def main():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    generator = make_generator(BASELINE)
    pipeline = build_pipeline(token)
    for split in SPLITS:
        pipeline.run(EvalContext(split=split, generator=generator))
    report(generator, PRICING.get(BASELINE))


if __name__ == "__main__":
    main()
