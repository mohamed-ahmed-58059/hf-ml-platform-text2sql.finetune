from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.eval.context import EvalContext
from src.pipeline.stage import Stage


class PredictStage(Stage):
    def __init__(self, concurrency=16):
        self.concurrency = concurrency

    def run(self, ctx: EvalContext) -> EvalContext:
        examples = ctx.examples
        preds = [None] * len(examples)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(ctx.generator.generate, ex): i for i, ex in enumerate(examples)}
            for fut in tqdm(as_completed(futures), total=len(examples), desc=f"[{ctx.split}] predict"):
                preds[futures[fut]] = fut.result()
        ctx.predictions = preds
        return ctx
