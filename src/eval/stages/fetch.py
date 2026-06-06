from src.eval.context import EvalContext
from src.pipeline.stage import Stage
from src.utils.hf_dataset import fetch_from_hf


class FetchStage(Stage):
    def __init__(self, repo_id, token=None, limit=None):
        self.repo_id = repo_id
        self.token = token
        self.limit = limit

    def run(self, ctx: EvalContext) -> EvalContext:
        examples, db_path = fetch_from_hf(self.repo_id, ctx.split, self.token)
        if self.limit:
            examples = examples[: self.limit]
        ctx.examples = examples
        ctx.db_path = db_path
        print(f"[{ctx.split}] fetched {len(examples)} examples + db")
        return ctx
