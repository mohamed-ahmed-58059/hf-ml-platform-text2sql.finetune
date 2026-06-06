from datasets import load_dataset

from src.model.example import Example
from src.pipeline.stage import Stage
from src.train.context import TrainContext


class DatasetStage(Stage):
    def __init__(self, repo_id, split="train", token=None):
        self.repo_id = repo_id
        self.split = split
        self.token = token

    def run(self, ctx: TrainContext) -> TrainContext:
        ds = load_dataset(self.repo_id, split=self.split, token=self.token)
        ctx.train_ds = ds.map(self._to_pair, remove_columns=ds.column_names)
        print(f"[dataset] {len(ctx.train_ds)} train rows from {self.repo_id}:{self.split}")
        return ctx

    @staticmethod
    def _to_pair(row) -> dict:
        ex = Example.from_row(row)
        return {"prompt": ex.create_prompt(), "completion": ex.gold_sql}
