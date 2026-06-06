import json

from src.model.question import Question
from src.model.table import Table
from src.wikisql.context import PrepContext
from src.pipeline.stage import Stage


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class IngestionStage(Stage):
    def run(self, ctx: PrepContext) -> PrepContext:
        ctx.tables = [Table.from_dict(d) for d in _load_jsonl(ctx.split.tables_path)]
        ctx.questions = [Question.from_dict(d) for d in _load_jsonl(ctx.split.questions_path)]
        print(f"[{ctx.split.name}] loaded {len(ctx.tables)} tables, {len(ctx.questions)} questions")
        return ctx
