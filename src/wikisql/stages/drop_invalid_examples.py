from tqdm import tqdm

from src.wikisql.context import PrepContext
from src.pipeline.stage import Stage
from src.utils.sqlite import connect


class DropInvalidExamplesStage(Stage):
    def run(self, ctx: PrepContext) -> PrepContext:
        con = connect(ctx.split.db_path)
        kept, dropped = [], 0
        for e in tqdm(ctx.examples, desc=f"[{ctx.split.name}] filter", leave=False):
            try:
                con.execute(e.gold_sql)
                kept.append(e)
            except Exception:
                dropped += 1
        con.close()

        ctx.examples = kept
        print(f"[{ctx.split.name}] kept {len(kept)}, dropped {dropped} (invalid gold)")
        return ctx
