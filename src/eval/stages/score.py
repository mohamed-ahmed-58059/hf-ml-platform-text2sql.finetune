import random

from src.eval.context import EvalContext
from src.eval.scoring import score
from src.pipeline.stage import Stage
from src.utils.sqlite import connect


class ScoreStage(Stage):
    def __init__(self, sample=5):
        self.sample = sample

    def run(self, ctx: EvalContext) -> EvalContext:
        con = connect(ctx.db_path)
        rows = [score(p, ex.gold_sql, con) for ex, p in zip(ctx.examples, ctx.predictions)]
        n = len(rows)
        exact = sum(r["exact"] for r in rows) / n if n else 0.0
        f1 = sum(r["f1"] for r in rows) / n if n else 0.0
        ctx.metrics = {"n": n, "exact": exact, "f1": f1}
        if self.sample and n:
            self._show(con, ctx, rows)
        con.close()
        print(f"[{ctx.split}] exact={exact:.4f} f1={f1:.4f} (n={n})")
        return ctx

    def _rows(self, con, sql, cap=8):
        if not sql:
            return "<none>"
        try:
            out = con.execute(sql).fetchall()
        except Exception as e:
            return f"<error: {e}>"
        return out[:cap] + [f"... (+{len(out) - cap} more)"] if len(out) > cap else out

    def _show(self, con, ctx, rows):
        picks = random.Random(0).sample(range(len(rows)), min(self.sample, len(rows)))
        print(f"\n[{ctx.split}] sample of {len(picks)}:")
        for i in picks:
            ex, pred = ctx.examples[i], ctx.predictions[i]
            print("-" * 70)
            print(f"Q:         {ex.question}")
            print(f"gold sql:  {ex.gold_sql}")
            print(f"pred sql:  {pred}")
            print(f"gold rows: {self._rows(con, ex.gold_sql)}")
            print(f"pred rows: {self._rows(con, pred)}")
            print(f"exact={rows[i]['exact']} f1={rows[i]['f1']:.3f}")
        print("-" * 70)
