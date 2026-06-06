from src.pipeline.stage import Stage
from src.train.context import TrainContext


class LengthFilterStage(Stage):
    def __init__(self, max_seq_len=256):
        self.max_seq_len = max_seq_len

    def run(self, ctx: TrainContext) -> TrainContext:
        ctx.train_ds = self._filter(ctx.train_ds, ctx.tokenizer, "train")
        ctx.val_ds = self._filter(ctx.val_ds, ctx.tokenizer, "val")
        return ctx

    def _filter(self, ds, tokenizer, name):
        before = len(ds)
        kept = ds.filter(lambda r: self._fits(r, tokenizer))
        dropped = before - len(kept)
        print(f"[length] {name}: kept {len(kept)}/{before} (dropped {dropped} > {self.max_seq_len} tok)")
        return kept

    def _fits(self, row, tokenizer) -> bool:
        return len(tokenizer(row["prompt"] + row["completion"]).input_ids) <= self.max_seq_len
