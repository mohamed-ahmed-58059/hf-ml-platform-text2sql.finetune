from src.pipeline.stage import Stage
from src.train.context import TrainContext


class HoldoutStage(Stage):
    def __init__(self, val_size=1000, seed=42):
        self.val_size = val_size
        self.seed = seed

    def run(self, ctx: TrainContext) -> TrainContext:
        split = ctx.train_ds.train_test_split(
            test_size=self.val_size, shuffle=True, seed=self.seed,
        )
        ctx.train_ds, ctx.val_ds = split["train"], split["test"]
        print(f"[holdout] train={len(ctx.train_ds)} val={len(ctx.val_ds)} (seed={self.seed})")
        return ctx