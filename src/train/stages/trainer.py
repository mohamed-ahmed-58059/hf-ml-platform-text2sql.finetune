from trl import SFTTrainer

from src.pipeline.stage import Stage
from src.train.context import TrainContext


class TrainerStage(Stage):
    def run(self, ctx: TrainContext) -> TrainContext:
        ctx.trainer = SFTTrainer(
            model=ctx.model,
            args=ctx.args,
            train_dataset=ctx.train_ds,
            eval_dataset=ctx.val_ds,
            peft_config=ctx.lora,
            processing_class=ctx.tokenizer,
        )
        print(f"[trainer] ready: train={len(ctx.train_ds)} val={len(ctx.val_ds)}")
        return ctx
