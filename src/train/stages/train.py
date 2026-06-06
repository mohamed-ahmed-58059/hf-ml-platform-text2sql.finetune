import wandb

from src.pipeline.stage import Stage
from src.train.context import TrainContext


class TrainStage(Stage):
    def __init__(self, push_to_hub=False, log_to_wandb=False, resume_from_checkpoint=None):
        self.push_to_hub = push_to_hub
        self.log_to_wandb = log_to_wandb
        self.resume_from_checkpoint = resume_from_checkpoint

    def run(self, ctx: TrainContext) -> TrainContext:
        ctx.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        if self.push_to_hub:
            ctx.trainer.push_to_hub()
            print("[train] pushed to hub")
        if self.log_to_wandb:
            wandb.finish()
        return ctx
