import os
import wandb

from huggingface_hub import login

from src.pipeline.stage import Stage
from src.train.context import TrainContext


class AuthStage(Stage):
    def __init__(self, wandb_project=None, run_name=None):
        self.wandb_project = wandb_project
        self.run_name = run_name

    def run(self, ctx: TrainContext) -> TrainContext:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token)
        if self.wandb_project:
            wandb.login()
            wandb.init(project=self.wandb_project, name=self.run_name)
        print(f"[auth] hf={'ok' if token else 'no token'} wandb={self.wandb_project or 'off'}")
        return ctx
