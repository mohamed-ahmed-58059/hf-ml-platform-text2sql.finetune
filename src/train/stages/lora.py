from peft import LoraConfig

from src.pipeline.stage import Stage
from src.train.context import TrainContext

ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]


class LoraStage(Stage):
    def __init__(self, r=32, alpha=None, dropout=0.1, target_modules=None):
        self.r = r
        self.alpha = alpha if alpha is not None else r * 2
        self.dropout = dropout
        self.target_modules = target_modules or (ATTENTION_LAYERS + MLP_LAYERS)

    def run(self, ctx: TrainContext) -> TrainContext:
        ctx.lora = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )
        print(f"[lora] r={self.r} alpha={self.alpha} targets={len(self.target_modules)}")
        return ctx
