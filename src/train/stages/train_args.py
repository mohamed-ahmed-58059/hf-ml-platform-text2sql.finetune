import torch
from trl import SFTConfig

from src.pipeline.stage import Stage
from src.train.context import TrainContext


class TrainArgsStage(Stage):
    def __init__(self, output_dir, run_name, hub_model_id=None, epochs=1, batch_size=8,
                 grad_accum=1, max_seq_len=256, learning_rate=1e-4, save_steps=100,
                 log_steps=5, max_steps=-1, push_to_hub=False, private=True, report_to=None):
        self.output_dir = output_dir
        self.run_name = run_name
        self.hub_model_id = hub_model_id
        self.epochs = epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.push_to_hub = push_to_hub
        self.private = private
        self.report_to = report_to

    def run(self, ctx: TrainContext) -> TrainContext:
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        ctx.args = SFTConfig(
            output_dir=self.output_dir,
            run_name=self.run_name,
            num_train_epochs=self.epochs,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.grad_accum,
            max_length=self.max_seq_len,
            completion_only_loss=True,
            learning_rate=self.learning_rate,
            optim="paged_adamw_32bit",
            weight_decay=0.001,
            max_grad_norm=0.3,
            warmup_ratio=0.01,
            lr_scheduler_type="cosine",
            bf16=use_bf16,
            fp16=not use_bf16,
            logging_steps=self.log_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=10,
            eval_strategy="steps",
            eval_steps=self.save_steps,
            report_to=self.report_to or "none",
            push_to_hub=self.push_to_hub,
            hub_model_id=self.hub_model_id,
            hub_private_repo=self.private,
            hub_strategy="every_save",
        )
        print(f"[args] epochs={self.epochs} bs={self.batch_size} lr={self.learning_rate} "
              f"max_len={self.max_seq_len} bf16={use_bf16}")
        return ctx
