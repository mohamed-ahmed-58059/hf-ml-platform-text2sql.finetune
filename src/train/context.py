from dataclasses import dataclass
from typing import Any, Optional

from src.pipeline.context import Context


@dataclass
class TrainContext(Context):
    train_ds: Optional[Any] = None     # datasets.Dataset (prompt/completion)
    val_ds: Optional[Any] = None       # datasets.Dataset (held out from train)
    tokenizer: Optional[Any] = None    # transformers tokenizer
    model: Optional[Any] = None        # quantized base model
    lora: Optional[Any] = None         # peft.LoraConfig
    args: Optional[Any] = None         # trl.SFTConfig
    trainer: Optional[Any] = None      # trl.SFTTrainer
