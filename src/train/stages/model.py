import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.pipeline.stage import Stage
from src.train.context import TrainContext


class ModelStage(Stage):
    def __init__(self, base_model, quant_4bit=True):
        self.base_model = base_model
        self.quant_4bit = quant_4bit

    def run(self, ctx: TrainContext) -> TrainContext:
        compute_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if self.quant_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quant = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype,
            )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load non-quantized params (and the LoRA adapters built on them) in the
        # same dtype we compute in. Llama-3.1's config default is bfloat16; on a T4
        # (compute capability < 8) training runs fp16=True with an fp16 GradScaler,
        # which can't unscale bf16 grads ("_amp_foreach_non_finite_check_and_unscale_
        # not implemented for BFloat16"). Forcing float16 here keeps dtypes consistent.
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model, quantization_config=quant, device_map="auto",
            torch_dtype=compute_dtype,
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        ctx.tokenizer, ctx.model = tokenizer, model
        print(f"[model] {self.base_model} loaded ({model.get_memory_footprint() / 1e6:.0f} MB)")
        return ctx
