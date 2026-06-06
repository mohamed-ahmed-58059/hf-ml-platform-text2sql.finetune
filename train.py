import os
from datetime import datetime

from dotenv import load_dotenv

from src.pipeline.pipeline import Pipeline
from src.train.context import TrainContext
from src.train.stages.auth import AuthStage
from src.train.stages.dataset import DatasetStage
from src.train.stages.holdout import HoldoutStage
from src.train.stages.length_filter import LengthFilterStage
from src.train.stages.lora import LoraStage
from src.train.stages.model import ModelStage
from src.train.stages.train import TrainStage
from src.train.stages.train_args import TrainArgsStage
from src.train.stages.trainer import TrainerStage


HF_REPO = "mohamed-ahmed-58059/wikisql-text2sql"
HF_USER = "mohamed-ahmed-58059"
PROJECT = "text2sql"

BASE_MODEL = "meta-llama/Llama-3.1-8B"   # the one we have access to (3.2 is a separate gate)

EPOCHS = 1
BATCH_SIZE = 8     # fits the 12GB 3080 Ti at seq 256 (smoke-verified, no OOM)
GRAD_ACCUM = 1     # effective batch = 8
MAX_SEQ_LEN = 256
LEARNING_RATE = 1e-4
LORA_R = 16
LORA_ALPHA = 32
VAL_SIZE = 1000
SEED = 42
SAVE_STEPS = 200
LOG_STEPS = 10

PUSH_TO_HUB = True
PRIVATE = True
LOG_TO_WANDB = True

# Resume an interrupted run: set to a LOCAL checkpoint dir (has optimizer/scheduler state).
# Continues at the same step/LR/data position. Leave None for a fresh run.
RESUME_FROM = "out/text2sql-20260604-063837/checkpoint-2600"

# Smoke mode: tiny subset + few steps for a fast end-to-end correctness check.
SMOKE = False
TRAIN_SPLIT = "train"
MAX_STEPS = -1
if SMOKE:
    TRAIN_SPLIT = "train[:200]"
    VAL_SIZE = 20
    MAX_STEPS = 10
    SAVE_STEPS = 5
    LOG_STEPS = 1


def build_pipeline(token, run_name, hub_model_id) -> Pipeline:
    return Pipeline([
        AuthStage(wandb_project=PROJECT if LOG_TO_WANDB else None, run_name=run_name),
        DatasetStage(HF_REPO, split=TRAIN_SPLIT, token=token),
        HoldoutStage(val_size=VAL_SIZE, seed=SEED),
        ModelStage(BASE_MODEL),
        LengthFilterStage(max_seq_len=MAX_SEQ_LEN),
        LoraStage(r=LORA_R, alpha=LORA_ALPHA),
        TrainArgsStage(
            output_dir=f"out/{run_name}", run_name=run_name, hub_model_id=hub_model_id,
            epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM,
            max_seq_len=MAX_SEQ_LEN, learning_rate=LEARNING_RATE,
            save_steps=SAVE_STEPS, log_steps=LOG_STEPS, max_steps=MAX_STEPS,
            push_to_hub=PUSH_TO_HUB, private=PRIVATE,
            report_to="wandb" if LOG_TO_WANDB else None,
        ),
        TrainerStage(),
        TrainStage(push_to_hub=PUSH_TO_HUB, log_to_wandb=LOG_TO_WANDB,
                   resume_from_checkpoint=RESUME_FROM),
    ])


def main():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    # Resuming: reuse the original run's folder so checkpoints/step numbering continue in place.
    run_name = (RESUME_FROM.split("/")[1] if RESUME_FROM
                else f"{PROJECT}-{datetime.now():%Y%m%d-%H%M%S}")
    hub_model_id = f"{HF_USER}/{run_name}" if PUSH_TO_HUB else None
    build_pipeline(token, run_name, hub_model_id).run(TrainContext())


if __name__ == "__main__":
    main()
