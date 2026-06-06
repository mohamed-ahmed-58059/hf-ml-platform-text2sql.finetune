"""Prepare the WikiSQL dataset and push it to HuggingFace.

Runs a per-split pipeline of stages, collecting one PrepContext per split:
  Load -> Name -> Build -> Filter   (push runs once, across all splits)

vLLM must be up (serve_hermes.sh) for the naming stage on a fresh run.
Set PUSH = True (and put HF_TOKEN in .env) to publish.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from src.generation.batch_runner import BatchRunner
from src.generation.llm_client import LLMClient
from src.utils.hf_dataset import push_to_hf
from src.model.split import Split
from src.wikisql.context import PrepContext
from src.pipeline.pipeline import Pipeline
from src.wikisql.stages.assemble_examples import AssembleExamplesStage
from src.wikisql.stages.drop_invalid_examples import DropInvalidExamplesStage
from src.wikisql.stages.ingestion import IngestionStage
from src.wikisql.stages.build_database import BuildDatabaseStage
from src.wikisql.stages.table_rename import TableRenameStage
from src.wikisql.stages.column_normalize import ColumnNormalizeStage


SPLITS = ["dev", "test", "train"]
CONCURRENCY = 16
MAX_RETRIES = 5
MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"
BASE_URL = "http://localhost:8000/v1"

HF_REPO = "mohamed-ahmed-58059/wikisql-text2sql"
PUSH = True
OVERWRITE = True
PRIVATE = True

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "wikisql"


def build_pipeline() -> Pipeline:
    client = LLMClient(BASE_URL, MODEL)
    runner = BatchRunner(client, max_retries=MAX_RETRIES, concurrency=CONCURRENCY)
    return Pipeline([
        IngestionStage(),
        TableRenameStage(runner),
        ColumnNormalizeStage(),
        AssembleExamplesStage(),
        BuildDatabaseStage(),
        DropInvalidExamplesStage(),
    ])


def main():
    load_dotenv()
    pipeline = build_pipeline()
    contexts = [
        pipeline.run(PrepContext(split=Split.make(DATA, split_name)))
        for split_name in SPLITS
    ]

    for ctx in contexts:
        print(f"[{ctx.split.name}] {len(ctx.examples)} examples ready")

    if PUSH:
        token = os.environ["HF_TOKEN"]
        push_to_hf(contexts, HF_REPO, token, overwrite=OVERWRITE, private=PRIVATE)


if __name__ == "__main__":
    main()
