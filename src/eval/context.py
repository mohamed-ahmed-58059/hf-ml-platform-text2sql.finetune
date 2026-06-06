from dataclasses import dataclass
from typing import Optional

from src.eval.sql_generator import SqlGenerator
from src.model.example import Example
from src.pipeline.context import Context


@dataclass
class EvalContext(Context):
    split: str
    generator: SqlGenerator
    examples: Optional[list[Example]] = None
    db_path: Optional[str] = None
    predictions: Optional[list[Optional[str]]] = None
    metrics: Optional[dict] = None
