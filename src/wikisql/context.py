from dataclasses import dataclass
from typing import Optional

from src.model.example import Example
from src.model.question import Question
from src.model.split import Split
from src.model.table import Table
from src.pipeline.context import Context


@dataclass
class PrepContext(Context):
    split: Split
    tables: Optional[list[Table]] = None
    questions: Optional[list[Question]] = None
    names: Optional[dict[str, str]] = None
    column_names: Optional[dict[str, list[str]]] = None
    examples: Optional[list[Example]] = None
