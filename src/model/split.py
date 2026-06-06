from dataclasses import dataclass
from pathlib import Path


@dataclass
class Split:
    name: str
    questions_path: Path
    tables_path: Path
    db_path: Path
    names_path: Path

    @classmethod
    def make(cls, data_dir: Path, name: str) -> "Split":
        return cls(
            name=name,
            questions_path=data_dir / f"{name}.jsonl",
            tables_path=data_dir / f"{name}.tables.jsonl",
            db_path=data_dir / f"{name}.db",
            names_path=data_dir / "table_names" / f"{name}.jsonl",
        )
