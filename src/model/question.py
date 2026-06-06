from dataclasses import dataclass


@dataclass
class Question:
    table_id: str
    text: str
    sql: dict
    
    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(table_id=d["table_id"], text=d["question"], sql=d["sql"])
