import re

from pydantic import BaseModel, field_validator

from src.utils.text import slug


class TableName(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _clean(cls, v: str) -> str:
        s = slug(v)
        s = "_".join(s.split("_")[:4])           # cap at 4 words (concise)
        if s[:1].isdigit():
            s = "t_" + s                          # SQL identifiers can't start with a digit
        if len(s) < 3 or not re.fullmatch(r"[a-z][a-z0-9_]*", s):
            raise ValueError(f"unusable table name: {v!r}")
        return s
