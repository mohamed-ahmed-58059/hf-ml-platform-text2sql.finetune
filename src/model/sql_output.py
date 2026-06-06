from pydantic import BaseModel


class SqlOutput(BaseModel):
    sql: str
