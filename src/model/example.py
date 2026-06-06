from dataclasses import asdict, dataclass


PROMPT_TEMPLATE = (
    "Construct a SQL statement using the following schema and question:\n"
    "Schema:\n{schema}\n\n"
    "Question: {question}\n"
    "SQL:\n"
)


@dataclass
class Example:
    question: str
    table_name: str
    columns: list[str]
    column_types: list[str]
    gold_sql: str
    db_id: str
    source: str

    @staticmethod
    def _sql_type(t: str) -> str:
        return "REAL" if t == "real" else "TEXT"

    def schema_text(self, fmt: str = "ddl") -> str:
        cols = ",\n".join(
            f'  "{c}" {self._sql_type(t)}' for c, t in zip(self.columns, self.column_types)
        )
        return f"CREATE TABLE {self.table_name} (\n{cols}\n)"

    def create_prompt(self, fmt: str = "ddl") -> str:
        return PROMPT_TEMPLATE.format(schema=self.schema_text(fmt), question=self.question)

    def get_data_from_db(self, con) -> list:
        return con.execute(self.gold_sql).fetchall()

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "Example":
        return cls(**row)
