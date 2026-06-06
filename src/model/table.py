from dataclasses import dataclass, field


@dataclass
class Table:
    id: str
    header: list[str]
    types: list[str]
    rows: list[list] = field(default_factory=list)
    page_title: str = ""
    section_title: str = ""
    caption: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Table":
        return cls(
            id=d["id"],
            header=d["header"],
            types=d["types"],
            rows=d.get("rows", []),
            page_title=d.get("page_title", ""),
            section_title=d.get("section_title", ""),
            caption=d.get("caption", ""),
        )

    @property
    def db_name(self) -> str:
        return "table_" + self.id.replace("-", "_")
