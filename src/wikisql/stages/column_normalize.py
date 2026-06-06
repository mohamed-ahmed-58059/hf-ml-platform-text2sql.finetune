import re

from src.wikisql.context import PrepContext
from src.pipeline.stage import Stage


class ColumnNormalizeStage(Stage):
    def run(self, ctx: PrepContext) -> PrepContext:
        ctx.column_names = {t.id: self._normalize(t.header) for t in ctx.tables}
        print(f"[{ctx.split.name}] normalized columns for {len(ctx.column_names)} tables")
        return ctx

    def _normalize(self, header: list[str]) -> list[str]:
        out: list[str] = []
        used: set[str] = set()
        for h in header:
            base = self._slug(h)
            name, i = base, 1
            while name in used:
                i += 1
                name = f"{base}_{i}"
            used.add(name)
            out.append(name)
        return out

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_") or "col"
