from src.filesystem.jsonl_checkpoint import JsonlCheckpoint
from src.generation.batch_runner import BatchRunner
from src.model.table import Table
from src.model.table_name import TableName
from src.pipeline.stage import Stage
from src.utils.dedupe import deduplicate_names
from src.wikisql.context import PrepContext


class TableRenameStage(Stage):
    SYSTEM = (
        "You name database tables concisely. Given a table's title, columns, and a "
        "sample row, reply with ONE short snake_case name of at most 4 words "
        "(lowercase words joined by underscores). No quotes, no explanation."
    )

    def __init__(self, runner: BatchRunner):
        self.runner = runner

    def run(self, ctx: PrepContext) -> PrepContext:
        ckpt = JsonlCheckpoint(str(ctx.split.names_path))
        done = {r["table_id"] for r in ckpt.load()}
        pending = [t for t in ctx.tables if t.id not in done]
        print(f"[{ctx.split.name}] {len(done)} named, {len(pending)} to name")

        failed = self.runner.run(
            pending,
            messages_of=self._messages,
            schema=TableName,
            on_success=lambda t, r: ckpt.append({"table_id": t.id, "name": r.name}),
        )
        if failed:
            print(f"[{ctx.split.name}] {len(failed)} unnamed — re-run (vLLM up) to retry")

        ctx.names = deduplicate_names(ckpt.load())
        return ctx

    def _messages(self, table: Table) -> list[dict]:
        title = table.page_title or table.section_title or table.caption or "(untitled)"
        cols = ", ".join(table.header)
        sample = table.rows[0] if table.rows else []
        user = f"Title: {title}\nColumns: {cols}\nSample row: {sample}"
        return [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": user},
        ]
