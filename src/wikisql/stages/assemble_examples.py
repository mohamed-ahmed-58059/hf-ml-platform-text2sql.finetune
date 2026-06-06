from src.model.example import Example
from src.wikisql.context import PrepContext
from src.pipeline.stage import Stage
from src.wikisql.sql_decoder import decode_sql


class AssembleExamplesStage(Stage):
    def run(self, ctx: PrepContext) -> PrepContext:
        by_id = {t.id: t for t in ctx.tables}
        examples, skipped = [], 0

        for q in ctx.questions:
            table = by_id.get(q.table_id)
            name = ctx.names.get(q.table_id) if ctx.names else None
            columns = ctx.column_names.get(q.table_id) if ctx.column_names else None
            if table is None or name is None or columns is None:
                skipped += 1
                continue
            examples.append(Example(
                question=q.text,
                table_name=name,
                columns=columns,
                column_types=table.types,
                gold_sql=decode_sql(q.sql, table, name, columns),
                db_id=ctx.split.name,
                source="wikisql",
            ))

        ctx.examples = examples
        msg = f"[{ctx.split.name}] built {len(examples)} examples"
        if skipped:
            msg += f" ({skipped} skipped: missing table/name/columns)"
        print(msg)
        return ctx
