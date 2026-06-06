import os

from src.wikisql.context import PrepContext
from src.pipeline.stage import Stage
from src.utils.sqlite import connect


class BuildDatabaseStage(Stage):
    def run(self, ctx: PrepContext) -> PrepContext:
        db = str(ctx.split.db_path)
        tmp = db + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        con = connect(tmp)

        built = 0
        for t in ctx.tables:
            name = ctx.names.get(t.id)
            cols = ctx.column_names.get(t.id)
            if not name or not cols:
                continue
            coldefs = [
                f'"{c}" REAL' if ty == "real" else f'"{c}" TEXT COLLATE U_NOCASE'
                for c, ty in zip(cols, t.types)
            ]
            con.execute(f"CREATE TABLE {name} ({', '.join(coldefs)})")
            cleaned = [
                [v.replace(",", "") if (ty == "real" and isinstance(v, str)) else v
                 for v, ty in zip(row, t.types)]
                for row in t.rows
            ]
            ph = ",".join("?" * len(cols))
            con.executemany(f"INSERT INTO {name} VALUES ({ph})", cleaned)
            built += 1

        con.commit()
        con.close()
        os.replace(tmp, db)
        print(f"[{ctx.split.name}] built real-name db -> {built} tables")
        return ctx
