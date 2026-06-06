from src.model.table import Table

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_str(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _select_clause(sql: dict, namer) -> str:
    agg = AGG_OPS[sql["agg"]]
    sel = _quote_ident(namer(sql["sel"]))
    return f"{agg}({sel})" if agg else sel


def decode_sql(sql: dict, table: Table, table_name: str, columns: list[str]) -> str:
    types = table.types
    query = f"SELECT {_select_clause(sql, lambda i: columns[i])} FROM {table_name}"
    if sql["conds"]:
        clauses = []
        for c, op, val in sql["conds"]:
            value = str(val) if types[c] == "real" else _quote_str(val)
            clauses.append(f"{_quote_ident(columns[c])} {COND_OPS[op]} {value}")
        query += " WHERE " + " AND ".join(clauses)
    return query
