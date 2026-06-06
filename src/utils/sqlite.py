import sqlite3


def _u_nocase(a: str, b: str) -> int:
    a, b = a.casefold(), b.casefold()
    return (a > b) - (a < b)


def connect(db_path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.create_collation("U_NOCASE", _u_nocase)
    return con
