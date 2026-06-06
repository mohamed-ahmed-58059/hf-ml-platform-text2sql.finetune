def deduplicate_names(rows: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    used: set[str] = set()
    for row in sorted(rows, key=lambda r: r["table_id"]):
        base = row["name"]
        name, i = base, 1
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used.add(name)
        out[row["table_id"]] = name
    return out
