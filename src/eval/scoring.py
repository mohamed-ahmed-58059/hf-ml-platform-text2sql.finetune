def score(prediction, gold_sql, con) -> dict:
    try:
        gold = set(con.execute(gold_sql).fetchall())
    except Exception:
        return {"exact": 0.0, "f1": 0.0}
    if not prediction:
        return {"exact": 0.0, "f1": 0.0}
    try:
        pred = set(con.execute(prediction).fetchall())
    except Exception:
        return {"exact": 0.0, "f1": 0.0}
    if not gold and not pred:
        return {"exact": 1.0, "f1": 1.0}
    if not gold or not pred:
        return {"exact": 0.0, "f1": 0.0}
    matched = len(pred & gold)
    p = matched / len(pred)
    r = matched / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"exact": 1.0 if pred == gold else 0.0, "f1": f1}
