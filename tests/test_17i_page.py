"""17i, part 4, point 25: the rows 0adb522's start-up fill mislabelled — a
typed note made a run on this server "rented", a GPU name with a bracket was
cut — are listed by scripts/where_check.py and set right at start-up (511854e
touched only blank rows). Rows invented on the board."""

from __future__ import annotations

import json

from service import config, db
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture


def a_log(sid: int, model: str, first: str) -> None:
    p = config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(first + "\n[import] the row …\n")


def test_25_rows_0adb522_mislabelled_are_listed_and_set_right(svc, capsys):  # noqa: F811
    import import_frontier as imf
    import where_check
    # as 0adb522's fill left them, from their notes
    typed = db.add("served/x", "instruct", "frontier", "masein",
                   "imported from a rented GPU (my own words) · nothing imported", status="done")
    cut = db.add("served/x", "instruct", "frontier", "masein",
                 "imported from a rented GPU (Tesla V100 (16 GB)) · box B3 · z.tar.gz",
                 status="done")
    failed = db.add("served/x", "instruct", "frontier", "masein",
                    "imported from a rented GPU (Tesla V100 (16 GB)) · y.tar.gz", status="failed")
    two = db.add("served/y", "instruct", "devicemark", "masein", "imported", status="done")
    here = db.add("served/x", "instruct", "frontier", "masein", "a run here", status="done")
    for sid, where in ((typed, "rented GPU · my own words"), (cut, "rented GPU · Tesla V100 (16 GB"
                       " · box B3"), (failed, "rented GPU · Tesla V100 (16 GB"),
                       (two, "rented GPU · RTX 5090 ×2")):
        db.update(sid, where_ran=where)
    (config.OUT_DIR / "served__x").mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "served__x" / imf.REGISTRY).write_text(json.dumps({"imports": [
        {"sid": cut, "gpu": "Tesla V100 (16 GB)", "box": "B3"}]}))
    (config.OUT_DIR / "served__y").mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "served__y" / "remote_imports.json").write_text(json.dumps({"imports": [
        {"sid": two, "gpu": "NVIDIA GeForce RTX 5090"}]}))
    a_log(failed, "served/x", f"===== [{failed}] imported y.tar.gz (sha256 0123456789abcdef) by "
          "masein: gpqa_diamond_epoch, run on a rented GPU (Tesla V100 (16 GB)) =====")
    # the one query, read-only, as masein runs it against the live board
    assert where_check.main([]) == 0
    said = capsys.readouterr().out
    assert f"#{typed} served/x: 'rented GPU · my own words' → 'this server'" in said, said
    assert f"#{cut} served/x: 'rented GPU · Tesla V100 (16 GB · box B3' → " \
           "'rented GPU · Tesla V100 (16 GB) · box B3'" in said
    assert f"#{failed} " in said and "3 row(s) not as their records say" in said
    assert f"#{two} " not in said and f"#{here} " not in said    # whole, or never imported
    assert db.get(typed)["where_ran"] == "rented GPU · my own words"   # nothing written yet
    # and the board sets them right as it starts (511854e: blank rows only)
    db.init()
    assert db.get(typed)["where_ran"] == ""
    assert db.get(cut)["where_ran"] == "rented GPU · Tesla V100 (16 GB) · box B3"
    assert db.get(failed)["where_ran"] == "rented GPU · Tesla V100 (16 GB)"
    assert db.get(two)["where_ran"] == "rented GPU · RTX 5090 ×2"
    assert where_check.wrong(__import__("sqlite3").connect(config.DB_PATH), config.OUT_DIR,
                             config.LOGS_DIR) == []
