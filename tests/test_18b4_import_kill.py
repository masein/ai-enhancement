"""18b part 4, point 26: a whole benchmark's import killed between moving its
answers into the row and writing its registry — a deploy during a round —
is recorded when the same bundle comes again (70df001: "imported already",
never recorded). Invented questions and a stand-in box; no model runs."""

from __future__ import annotations

from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import TASK as GT
from test_17_gguf_box import box, bundle_of, register, run_box  # noqa: F401 — box: fixture
from test_17b_review import imported


def test_26_an_import_killed_after_its_answers_moved_is_recorded(box):  # noqa: F811
    """a non-shard import killed between moving its answers and writing its
    registry: the same bundle again records it"""
    import import_frontier as imf
    from service import config
    assert run_box(box, "run", thinking="off") == 0
    register(box["sha"])
    path = bundle_of(box, "run", thinking=False)
    code, said = imported(path)
    assert code == 0, said
    row = config.OUT_DIR / "served__lda-box"
    reg = imf.registry(row)
    assert GT in reg["tasks"] and len(reg["imports"]) == 1
    imf._write_registry(row, {**reg, "imports": [], "tasks": {}})     # killed before this
    code, said = imported(path)
    assert code == 0, said
    assert not any(x.startswith("imported already") for x in said), said   # 70df001: it was
    assert any(f"{GT}: its answers were here, not recorded" in x for x in said), said
    reg = imf.registry(row)
    assert GT in reg["tasks"] and len(reg["imports"]) == 1
