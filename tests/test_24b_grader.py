"""24b: a grader batch stuck on a provider that's gone — OTIS's check stayed
pinned to Google Vertex, 404 every time, after Gemini 2.5 Flash was chosen
again on Google AI Studio (21 Oct). A batch moves to the grader chosen now
when only its provider differs; a pinned provider OpenRouter no longer lists
is said, on its grader's row and on the batch; each grade names the provider
that gave it, and the score counts them. Invented questions and a stand-in
for OpenRouter only: nothing is sent anywhere."""

from __future__ import annotations

import threading
import time

import report_lm_eval as report
from service import ai_models, config, llm_poller
from service import frontier as sf
from service import frontier_grade as fgr
from test_17_grading import GRADERS, ROW, drain, results, svc  # noqa: F401 — svc is the fixture
from test_17b_grading import GEMINI, GPT, plain, stub

DRIFTED = ai_models.drifted                      # as it is, before the fixture stands in for it
PROVIDERS = {"a": ("openai-a/global", "Provider A"), "b": ("openai-b/flex", "Provider B")}


def pins(monkeypatch, now: dict) -> None:
    """the job's model pinned to the provider `now["p"]` names — same version"""
    def pin(mid, job="pin"):
        tag, name = PROVIDERS[now["p"]]
        return {"kind": "openrouter", "id": mid, "version": GRADERS[mid][0], "name": mid,
                "provider": tag, "provider_name": name, "precision": "unknown",
                "price_in": GRADERS[mid][1], "price_out": GRADERS[mid][2]}
    monkeypatch.setattr(ai_models, "pin", pin)


def test_another_provider_moves_the_unsent_and_the_grades_name_each(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "OPENROUTER_CONCURRENCY", 1)
    now = {"p": "a"}
    pins(monkeypatch, now)
    fgr.save("simpleqa", GPT, "masein")
    gate = threading.Event()
    seen: list = []

    asked = stub(monkeypatch, plain)
    real = fgr.GraderChat._complete

    def complete(self, row):
        seen.append((self.pin.get("provider"), row["custom_id"]))
        if self.model == GPT and sum(1 for _, c in seen if "algebra/" not in c) == 1:
            gate.wait(10)
        return real(self, row)
    monkeypatch.setattr(fgr.GraderChat, "_complete", complete)
    fgr.start("masein")
    for _ in range(200):
        if any("algebra/" not in c for _, c in seen):
            break
        time.sleep(0.05)
    # stopped, as the stuck batch was (its requests refused), and the same model
    # chosen again: OpenRouter pins it to another provider
    fgr.stop("masein")
    now["p"] = "b"
    fgr.save("simpleqa", GPT, "masein")
    gate.set()
    time.sleep(0.5)
    llm_poller.tick()
    out = next(p for p in fgr.pending() if p["slot"] == "simpleqa")
    assert out["pin"]["provider"] == "openai-a/global"
    assert fgr._moving(out) == "another provider was chosen"      # 24b: the version is the same
    got = fgr.start("masein")
    assert got["moved"] == 4
    drain()
    sqa = [(p, c) for p, c in seen if "algebra/" not in c]
    assert sqa[0][0] == "openai-a/global"
    assert sorted(c for p, c in sqa[1:]) == [f"frgr:{k}#0" for k in range(1, 5)]
    assert {p for p, _ in sqa[1:]} == {"openai-b/flex"}         # nothing asked twice
    assert len(asked) == len(seen)
    # each grade names the provider that gave it, and the score counts them
    g = sf.read_grades(sf.task_dir(config.OUT_DIR / ROW, "simpleqa_epoch"))
    by = sorted(x["provider"] for x in g["items"].values())
    assert by == ["Provider A"] + ["Provider B"] * 4
    fr = results("simpleqa_epoch")["frontier"]
    assert fr["grader"]["providers"] == {"Provider A": 1, "Provider B": 4}
    assert "1 on Provider A, 4 on Provider B" in report.frontier_how(fr)


def test_a_version_change_still_moves(svc, monkeypatch):  # noqa: F811
    p = {"slot": "simpleqa", "pin": {"kind": "openrouter", "id": GPT, "version": "old",
                                     "provider": "openai-a/global"}}
    monkeypatch.setattr(fgr, "chosen", lambda slot: {"kind": "openrouter", "id": GPT,
                                                     "version": "new", "provider": "openai-a/global"})
    assert fgr._moving(p) == "another grader was chosen"
    monkeypatch.setattr(fgr, "chosen", lambda slot: {"kind": "openrouter", "id": GPT,
                                                     "version": "old", "provider": "openai-a/global"})
    assert fgr._moving(p) == ""


def test_a_provider_no_longer_listed_is_said_on_the_row_and_the_batch(svc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ai_models, "drifted", DRIFTED)
    monkeypatch.setattr(ai_models, "_EP", {})
    monkeypatch.setattr(ai_models, "providers", lambda mid: [
        {"name": "Google AI Studio", "tag": "google-ai-studio/flex", "precision": "unknown",
         "price_in": 0.3, "price_out": 2.5, "up": True}])
    vertex = {"kind": "openrouter", "id": GEMINI, "version": GRADERS[GEMINI][0],
              "provider": "google-vertex/global", "provider_name": "Google Vertex"}
    words = ("OpenRouter no longer serves google/gemini-2.5-flash on google-vertex/global — "
             "choose the job's model again on AI models")
    # not known yet: nothing said, the providers asked in the background
    assert ai_models.provider_gone(vertex) == ""
    for _ in range(100):
        if GEMINI in ai_models._EP:
            break
        time.sleep(0.02)
    assert ai_models.provider_gone(vertex) == words
    assert ai_models.drifted(vertex) == words                   # the batch's own check
    studio = {**vertex, "provider": "google-ai-studio/flex"}
    assert ai_models.drifted(studio) == ""
    # the grading section says it on that grader's row
    monkeypatch.setattr(fgr, "chosen", lambda slot: vertex if slot == "math" else None)
    rows = {g["slot"]: g for g in svc.get("/api/frontier/grading").json()["graders"]}
    assert rows["math"]["gone"] == words and rows["simpleqa"]["gone"] == ""
    # with no key nothing is asked or said
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    assert ai_models.provider_gone(vertex) == ""
