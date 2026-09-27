"""12f.3 addendum: served entries that share one file are setups of it, side by
side on the model page — the score, the median answer length, how many ran
out, and MTP's draft acceptance (timings.draft_n_accepted / draft_n) when the
server reports it, recorded with every answer. The server is
tests/fake_openai.py."""

from __future__ import annotations

import json

from service import config, db, runner
from test_served_12f1 import HOW, ME, fake, svc  # noqa: F401

MTP = "served/LDA-MTP-3"
PLAIN = "served/LDA-no-MTP"


def register(client, fake, name, **over):  # noqa: F811
    body = {"name": name, "base_url": fake.base, "based_on": "Qwen/Qwen3.6-35B-A3B",
            "how": HOW, "thinking": "off", "by": ME, **over}
    r = client.post("/api/served", json=body)
    assert r.status_code == 200, r.text
    return r.json()["model"]


def test_setups_of_one_file_record_and_sum_mtps_drafts(svc, fake):  # noqa: F811
    client, appmod, _ = svc
    a, b = register(client, fake, "LDA MTP 3"), register(client, fake, "LDA no MTP")
    assert a["pin"] == b["pin"]                      # one file, as the server reports it
    # the first is served with MTP: 10 tokens drafted an answer, 7 accepted
    fake.timings = lambda body: {"draft_n": 10, "draft_n_accepted": 7, "predicted_n": 40}
    runner.run_submission(db.get(db.add(MTP, "instruct", "everyday", ME, "")))
    fake.timings = lambda body: {"predicted_n": 40}          # no drafts reported
    runner.run_submission(db.get(db.add(PLAIN, "instruct", "everyday", ME, "")))
    # every answer records what the server said
    d = config.OUT_DIR / MTP.replace("/", "__")
    [samples] = d.glob("everyday_0shot/served/samples_*.jsonl")
    first = json.loads(samples.read_text().splitlines()[0])
    assert first["draft"] == {"n": 10, "accepted": 7}
    appmod._cache.update(key=None, payload=None, at=0.0)
    rows = {m["id"]: m for m in client.get("/api/results").json()["models"]}
    got = rows[MTP]["answerLength"]["everyday"]["draft"]
    assert got["rate"] == 0.7 and got["n"] == 10 * got["accepted"] // 7
    assert rows[PLAIN]["answerLength"]["everyday"]["draft"] is None
    # both carry the pin the page groups them by
    served = client.get("/api/results").json()["served"]
    assert served[MTP]["pin"]["file"] == served[PLAIN]["pin"]["file"]
