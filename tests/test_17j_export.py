"""17j, part 4: the export (docs/prompts/phase-17j-eighth-review-hle-cases-
runs-page.md, points 22 to 27). A test a point, each failing on 0abb757. Runs
invented on the board; Hugging Face is never asked (its answers are stood in
for); nothing is uploaded."""

from __future__ import annotations

import base64
import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest

import export_frontier_raw as efr
import export_safe as es
from service import config, db, served
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import SERVED, box  # noqa: F401
from test_17g_review import LONG
from test_17h_review import _export_world

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCT = "teamacct"
QWEN = "served/Qwen3.6-35B-A3B-BF16"


def log_of(sid: int) -> Path:
    return config.LOGS_DIR / f"service_{sid}_{SERVED.replace('/', '__')}.log"


def exported_log(sid: int, tmp_path, *lines: str) -> str:
    log = log_of(sid)
    log.write_text(log.read_text() + "\n" + "\n".join(lines) + "\n")
    return (efr.export_run(sid, tmp_path / "raw") / "log.txt").read_text()


def fetched_from(source: str) -> None:
    """the served build's file, as the board registered it: where it came from"""
    rec = served.get(SERVED)
    db.served_put({**rec, "file_sha256": {"sha256": (rec.get("gguf_pin") or {}).get("sha256"),
                                          "name": "x.gguf", "source": source}})


# ---------------------------------------------------------------------------
# 22: the scrub's names worked out, said, and a value too short refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hosts", [",", "x", "ab"])
def test_22_a_value_too_short_to_be_a_name_refuses_public(box, tmp_path, monkeypatch, capsys,  # noqa: F811
                                                          hosts):
    sid = _export_world(box, monkeypatch)
    monkeypatch.setenv("SCRUB_HOSTS", hosts)
    monkeypatch.setenv("SCRUB_ACCOUNTS", ACCT)
    db.public_set(SERVED, True, "masein")
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    out = tmp_path / "raw"
    assert efr.main(["--run", str(sid), "--out", str(out)]) == 1      # 0abb757: 0, public
    said = capsys.readouterr().out
    assert "public/ refused: " in said, said
    assert ("too short to be a host's name" if hosts != "," else "gives no name") in said
    assert not (out / "public").exists() and (out / "private").exists()


def test_22_the_containers_own_name_isnt_the_servers(box, tmp_path, monkeypatch, capsys):  # noqa: F811
    import socket
    sid = _export_world(box, monkeypatch)
    monkeypatch.setattr(socket, "gethostname", lambda: "c0ffee123456")
    monkeypatch.setenv("SCRUB_HOSTS", "c0ffee123456")
    monkeypatch.setenv("SCRUB_ACCOUNTS", ACCT)
    db.public_set(SERVED, True, "masein")
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    assert efr.main(["--run", str(sid), "--out", str(tmp_path / "raw")]) == 1   # 0abb757: 0
    said = capsys.readouterr().out
    assert "this server's own name isn't in SCRUB_HOSTS" in said and "c0ffee123456" in said, said


def test_22_the_export_works_out_the_names_and_says_them_above_the_list(
        box, tmp_path, monkeypatch, capsys):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    # a wrong account given; the real one is where the build was fetched from
    fetched_from(f"hf://{ACCT}/evalboard-private@main/x.gguf")
    rec = served.get(SERVED)
    db.served_put({**rec, "base_url": "http://judge-box:8000/v1"})
    monkeypatch.setenv("SCRUB_HOSTS", "board-host")
    monkeypatch.setenv("SCRUB_ACCOUNTS", "wrongacct")
    db.public_set(SERVED, True, "masein")
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    log = log_of(sid)
    log.write_text(log.read_text() + f"\n[import] the files came from {ACCT} · judge-box\n")
    out = tmp_path / "raw"
    assert efr.main(["--run", str(sid), "--out", str(out)]) == 0
    said = capsys.readouterr().out
    first = next(x for x in said.splitlines() if x.startswith("will remove: "))   # 0abb757: none
    assert said.index(first) < said.index("These would go to public/")
    for name in ("board-host", ACCT, "wrongacct"):
        assert name in first, (name, first)
    # 18b: a dotless host from the settings is no name of the scrub's (only
    # dotted ones are); its line stays out here for the account it names
    assert "judge-box" not in first, first
    for f in (next((out / "public").iterdir())).iterdir():
        text = f.read_text()
        assert ACCT not in text and "judge-box" not in text, f.name


# ---------------------------------------------------------------------------
# 23: the public file a model is, checked; anything else said
# ---------------------------------------------------------------------------

def hf_answers(monkeypatch, files: dict, private: set = frozenset()):
    """Hugging Face, stood in for: {repo/path: sha256}; a repo in `private`
    asks to sign in"""
    from service import public_files as pf

    def ask(url, method="HEAD"):
        rest = url.removeprefix(pf.HF + "/")
        if rest.startswith("api/models/"):
            repo = rest.removeprefix("api/models/")
            if repo in private:
                raise pf.NotPublic("Hugging Face asks to sign in for it: it is private or gated")
            return {}, json.dumps({"id": repo, "private": False}).encode()
        repo, _, path = rest.partition("/resolve/main/")
        if repo in private:
            raise pf.NotPublic("Hugging Face asks to sign in for it: it is private or gated")
        sha = files.get(f"{repo}/{path}")
        if not sha:
            raise pf.NotPublic("Hugging Face has no such file there")
        return {"X-Linked-Etag": f'"{sha}"', "X-Repo-Commit": "c" * 40}, b""
    monkeypatch.setattr(pf, "_ask", ask)


def qwen(parts: list[tuple[str, str]] | None = None, sha: str = "ab" * 32) -> None:
    db.served_put({"id": QWEN, "name": "BF16", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto", "pin": {},
                   "by": "masein", "at": 0, "file_sha256": {
                       "sha256": sha, "name": "Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
                       "source": f"hf://{ACCT}/evalboard-private@main/BF16",
                       **({"parts": [{"name": n, "sha256": h} for n, h in parts]}
                          if parts else {})}})


def test_23_a_checked_public_file_says_so_and_anything_else_says_check(svc, monkeypatch):  # noqa: F811
    from service import public_files as pf               # 0abb757: no such module
    a, b = "Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf", "Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf"
    qwen([(a, "1" * 64), (b, "2" * 64)])
    db.served_put({"id": "served/team-build", "name": "team", "base_url": "", "key": "",
                   "how": "x", "thinking": "auto", "pin": {}, "by": "masein", "at": 0,
                   "file_sha256": {"sha256": "cd" * 32, "name": "team.gguf",
                                   "source": f"hf://{ACCT}/evalboard-private@main/team.gguf"}})
    repo = "unsloth/Qwen3.6-35B-A3B-GGUF"
    hf_answers(monkeypatch, {f"{repo}/BF16/{a}": "1" * 64, f"{repo}/BF16/{b}": "2" * 64,
                             f"{ACCT}/evalboard-private/team.gguf": "cd" * 32},
               private={f"{ACCT}/evalboard-private"})
    rec = pf.check(QWEN, repo, f"BF16/{a}")
    assert rec["same"] and not rec["why"], rec
    db.public_file_set(QWEN, rec, "masein")
    team = pf.check("served/team-build", f"{ACCT}/evalboard-private", "team.gguf")
    assert not team["same"] and "sign in" in team["why"]
    db.public_file_set("served/team-build", team, "masein")
    said: list[str] = []
    es.confirm([QWEN, "served/team-build"], ask=lambda p: "no", say=said.append)
    line = next(x for x in said if x.startswith(f"  {QWEN}"))
    assert f"the same file as {repo}/BF16/{a}, checked" in line, said
    k = said.index(next(x for x in said if x.startswith("  served/team-build")))
    assert said[k + 1].strip().startswith("CHECK: not shown to be a public file"), said
    assert not any("CHECK" in x for x in said[:k]), said
    # a file registered since the check isn't checked
    qwen([(a, "1" * 64), (b, "3" * 64)], sha="ef" * 32)
    said.clear()
    es.confirm([QWEN], ask=lambda p: "no", say=said.append)
    assert any(x.strip().startswith("CHECK: not shown") for x in said), said


def test_23_a_part_that_differs_is_said(svc, monkeypatch):  # noqa: F811
    from service import public_files as pf
    a, b = "q-00001-of-00002.gguf", "q-00002-of-00002.gguf"
    qwen([(a, "1" * 64), (b, "2" * 64)])
    hf_answers(monkeypatch, {f"unsloth/Q/BF16/{a}": "1" * 64, f"unsloth/Q/BF16/{b}": "9" * 64})
    rec = pf.check(QWEN, "unsloth/Q", f"BF16/{a}")
    assert not rec["same"] and rec["why"].startswith("1 of its 2 parts differ"), rec


def test_23_the_page_gives_and_checks_the_public_file(svc, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    import service.app as appmod
    qwen()
    hf_answers(monkeypatch, {"unsloth/Q/BF16/x.gguf": "ab" * 32})
    c = TestClient(appmod.app)
    r = c.post("/api/models/public-file", json={"model": QWEN, "repo": "unsloth/Q",
                                                "path": "BF16/x.gguf", "by": "masein"})
    assert r.status_code == 200, r.text                         # 0abb757: 404
    assert r.json()["file"]["holds"] is True
    appmod._cache.update(key=None, payload=None, at=0.0)
    got = c.get("/api/results").json()["public_files"][QWEN]
    assert got["repo"] == "unsloth/Q" and got["holds"] and got["by"] == "masein"
    bad = c.post("/api/models/public-file", json={"model": QWEN, "repo": "not a repo",
                                                  "path": "", "by": "masein"})
    assert bad.status_code == 422


# ---------------------------------------------------------------------------
# 24: no name in the header, no exception's text in a tagged line
# ---------------------------------------------------------------------------

def test_24_free_words_dont_go_out_in_kept_lines(box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    first = log_of(sid).read_text().splitlines()[0]
    assert first.startswith(f"===== [{sid}] imported ") and " by " not in first, first  # 0abb757
    text = exported_log(
        sid, tmp_path,
        f"===== [{sid + 1}] imported a.tar.gz (sha256 0123456789abcdef) by masein: "
        "gpqa_diamond_epoch, run on a rented GPU (RTX 5090) =====",
        f"===== [{sid + 2}] imported a b.tar.gz (sha256 0123456789abcdef) by masein: "
        "gpqa_diamond_epoch, run on a rented GPU (RTX 5090) =====",
        "[frontier] GPQA Diamond couldn't be loaded. Nothing was asked "
        "(ConnectionError(judge-box refused the connection))",
        "[frontier] judge: ConnectionRefusedError at judge-box port 8000",
        "2026-10-07 09:00:09 the run: failed · the judge-box answered 502 at 41 of 198")
    assert "masein" not in text, text                           # 0abb757: in the header
    assert f"===== [{sid + 1}] imported a.tar.gz (sha256 0123456789abcdef): " in text
    assert "judge-box" not in text and "ConnectionRefusedError" not in text, text
    assert "[frontier] GPQA Diamond couldn't be loaded. Nothing was asked\n" in text
    assert "2026-10-07 09:00:09 the run: failed\n" in text


# ---------------------------------------------------------------------------
# 25: a gated question with &nbsp;, as five words, in base64; the written-off
# line without its reason
# ---------------------------------------------------------------------------

def test_25_a_quote_with_nbsp_five_words_or_base64_is_caught():
    q = es.Questions(["Which enzyme splits the lactose sugar found in fresh milk products?"])
    for line in ("[frontier] Which&nbsp;enzyme&nbsp;splits&nbsp;the&nbsp;lactose&nbsp;sugar",
                 "[frontier] the lactose sugar found in",
                 "[frontier] " + base64.b64encode(b"which enzyme splits the lactose sugar found")
                 .decode()):
        assert q.quotes(line), line                             # 0abb757: through
    assert not q.quotes("[frontier] gpqa_diamond_epoch: 120 of 198 answered")


def test_25_the_written_off_line_goes_out_without_the_servers_words(box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    q = LONG[2]["question"]
    text = exported_log(
        sid, tmp_path,
        "[frontier] gpqa_diamond_epoch: question q7, run 0: the server failed on it twice — "
        f"written as no answer, counted wrong (HTTP 400: the prompt {q} is too long)")
    assert "boiling point" not in text, text
    assert ("[frontier] gpqa_diamond_epoch: question q7, run 0: the server failed on it twice — "
            "written as no answer, counted wrong\n") in text, text   # 0abb757: left out whole


# ---------------------------------------------------------------------------
# 26: loaded once, each set's count said, and a failure exits non-zero
# ---------------------------------------------------------------------------

def test_26_the_sets_load_once_their_counts_are_said_and_a_failure_exits_1(
        box, tmp_path, monkeypatch, capsys):  # noqa: F811
    import frontier as fb
    sid = _export_world(box, monkeypatch)
    loads = []
    once = es.private_questions

    def counted(say=None):
        loads.append(1)
        return once(say)
    monkeypatch.setattr(es, "private_questions", counted)
    capsys.readouterr()
    assert efr.main(["--run", str(sid), "--out", str(tmp_path / "raw"), "--private"]) == 0
    said = capsys.readouterr().out
    assert "the logs are checked against: " in said, said     # 0abb757: not said
    assert f"GPQA Diamond {len(LONG)}" in said and "(none on this server)" in said, said
    assert len(loads) == 1
    real = fb.load
    monkeypatch.setattr(fb, "load", lambda task, root=None: (_ for _ in ()).throw(
        RuntimeError("gated")) if task == "hle_text_cais" else real(task, root))
    loads.clear()
    assert efr.main(["--run", str(sid), "--out", str(tmp_path / "raw2"), "--private"]) == 1
    assert len(loads) == 1, loads                               # 0abb757: again for the run


# ---------------------------------------------------------------------------
# 27: closed stdin, values by kind, where from a list
# ---------------------------------------------------------------------------

def test_27_closed_stdin_is_a_no_and_never_a_traceback():
    code = ("import sys; sys.path[:0] = ['scripts', '.']; import export_safe as es; "
            "es.file_of = lambda m: {}; es.will_remove = lambda: 'will remove: -'; "
            "print('answer', es.confirm(['served/x']))")
    got = subprocess.run(["sh", "-c", f'"{sys.executable}" -c "{code}" <&-'], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=60)
    assert "Traceback" not in got.stderr, got.stderr            # 0abb757: lost sys.stdin
    assert "answer False" in got.stdout, (got.stdout, got.stderr)


def test_27_values_are_checked_by_their_kind():
    token = "abcdefghijklmnopqrstuvw"                          # 23 characters
    got, _ = es.env(f"CUDA_VISIBLE_DEVICES={token} LLAMA_ARG_CTX_SIZE=81920 "
                    f"LLAMA_MOE_ROUTE_MODE={token} OMP_NUM_THREADS=8")
    assert got["CUDA_VISIBLE_DEVICES"] == es.WITHHELD, got      # 0abb757: the token
    assert got["LLAMA_MOE_ROUTE_MODE"] == es.WITHHELD
    assert got["LLAMA_ARG_CTX_SIZE"] == "81920" and got["OMP_NUM_THREADS"] == "8"
    fl, left = es.flags(f"-c 8192 -ctk {token} --temp 0.6 -fa {token} -ts 1,1 "
                        f"-ot .ffn_.*_exps.=CPU -ot {token}=CPU")
    assert token not in " ".join(fl), fl                       # 0abb757: through
    assert fl == ["-c", "8192", "--temp", "0.6", "-ts", "1,1", "-ot", ".ffn_.*_exps.=CPU"] \
        and left == 3, (fl, left)


def test_27_where_is_picked_from_a_list(box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    db.update(sid, where_ran="rented GPU · RTX 5090 · box A3 kept by masein at the office")
    dest = efr.export_run(sid, tmp_path / "raw")
    setup = json.loads((dest / "setup.json").read_text())
    assert setup["where"] == "rented GPU · RTX 5090", setup["where"]   # 0abb757: the words
    assert "Where it ran: rented GPU · RTX 5090.\n" in (dest / "README.md").read_text()
    assert es.where_of("rented GPU · 2 × RTX 5090 · boxes A1, A3-2") == \
        "rented GPU · 2 × RTX 5090 · boxes A1, A3-2"
    assert es.where_of("this server") == "this server"
    assert es.where_of("my laptop") == "rented GPU"
