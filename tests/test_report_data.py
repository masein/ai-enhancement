"""The report's public data repository: paper-data/ built from the paper itself, the scan
for internal terms, and the upload's checks before anything is sent (no network here)."""
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "scripts" / "report_data"


def load(name):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = load("build_paper_data")
scan = load("scan_terms")
upload = load("upload_report_data")

PAPER = """# For the authors: notes

Notes before the title are never read. [TBD here doesn't count]

# A report: its title

## Abstract

Text.

## Key numbers

| Measure | Masein model (measured here) | How it compares |
| --- | --- | --- |
| GPQA Diamond (%), High / Medium / Instant | 82.7 ± 2.2 / 80.1 ± 2.4 / 75.3 ± 2.4 | High: level with o1 (76.8) |
| Token generation throughput on a phone (decode, tokens/s), Instant mode | OnePlus 15 (16 GB): 15.2 in short chats, 7.3 with 14,000 tokens of context; iPhone 17 Pro Max (12 GB): 11.4 in short chats. Medium mode, preliminary: 12.6 and 9.7 in short chats | No verdict |

**Table 5.** The Masein model, measured here, in its three modes: score (%) ± standard error.

| Benchmark | Questions × runs | High (%) | Ran out, High | Medium (%) | Ran out, Medium | Instant (%) | Ran out, Instant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPQA Diamond | 198 × 4 | 82.7 ± 2.2 | 7 of 792 | 80.1 ± 2.4 | 1 of 792 | 75.3 ± 2.4 | 0 of 792 |
| IFBench | 300 × 1 | 63.7 ± 2.8 | 15 of 300 | 33.3 ± 2.7 | 2 of 300 | 33.0 ± 2.7 | 6 of 300 |

**Table 7.** How the thinking goes, thinking on: finished answers only.

| Benchmark | Finished answers, right / wrong | Mean thinking tokens, right / wrong |
| --- | --- | --- |
| GPQA Diamond | 655 / 130 | 15,649 / 24,995 |

**Table 8.** Answers that ran out of room at the final limit.

| Benchmark | Ran out of room | Ended in a loop |
| --- | --- | --- |
| GPQA Diamond | 7 of 792 | 5 of 792 |
| IFBench | 15 of 300 | 12 of 300 |

**Table 9.** Against the frontier models. The frontier models' columns are Epoch AI's runs, not measured by us.

| Benchmark | Masein model | Gemini 4 Argon (high) |
| --- | --- | --- |
| GPQA Diamond | **82.7 ± 2.2** | 96.4 ± 1.2 |

**Table 10.** Against five models (Section 4.3). The other models' scores are Epoch AI's runs.

| Model | GPQA Diamond |
| --- | --- |
| o1 | level, +5.9 (−1.4 to +13.3) |
| Claude Haiku 4.5 | higher, +11.5 (+3.8 to +19.2) ‡ |

**Table 12.** DeviceMark, all 28 rows. Rows run by DeviceMark are its published leaderboard.

| Model | Run by | Composite \\[95% range\\] |
| --- | --- | --- |
| **Masein model, Instant** | **us (replica)** | **84.4 \\[81.4, 87.3\\]** |
| **Masein model, High** | **us (replica)** | **45.2 \\[41.4, 49.6\\]** |
| Qwen3.5-4B (int8) | DeviceMark | 26.1 \\[22.4, 29.9\\] |

**Table F1.** GPQA Diamond, 2 models. The Epoch scores are Epoch AI's runs.

| Model (setting) | Epoch score | Verdict | Gap (95% range) |
| --- | --- | --- | --- |
| **Masein model (measured here)** | **82.7 ± 2.2** | – | – |
| Grok 3 mini (low) † | 76.3 ± 3.0 | level with ‡ | +6.4 (−0.9 to +13.7) |

**Table G1.** On-device models: the Masein model, measured by us, and other makers' numbers as published, none measured by us.

| Model | GPQA Diamond | Source, date |
| --- | --- | --- |
| **Masein model (measured here)** | **82.7 ± 2.2** | this report |
| Qwen3.5-9B | 81.7 | Qwen's card, Feb 2026 |

&#91;embedded content: Figure 2. A grader marks every answer. How one question becomes a score. Source: Section 4.\\]

&#91;embedded content: Figure 3. Some questions are right on some runs. The share right on every run, on some runs and on none, for 198 GPQA Diamond questions (4 runs each: 141, 44 and 13 questions). Source: ours.\\]

&#91;embedded content: Figure 4. It thinks longer when wrong. Mean thinking tokens per finished answer, right and wrong (Table 7). Source: Table 7.\\]

&#91;embedded content: Figure 5. Out of room: 5.0% of IFBench's answers. The share of answers that ran out of room at the final limit, thinking on, on GPQA Diamond and IFBench; limits in Table 4. Source: Table 8.\\]

&#91;embedded content: Figure 10. DeviceMark. DeviceMark composite with its 95% range for all rows of Table 12. Source: Table 12.\\]
"""


def rows(path):
    return list(csv.reader(open(path, encoding="utf-8")))


@pytest.fixture
def paper(tmp_path):
    p = tmp_path / "paper.md"
    p.write_text(PAPER, encoding="utf-8")
    return p


def test_each_table_and_figure_is_a_file_named_by_the_papers_own_numbers(paper, tmp_path):
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper)]) == 0
    got = sorted(p.name for p in out.glob("*.csv"))
    assert got == sorted(["key-numbers.csv", "table-5.csv", "table-7.csv", "table-8.csv",
                          "table-9.csv", "table-10.csv", "table-12.csv", "table-f1.csv", "table-g1.csv",
                          "figure-3.csv", "figure-4.csv", "figure-5.csv", "figure-10.csv",
                          "index.csv"])
    # the cells as printed: a score and its SE apart, a "right / wrong" pair apart,
    # thousands separators gone, nothing re-rounded
    t5 = rows(out / "table-5.csv")
    assert t5[0][2:4] == ["High (%)", "High (%), SE"]
    assert t5[1][2:11] == ["82.7", "2.2", "7 of 792", "80.1", "2.4", "1 of 792", "75.3", "2.4", "0 of 792"]
    t7 = rows(out / "table-7.csv")
    assert t7[0][1:3] == ["Finished answers, right", "Finished answers, wrong"]
    assert t7[1] == ["GPQA Diamond", "655", "130", "15649", "24995"]
    # key numbers: one row per mode (r23: High / Medium / Instant), the comparison on the mode
    # it is about, and a row that isn't a score stays as printed
    k = rows(out / "key-numbers.csv")
    assert [r[1:4] for r in k[1:4]] == [["High", "82.7", "2.2"], ["Medium", "80.1", "2.4"],
                                        ["Instant", "75.3", "2.4"]]
    assert [r[-1] for r in k[1:4]] == ["High: level with o1 (76.8)", "", ""]
    assert k[4][2] == ("OnePlus 15 (16 GB): 15.2 in short chats, 7.3 with 14,000 tokens of context; iPhone 17 Pro "
                       "Max (12 GB): 11.4 in short chats. Medium mode, preliminary: 12.6 and 9.7 in short chats")
    # the index names each file's caption
    ix = {r[0]: r for r in rows(out / "index.csv")[1:]}
    assert ix["table-8.csv"][1:] == ["Table 8", "Answers that ran out of room at the final limit."]
    assert ix["figure-10.csv"][1] == "Figure 10"


def test_numbers_we_didnt_measure_are_marked_as_theirs(paper, tmp_path):
    out = tmp_path / "paper-data"
    build.main([str(out), str(paper)])
    # whole columns someone else's: their heading says whose
    t9 = rows(out / "table-9.csv")
    assert t9[0][1] == "Masein model"
    assert t9[0][3] == "Gemini 4 Argon (high) [Epoch AI Benchmarking Hub, CC BY 4.0]"
    # rows that differ: a Source column
    # r22: whose a number is comes from the paper's words; no table has italics since r18.
    # Table 12 says it row by row ("Run by")
    t12 = rows(out / "table-12.csv")
    assert [r[-1] for r in t12[1:]] == ["measured by Teraformer", "measured by Teraformer", build.DEVICEMARK]
    assert t12[0][2:5] == ["Composite", "Composite, 95% low", "Composite, 95% high"]
    f1 = rows(out / "table-f1.csv")
    assert f1[1][-1] == build.OURS          # every row but the Masein model's is Epoch's
    assert f1[2] == ["Grok 3 mini (low)", "yes", "76.3", "3.0", "level with", "yes", "+6.4",
                     "-0.9", "+13.7", build.EPOCH]
    # a caption that says the other rows are as published, none measured by us (Table G1)
    g1 = rows(out / "table-g1.csv")
    assert g1[0][-1] == "Whose numbers" and [r[-1] for r in g1[1:]] == [build.OURS, build.MAKER]
    # a verdict cell as its parts, with its marks apart
    t10 = rows(out / "table-10.csv")
    assert t10[2][1:6] == ["higher", "+11.5", "+3.8", "+19.2", "‡"]


def test_figures_come_from_the_tables_and_captions_they_print(paper, tmp_path):
    out = tmp_path / "paper-data"
    build.main([str(out), str(paper)])
    assert rows(out / "figure-3.csv")[1] == ["GPQA Diamond", "198", "4", "141", "44", "13"]
    assert rows(out / "figure-4.csv")[1] == ["GPQA Diamond", "655", "130", "15649", "24995"]
    assert [r[0] for r in rows(out / "figure-5.csv")[1:]] == ["IFBench", "GPQA Diamond"]
    assert rows(out / "figure-5.csv")[1][3] == "5.0"
    assert rows(out / "figure-10.csv")[3][2:] == ["26.1", "22.4", "29.9", build.DEVICEMARK]


def test_a_caption_that_disagrees_with_its_table_stops_the_build(paper, tmp_path):
    paper.write_text(PAPER.replace("Out of room: 5.0%", "Out of room: 5.3%"), encoding="utf-8")
    with pytest.raises(SystemExit, match="says 5.3% for IFBench"):
        build.main([str(tmp_path / "out"), str(paper)])


def test_a_figure_the_script_doesnt_know_stops_it(paper, tmp_path):
    paper.write_text(PAPER + "\n&#91;embedded content: Figure 11. Something new. A pie.\\]\n",
                     encoding="utf-8")
    with pytest.raises(SystemExit, match="Figure 11 is new"):
        build.main([str(tmp_path / "out"), str(paper)])


SPEED_ROW = ("| Masein model, Medium | us (replica) | 84.6 \\[81.6, 87.3\\] |\n"
             "| Qwen3.5-4B, thinking off | us (replica) | 80.1 \\[76.6, 83.4\\] |\n")
SPEED_FIG = ("\n&#91;embedded content: Figure 1. Fast and right. DeviceMark composite against token generation throughput "
             "on a phone (Table 12). Source: Table 12.\\]\n")


def speed_paper(paper):
    """the fixture with a replica row for Qwen3.5-4B and the speed chart as Figure 1"""
    text = PAPER.replace("| Qwen3.5-4B (int8) |", SPEED_ROW + "| Qwen3.5-4B (int8) |", 1) + SPEED_FIG
    paper.write_text(text, encoding="utf-8")
    return paper


def speeds_file(tmp_path, **change):
    """figure-1.csv as the figure script writes it; change={"model, mode": {column: value}}"""
    dm = "DeviceMark leaderboard, iPhone 17 Pro: Qwen3.5-4B int8hu on Core AI, decode tokens/s"
    rows = [{"model": "Masein model", "mode": "Instant", "devicemark_composite": "84.4", "ci95_low": "81.4",
             "ci95_high": "87.3", "score_source": build.OURS, "speed_tok_s": "11.4",
             "speed_source": "Teraformer's app on an iPhone 17 Pro Max, Instant mode, short chats (Table 1)",
             "speed_credit": build.OURS},
            {"model": "Masein model", "mode": "Medium", "devicemark_composite": "84.6", "ci95_low": "81.6",
             "ci95_high": "87.3", "score_source": build.OURS, "speed_tok_s": "9.7",
             "speed_source": "Teraformer's app on an iPhone 17 Pro Max, Medium mode, short chats, preliminary (Table 1)",
             "speed_credit": build.OURS},
            {"model": "Masein model", "mode": "High", "devicemark_composite": "78.7",
             "ci95_low": "75.4", "ci95_high": "82.0", "score_source": build.OURS, "speed_tok_s": "",
             "speed_source": "", "speed_credit": ""},
            {"model": "Qwen3.5-4B", "mode": "thinking off", "devicemark_composite": "80.1", "ci95_low": "76.6",
             "ci95_high": "83.4", "score_source": build.OURS + " (replica)", "speed_tok_s": "10.8",
             "speed_source": dm, "speed_credit": build.DEVICEMARK},
            {"model": "Qwen3.5-4B (int8)", "mode": "", "devicemark_composite": "26.1", "ci95_low": "22.4",
             "ci95_high": "29.9", "score_source": build.DEVICEMARK, "speed_tok_s": "", "speed_source": "",
             "speed_credit": ""}]
    for r in rows:
        r.update(change.get(r["model"] + (f", {r['mode']}" if r["mode"] else ""), {}))
    f = tmp_path / "figure-1.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=build.SPEED_HEAD, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return f


def test_the_speed_chart_takes_its_scores_from_table_12_and_its_speeds_from_devicemark(paper, tmp_path):
    # Figure 1 plots our DeviceMark scores against speeds from two setups: each score must
    # be the one Table 12 prints; another model's speed must be DeviceMark's iPhone 17 Pro
    # number, as the repo's copy of its board gives it, and credited to DeviceMark (r25: no
    # Pipette numbers); each of the Masein model's must be the iPhone 17 Pro Max's short-chat
    # median the key numbers print for its mode — Instant's in its own row, Medium's,
    # preliminary, in that row's words — not another number of the row. High, not measured in
    # the app yet, is skipped
    speed_paper(paper)
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper), "--speeds", str(speeds_file(tmp_path))]) == 0
    f1 = rows(out / "figure-1.csv")
    assert f1[1][:7] == ["Masein model, Instant", "84.4", "81.4", "87.3", build.OURS, "11.4",
                         "Teraformer's app on an iPhone 17 Pro Max, Instant mode, short chats (Table 1)"]
    assert f1[2][:6] == ["Masein model, Medium", "84.6", "81.6", "87.3", build.OURS, "9.7"]
    assert f1[3][5] == "10.8" and f1[3][7] == build.DEVICEMARK
    assert f1[4][:5] == ["Qwen3.5-4B (int8)", "26.1", "22.4", "29.9", build.DEVICEMARK]
    assert len(f1) == 5
    assert not [c for r in f1 for c in r if "Pipette" in c or "Galaxy" in c]
    off, med, high, qwen = ("Masein model, Instant", "Masein model, Medium", "Masein model, High",
                            "Qwen3.5-4B, thinking off")
    for change, says in (({qwen: {"devicemark_composite": "80.4"}}, "Table 12 prints 80.1"),
                         ({qwen: {"speed_credit": "Pipette"}}, "must be DeviceMark's"),
                         ({qwen: {"speed_source": "Pipette leaderboard, Galaxy S26 Ultra"}}, "must be DeviceMark's"),
                         ({qwen: {"speed_tok_s": "15"}}, "plotted at 15 tokens/s; DeviceMark's board gives 10.8"),
                         ({qwen: {"speed_tok_s": ""}}, "Qwen3.5-4B, thinking off has no speed"),
                         ({off: {"speed_tok_s": "15.2"}}, "plotted at 15.2 tokens/s; the key numbers print 11.4 "
                          "for Instant mode on the iPhone 17 Pro Max"),
                         ({off: {"speed_tok_s": "7.3"}}, "plotted at 7.3 tokens/s"),
                         ({off: {"speed_source": "Teraformer's app on an iPhone 17 Pro Max"}}, "must name its mode"),
                         ({off: {"speed_source": "Teraformer's app, Instant mode"}}, "must name the phone"),
                         ({med: {"speed_tok_s": "12.6"}}, "plotted at 12.6 tokens/s; the key numbers print 9.7 "
                          "for Medium mode"),
                         ({med: {"speed_source": "Teraformer's app on an iPhone 17 Pro Max, Medium mode"}},
                          "call Medium mode's throughput preliminary"),
                         ({high: {"speed_tok_s": "13.3", "speed_source": "iPhone 17 Pro Max, High mode",
                                  "speed_credit": build.OURS}}, "Table 12 prints 45.2"),
                         ({high: {"devicemark_composite": "45.2", "ci95_low": "41.4", "ci95_high": "49.6",
                                  "speed_tok_s": "13.3", "speed_source": "iPhone 17 Pro Max, High mode",
                                  "speed_credit": build.OURS}},
                          "give High mode's short-chat throughput 0 times"),
                         ({"Qwen3.5-4B (int8)": {"speed_tok_s": "20"}}, "is a published row and has no speed")):
        with pytest.raises(SystemExit, match=re.escape(says)):
            build.main([str(tmp_path / "x"), str(paper), "--speeds", str(speeds_file(tmp_path, **change))])
    text = paper.read_text(encoding="utf-8")
    # r22: whose a row is, Table 12 says in words ("Run by"); since r18 no table has italics
    assert "*" not in "".join(ln for ln in text.splitlines() if "DeviceMark |" in ln)
    # each mode's throughput must be printed once, with the iPhone 17 Pro Max in short chats
    for old, new, says in (("), Instant mode |", "), thinking off |", "give Instant mode's short-chat throughput 0 times"),
                           ("iPhone 17 Pro Max (12 GB): 11.4 in short chats", "iPhone 17 Pro Max (12 GB): 11.4 at 1,000 tokens",
                            "gives no iPhone 17 Pro Max median in short chats"),
                           ("Medium mode, preliminary: 12.6 and 9.7 in short chats", "Medium mode, preliminary: 12.6 in short chats",
                            "give Medium mode 1 short-chat numbers for 2 phones"),
                           ("| Masein model, Medium |", "| Masein model, Medium mode |", "Masein model, Medium is not a row")):
        paper.write_text(text.replace(old, new, 1), encoding="utf-8")
        with pytest.raises(SystemExit, match=re.escape(says)):
            build.main([str(tmp_path / "y"), str(paper), "--speeds", str(speeds_file(tmp_path))])


def test_a_renumbered_paper_needs_no_change_to_the_build(paper, tmp_path):
    # round 19 put a new figure first and moved the rest down one: figures are known by
    # their captions, so each file follows the paper's new number and keeps its contents
    old = tmp_path / "old"
    build.main([str(old), str(paper)])
    text = re.sub(r"embedded content: Figure (\d+)\.",
                  lambda m: f"embedded content: Figure {int(m.group(1)) + 1}.", PAPER)
    paper.write_text(text.replace("| Qwen3.5-4B (int8) |", SPEED_ROW + "| Qwen3.5-4B (int8) |", 1)
                     + SPEED_FIG, encoding="utf-8")
    new = tmp_path / "new"
    assert build.main([str(new), str(paper), "--speeds", str(speeds_file(tmp_path))]) == 0
    assert sorted(p.name for p in new.glob("figure-*.csv")) == ["figure-1.csv", "figure-11.csv", "figure-4.csv",
                                                                "figure-5.csv", "figure-6.csv"]
    for n in (3, 4, 5):
        assert rows(new / f"figure-{n + 1}.csv") == rows(old / f"figure-{n}.csv")
    ix = {r[0]: r for r in rows(new / "index.csv")[1:]}
    assert ix["figure-4.csv"][1] == "Figure 4" and ix["figure-1.csv"][2] == "Fast and right."


def test_figure_ones_speeds_come_from_its_file_so_new_timings_need_one_rebuild(tmp_path):
    # new timings need one rebuild: a speed in figure-1.csv wins, and a row without one falls
    # back to DeviceMark's board (r25) or, for the Masein model, the value the script carries
    pytest.importorskip("matplotlib")
    figs = load("make_figures")
    f = tmp_path / "figure-1.csv"
    f.write_text("model,mode,speed_tok_s,speed_source\nQwen3.5-4B,thinking off,14.2,OnePlus 15, timed\n"
                 "LFM2.5-1.2B,,,\n".replace("OnePlus 15, timed", '"OnePlus 15, timed"'), encoding="utf-8")
    got = figs.speeds(tmp_path)
    assert got["Qwen3.5-4B, thinking off"] == ("14.2", "OnePlus 15, timed")
    assert got["LFM2.5-1.2B"][0] == "45.5" and got["LFM2.5-1.2B"][1].startswith("DeviceMark leaderboard, iPhone 17 Pro")
    assert got["Masein model, Instant"][0] == "11.4" and "iPhone 17 Pro Max" in got["Masein model, Instant"][1]
    # Medium's preliminary 9.7; High has no speed until the app's own number comes
    assert got["Masein model, Medium"][0] == "9.7" and "preliminary" in got["Masein model, Medium"][1]
    assert not [x for v in got.values() for x in v if "Pipette" in x or "Galaxy" in x]
    assert "Masein model, High" not in got
    # r22: a speed under a name no point has stops the build, instead of being dropped
    f.write_text("model,mode,speed_tok_s,speed_source\nMasein model,thinking off,15.2,key numbers\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no point of Figure 1 is Masein model, thinking off"):
        figs.speeds(tmp_path)


def test_a_tbd_stops_the_build_and_trial_lists_the_files_that_carry_one(paper, tmp_path, capsys):
    paper.write_text(PAPER.replace("| 7 of 792 | 5 of 792 |", "| 7 of 792 | \\[TBD K1: count\\] |"),
                     encoding="utf-8")
    out = tmp_path / "out"
    assert build.main([str(out), str(paper)]) == 1
    assert not out.exists()
    assert build.main([str(out), str(paper), "--trial"]) == 0
    assert "files that carry a TBD from the paper: table-8" in capsys.readouterr().out
    assert rows(out / "table-8.csv")[1][2] == "[TBD K1: count]"


def test_a_number_split_from_the_wrong_cell_is_caught():
    t = {"rows": [["GPQA Diamond", "82.7 ± 2.2"]]}
    build.check_table("table-x", t, [["GPQA Diamond", "82.7", "2.2"]])
    with pytest.raises(SystemExit, match="82.8 is not printed"):
        build.check_table("table-x", t, [["GPQA Diamond", "82.8", "2.2"]])


def folder(tmp_path, readme="A report.\n"):
    d = tmp_path / "data"
    (d / "logs").mkdir(parents=True)
    (d / "README.md").write_text(readme, encoding="utf-8")
    (d / "logs" / "ifeval-thinking-on.jsonl").write_text(json.dumps(
        {"benchmark": "IFEval", "question_id": "1", "prompt": "Explain it step by step.",
         "final_answer": "Step one.", "grade": {"passed_strict": True}}) + "\n", encoding="utf-8")
    priv = tmp_path / "private.json"
    priv.write_text(json.dumps({"people": [r"(?i)\bjordan\b"]}), encoding="utf-8")
    return d, priv


def test_the_scan_fails_our_own_text_and_only_reports_the_benchmarks(tmp_path, capsys):
    d, priv = folder(tmp_path)
    assert scan.main([str(d), "--private", str(priv)]) == 0
    out = capsys.readouterr().out
    assert "step or checkpoint: 2 in logs/ifeval-thinking-on.jsonl, field prompt" in out
    assert "CLEAN" in out
    (d / "README.md").write_text("Built from the step2388 checkpoint on k6, thanks to Jordan.\n",
                                 encoding="utf-8")
    assert scan.main([str(d), "--private", str(priv)]) == 1
    out = capsys.readouterr().out
    for term in ("step or checkpoint: 2", "k3, k6, k3k6: 1", "people: 1"):
        assert term in out
    # a model's name isn't an internal one
    (d / "README.md").write_text("Kimi K3 (max) and Q4_K_M files.\n", encoding="utf-8")
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps([{"file": "README.md", "row": "", "column": "", "term": "Q4",
                                  "label": "Q4_K_M", "count": 1}]), encoding="utf-8")
    assert scan.main([str(d), "--private", str(priv), "--allow", str(allow)]) == 0


@pytest.mark.parametrize("text, term", [
    ("It ran with LOOKAHEAD_MOE_ROUTER_NORM set.", "runtime switches"),
    ("LLAMA_MOE_ROUTE_MODE was local.", "runtime switches"),
    ("GGML_CPU_DISABLE_FUSION held.", "runtime switches"),
    ("launched as llama-server with its defaults", "server argv"),
    ("flags: --jinja --metrics", "server argv"),
    ("all layers on the GPU (-ngl 99)", "server argv"),
    ("-ctk q8_0 -ctv q8_0", "server argv"),
    ("env CUDA_VISIBLE_DEVICES=0", "launch env"),
    ("the row served/step2388-k3k6-2611-medium", "served model ids"),
    ("its folder served__Qwen3.6-35B-A3B-Q4-lda-2610", "served model ids"),
])
def test_runtime_details_are_internal_terms(tmp_path, capsys, text, term):
    # r25 (masein and Roohi, 14 Nov): the runtime's switches, the server's command line and
    # launch environment, and our serving ids stay in internal docs; the report's own words
    # for them ("lookahead routing on", "flash attention on") are fine
    d, priv = folder(tmp_path, "Lookahead routing on; flash attention on; all layers on the GPU.\n")
    assert scan.main([str(d), "--private", str(priv)]) == 0
    (d / "README.md").write_text(text + "\n", encoding="utf-8")
    assert scan.main([str(d), "--private", str(priv)]) == 1
    out = capsys.readouterr().out
    assert f"{term}: 1" in out or f"{term}: 2" in out, out


def test_the_upload_sends_nothing_with_a_tbd_or_without_the_private_terms(tmp_path, capsys):
    d, priv = folder(tmp_path, "Counts: [TBD from the logs export]\n")
    args = ["--dry-run", "--folder", str(d), "--private", str(priv), "--allow", str(tmp_path / "none")]
    assert upload.main(args) == 1
    assert "stopped: 1 files hold a TBD" in capsys.readouterr().out
    assert upload.main(args + ["--trial"]) == 0
    assert upload.main(["--dry-run", "--folder", str(d), "--private", str(tmp_path / "gone.json"),
                        "--trial"]) == 1
    assert "the private terms" in capsys.readouterr().out


def test_a_line_separator_inside_a_jsonl_string_is_part_of_its_record(tmp_path):
    """U+2028 is legal inside a JSON string; a model's answer can hold one, and the scan
    reads a JSONL file by its newlines alone"""
    d, priv = folder(tmp_path)
    (d / "logs" / "mmlu-pro-thinking-on.jsonl").write_text(json.dumps(
        {"benchmark": "MMLU-Pro", "question_id": "7", "final_answer": "one\u2028two\u0085three",
         "grade": {"right": True}}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert scan.main([str(d), "--private", str(priv)]) == 0


# ---------------------------------------------------------------------------
# r29 (masein, 16 Nov): LaTeX numbers the tables. Section 3 is Kevin's own file, whose five
# tables the md's text calls P1 to P5 (printed 1 to 5); the md's own Section 3 stands in for it
# and prints nowhere; the md's Tables 4 on print two higher. The files go by the printed numbers
# ---------------------------------------------------------------------------

R29_SPEED = ("| Token generation throughput on a phone (decode, tokens/s), Instant mode | OnePlus 15 (16 GB): "
             "15.2 in short chats, 7.3 with 14,000 tokens of context; iPhone 17 Pro Max (12 GB): 11.4 in short "
             "chats, 12.4 with 14,000 tokens of context. In short chats, Medium mode: 12.6 and 9.2; High mode, "
             "thinking included: 10.2 and 8.5 | No verdict (Section 3) |")
STAND_IN = """## 3. On-device inference

**Table 1.** Short-chat performance; provisional.

| Phone | Decode |
| --- | --- |
| OnePlus 15 | 15.2 |

**Table 2.** Context-length scaling; provisional.

| Context | Decode |
| --- | --- |
| 2,000 | 14.0 |

**Table 3.** Accuracy kept at a limit.

| Limit | Score |
| --- | --- |
| 2,048 | 70.1 |
"""
R29 = f"""# A report: its title

## Abstract

The phones in Table P1, and all of Kevin's in Tables P1 to P5.

## Key numbers

| Measure | Masein model (measured here) | How it compares |
| --- | --- | --- |
| GPQA Diamond (%), High / Medium / Instant | 82.7 ± 2.2 / 80.1 ± 2.4 / 75.3 ± 2.4 | High: level with o1 (76.8) |
{R29_SPEED}

{STAND_IN}
## 4. Capabilities

**Table 4.** The protocol per benchmark, its scores in Table 5.

| Benchmark | Limit |
| --- | --- |
| GPQA Diamond | 98,304 |

**Table 5.** The scores, with the limits of Table 4 and Tables 5, F1 and F3 beside them.

| Benchmark | Score |
| --- | --- |
| GPQA Diamond | 82.7 |

**Table 6.** List prices, as in Tables 4 and 5.

| Model | Price |
| --- | --- |
| o1 | 15.0 |

## Appendix F

**Table F1.** GPQA Diamond, as Table 5 has it.

| Model | Score |
| --- | --- |
| o1 | 76.8 |
"""


def r29(tmp_path, text=R29):
    p = tmp_path / "paper-r29.md"
    p.write_text(text, encoding="utf-8")
    return p


def test_r29_the_files_and_their_references_go_by_the_numbers_latex_prints(tmp_path, capsys):
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(r29(tmp_path)), "--trial"]) == 0
    said = capsys.readouterr().out
    assert "the md's Table 1, Table 2, Table 3 there stand in for it" in said
    assert "Kevin's Tables P1 to P5 (printed 1 to 5) are read from his own file, and none was given" in said
    got = sorted(p.name for p in out.iterdir())
    assert got == ["index.csv", "key-numbers.csv", "table-6.csv", "table-7.csv", "table-8.csv", "table-f1.csv"]
    ix = {r[0]: r[1:] for r in rows(out / "index.csv")}
    assert ix["table-6.csv"] == ["Table 6", "The protocol per benchmark, its scores in Table 7."]
    assert ix["table-7.csv"] == ["Table 7", "The scores, with the limits of Table 6 and Tables 7, F1 and F3 "
                                            "beside them."]
    assert ix["table-8.csv"] == ["Table 8", "List prices, as in Tables 6 and 7."]
    assert ix["table-f1.csv"] == ["Table F1", "GPQA Diamond, as Table 7 has it."]
    # a release build stops: the printed Tables 1 to 5 have no file
    assert build.main([str(tmp_path / "release"), str(r29(tmp_path))]) == 1
    assert "stopped: nothing written. Give his file (on-the-phone.tex, from Overleaf) with --phone-tex" in \
        capsys.readouterr().out
    assert not (tmp_path / "release").exists()


def tex_table(label: str, caption: str, spec: str, head: str, rows: list[str]) -> str:
    return ("\\begin{table}[!htbp]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\footnotesize\n\\begin{tabular}{" + spec + "}\n\\toprule\n" + head + "\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


GROUPS = ("& \\multicolumn{3}{c}{OnePlus 15} & \\multicolumn{3}{c}{iPhone 17 Pro Max} \\\\\n"
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
          "Context & Prefill & TTFT (s) & Decode & Prefill & TTFT (s) & Decode \\\\")
HIS = {
    "tab:phone-modes": tex_table(
        "tab:phone-modes", "Short-chat inference performance. Values are medians of five runs.", "llrrrr",
        "Device & Mode & Prefill & Decode & TTFT & TTFA \\\\\n& & (tok/s) & (tok/s) & (s) & (s) \\\\",
        ["OnePlus 15\n & Instant & 53.6 & 15.2 & 14.69 & 14.69 \\\\",
         " & Medium  & 32.6 & 12.6 & 23.82 & 23.82 \\\\",
         " & High    & 23.9 & 10.2 & 42.82 & 145.5 \\\\", "\\midrule",
         "iPhone 17 Pro Max\n & Instant & 43.7 & 11.4 & 17.68 & 17.68 \\\\",
         " & Medium  & 21.2 & 9.2  & 36.73 & 36.73 \\\\",
         " & High    & 23.0 & 8.5  & 43.70 & 158.2 \\\\"]),
    "tab:phone-memory": tex_table(
        "tab:phone-memory", "Runtime memory by mode, in GB. Counters differ by system.", "llrrr",
        "Device & Mode & Reported & Incl. mapped & Available \\\\",
        ["OnePlus 15\n & Instant & 6.71 & -- & 4.00 \\\\", "iPhone 17 Pro Max\n & Instant & 2.86 & 5.06 & 3.59 \\\\"]),
    **{f"tab:context-{m}": tex_table(f"tab:context-{m}", f"{m.title()}-mode context scaling. Medians.",
                                    "lrrrrrr", GROUPS, ["2K  & 51.8 & 39.7  & 13.1 & 54.5 & 37.8  & 12.3 \\\\",
                                                        "14K & 16.9 & 828.4 & 7.3  & 40.8 & 342.6 & 12.4 \\\\"])
       for m in ("instant", "medium", "high")},
}


def his_file(tmp_path, order=tuple(HIS), change=None) -> Path:
    f = tmp_path / "on-the-phone.tex"
    text = "\\subsection{3. On-device inference}\n% a comment & \\\\ never read\n" + "\n".join(HIS[k] for k in order)
    for a, b in (change or {}).items():
        assert a in text, a
        text = text.replace(a, b)
    f.write_text(text, encoding="utf-8")
    return f


def test_r29_kevins_tables_come_from_his_file_as_tables_1_to_5(tmp_path):
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(r29(tmp_path, R29.replace(R29_SPEED, R29_SPEED.replace(" (Section 3)", "")))),
                       "--phone-tex", str(his_file(tmp_path))]) == 0
    assert sorted(p.name for p in out.glob("table-*.csv")) == sorted(
        [f"table-{k}.csv" for k in range(1, 9)] + ["table-f1.csv"])
    t1 = rows(out / "table-1.csv")
    assert t1[0] == ["Device", "Mode", "Prefill (tok/s)", "Decode (tok/s)", "TTFT (s)", "TTFA (s)"]
    assert t1[4] == ["iPhone 17 Pro Max", "Instant", "43.7", "11.4", "17.68", "17.68"]   # said once a group
    assert rows(out / "table-2.csv")[1] == ["OnePlus 15", "Instant", "6.71", "–", "4.00"]
    assert rows(out / "table-3.csv")[0] == ["Context", "OnePlus 15, Prefill", "OnePlus 15, TTFT (s)",
                                           "OnePlus 15, Decode", "iPhone 17 Pro Max, Prefill",
                                           "iPhone 17 Pro Max, TTFT (s)", "iPhone 17 Pro Max, Decode"]
    ix = {r[0]: r[1:] for r in rows(out / "index.csv")}
    assert ix["table-1.csv"] == ["Table 1", "Short-chat inference performance."]
    assert ix["table-5.csv"] == ["Table 5", "High-mode context scaling."]
    assert ix["table-6.csv"][0] == "Table 6"


def test_r29_his_tables_print_in_his_files_order(tmp_path):
    # LaTeX numbers his tables as his file has them; the text's P2 is his memory table wherever it is
    paper = r29(tmp_path, R29.replace("its scores in Table 5", "its memory in Table P2"))
    order = ("tab:phone-modes", "tab:context-instant", "tab:phone-memory", "tab:context-medium", "tab:context-high")
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper), "--phone-tex", str(his_file(tmp_path, order))]) == 0
    ix = {r[0]: r[1:] for r in rows(out / "index.csv")}
    assert ix["table-3.csv"] == ["Table 3", "Runtime memory by mode, in GB."]
    assert ix["table-6.csv"] == ["Table 6", "The protocol per benchmark, its memory in Table 3."]


@pytest.mark.parametrize("change,says", [
    ({"\\label{tab:phone-memory}": "\\label{tab:memory}"}, "Kevin's file has no tab:phone-memory"),
    ({" & Medium  & 32.6 & 12.6": " & Medium  & n/a & 12.6"}, "'Prefill (tok/s)' has 'n/a', not a number"),
    ({" & High    & 23.9 & 10.2 & 42.82 & 145.5": " & High    & 23.9 & 10.2 & 42.82"}, "has 5 cells for 6 columns"),
    ({"\\label{tab:context-high}\n": ""}, "a table without its label"),
    # the key numbers' iPhone speed is the one Figure 1 credits to his table
    ({" & Medium  & 21.2 & 9.2 ": " & Medium  & 21.2 & 9.4 "}, "Kevin's short-chat table has ['9.4']"),
])
def test_r29_a_label_missing_or_a_row_that_doesnt_parse_stops_the_build(tmp_path, change, says):
    with pytest.raises(SystemExit, match=re.escape(says)):
        build.main([str(tmp_path / "out"), str(r29(tmp_path)), "--phone-tex", str(his_file(tmp_path, change=change)),
                    "--trial"])
    assert not (tmp_path / "out").exists()


def test_r29_a_gap_in_the_main_text_tables_stops_the_build(tmp_path):
    with pytest.raises(SystemExit, match="so they must run with no gap"):
        build.main([str(tmp_path / "out"), str(r29(tmp_path, R29.replace("**Table 6.**", "**Table 7.**"))),
                    "--trial"])


def test_r29_a_paper_with_no_table_p_keeps_its_own_numbers(paper, tmp_path):
    # the fixture above names no "Table P": nothing is renumbered and nothing is skipped
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper)]) == 0
    assert (out / "table-5.csv").exists() and (out / "table-12.csv").exists()


def test_r29_the_speed_row_gives_each_mode_its_iphone_number():
    lines = R29.split("\n")
    p = build.Paper(build.tables(lines))
    assert build.masein_speed(p, "Instant") == ("11.4", False)
    assert build.masein_speed(p, "Medium") == ("9.2", False)
    assert build.masein_speed(p, "High") == ("8.5", False)
    # r25's wording still reads, preliminary with it
    old = build.Paper(build.tables(PAPER.split("\n")))
    assert build.masein_speed(old, "Medium") == ("9.7", True)


def test_r29_figure_one_cant_call_a_final_speed_preliminary(paper, tmp_path):
    speed_paper(paper)
    text = paper.read_text(encoding="utf-8")
    old_row = next(x for x in text.split("\n") if x.startswith("| Token generation throughput"))
    paper.write_text(text.replace(old_row, R29_SPEED.replace(" (Section 3)", "")), encoding="utf-8")
    final = "Teraformer's app on an iPhone 17 Pro Max, Medium mode, short chats (Table 1)"
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper), "--speeds", str(speeds_file(
        tmp_path, **{"Masein model, Medium": {"speed_tok_s": "9.2", "speed_source": final}}))]) == 0
    assert rows(out / "figure-1.csv")[2][:6] == ["Masein model, Medium", "84.6", "81.6", "87.3", build.OURS, "9.2"]
    with pytest.raises(SystemExit, match="calls it preliminary, and the key numbers don't"):
        build.main([str(tmp_path / "again"), str(paper), "--speeds", str(speeds_file(
            tmp_path, **{"Masein model, Medium": {"speed_tok_s": "9.2"}}))])


# ---------------------------------------------------------------------------
# r29 (masein, 16 Nov): High at the app's limits — H1 re-cut at 2,048 tokens in total — is a row
# of Table 12 and a point of Figure 1 (at High's 8.5 on the iPhone), traced to the re-cut's own
# result (recut-result.json) and the run records beside it
# ---------------------------------------------------------------------------

T12_FULL = """**Table 12.** DeviceMark, all 29 rows. Rows run by DeviceMark are its published leaderboard.

| Model | Run by | Composite \\[95% range\\] | IFEval (DeviceMark's measure) | MMLU-Pro (draw) | MATH (draw) | Answered |
| --- | --- | --- | --- | --- | --- | --- |
| **Masein model, Instant** | **us (replica)** | **84.4 \\[81.4, 87.3\\]** | **86.4** | **74.0** | **93.0** | **97%** |
| **Masein model, Medium** | **us (replica)** | **84.6 \\[81.6, 87.3\\]** | **85.3** | **74.5** | **94.0** | **97%** |
| **Masein model, High** | **us (replica)** | **45.2 \\[41.4, 49.6\\]** | **79.2** | **26.5** | **30.0** | **56%** |
| **Masein model, High, at the app's limits** | **us (replica)** | **72.4 \\[68.5, 76.2\\]** | **73.2** | **68.9** | **75.0** | **77%** |
| Qwen3.5-4B, thinking off | us (replica) | 80.1 \\[76.6, 83.4\\] | 85.9 | 69.4 | 85.0 | 94% |
| Qwen3.5-4B (int8) | DeviceMark | 26.1 \\[22.4, 29.9\\] | 30.0 | 20.0 | 28.3 | 60% |
"""
APP_NAME = "Masein model, High, at the app's limits"


def app_paper(tmp_path) -> Path:
    a = PAPER.index("**Table 12.**")
    b = PAPER.index("**Table F1.**")
    text = PAPER[:a] + T12_FULL + "\n" + PAPER[b:] + SPEED_FIG
    old_row = next(x for x in text.split("\n") if x.startswith("| Token generation throughput"))
    p = tmp_path / "paper-app.md"
    p.write_text(text.replace(old_row, R29_SPEED.replace(" (Section 3)", "")), encoding="utf-8")
    return p


def recut_result(tmp_path, **change) -> Path:
    d = tmp_path / "h1-recut"
    d.mkdir(exist_ok=True)
    recs = {}
    for what, name in (("answers", "recut-answers.jsonl"), ("items", "recut-items.jsonl")):
        data = b'{"bench": "math", "key": "a"}\n{"bench": "ifeval", "key": "1"}\n'
        (d / name).write_bytes(data)
        recs[what] = {"file": name, "n": 2, "sha256": hashlib.sha256(data).hexdigest()}
    res = {"row": APP_NAME, "records": recs,
           "after": {"composite": {"value": 0.7237, "ci": [0.6853, 0.7618]}, "answered_pct": 0.7685,
                     "benches": {"ifeval": {"acc": 0.7324}, "mmlu_pro": {"acc": 0.6888}, "math": {"acc": 0.75}}}}
    res.update(change)
    f = d / "recut-result.json"
    f.write_text(json.dumps(res), encoding="utf-8")
    return f


def app_speeds(tmp_path):
    final = "Teraformer's app on an iPhone 17 Pro Max, Medium mode, short chats (Table 1)"
    return speeds_file(tmp_path, **{
        "Masein model, Medium": {"speed_tok_s": "9.2", "speed_source": final},
        "Masein model, High": {"mode": "High, at the app's limits", "devicemark_composite": "72.4",
                               "ci95_low": "68.5", "ci95_high": "76.2", "speed_tok_s": "8.5",
                               "speed_source": "Teraformer's app on an iPhone 17 Pro Max, High mode, thinking "
                                               "included, short chats (Table 1)", "speed_credit": build.OURS}})


def test_r29_the_app_limits_row_is_traced_to_its_recut_and_plotted_at_highs_speed(tmp_path):
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(app_paper(tmp_path)), "--speeds", str(app_speeds(tmp_path)),
                       "--app-limits", str(recut_result(tmp_path))]) == 0
    f1 = {r[0]: r for r in rows(out / "figure-1.csv")}
    assert f1[APP_NAME][1:6] == ["72.4", "68.5", "76.2", build.OURS, "8.5"]
    t12 = {r[0]: r for r in rows(out / "table-12.csv")}
    assert APP_NAME in t12


@pytest.mark.parametrize("how,says", [
    ("no file", "give its re-cut's result (recut-result.json) with --app-limits"),
    ("answered", "Answered prints 78%; the re-cut (recut-result.json) gives 77%"),
    ("composite", "prints 72.4 [68.5, 76.2]; the re-cut (recut-result.json) gives 72.5 [68.5, 76.2]"),
    ("records", "its items records (recut-items.jsonl) are not the ones it names"),
    ("row", "is the result for 'Masein model, High', not"),
])
def test_r29_an_app_limits_row_that_isnt_its_recuts_stops_the_build(tmp_path, how, says):
    paper = app_paper(tmp_path)
    res = recut_result(tmp_path)
    if how == "answered":
        paper.write_text(paper.read_text(encoding="utf-8").replace("**75.0** | **77%**", "**75.0** | **78%**"),
                         encoding="utf-8")
    if how == "composite":
        res = recut_result(tmp_path, after={**json.loads(res.read_text())["after"],
                                            "composite": {"value": 0.7246, "ci": [0.6853, 0.7618]}})
    if how == "records":
        (res.parent / "recut-items.jsonl").write_bytes(b'{"bench": "math", "key": "b"}\n{"bench": "ifeval", "key": "1"}\n')
    if how == "row":
        res = recut_result(tmp_path, row="Masein model, High")
    args = [str(tmp_path / "out"), str(paper), "--speeds", str(app_speeds(tmp_path))]
    if how != "no file":
        args += ["--app-limits", str(res)]
    with pytest.raises(SystemExit, match=re.escape(says)):
        build.main(args)


def test_r29_the_gpqa_lanes_carry_medium_and_instant_as_reference_lines_with_no_verdict(paper, tmp_path):
    # Figure 7's data: High's line (its verdicts are the lanes), then Medium and Instant from the
    # scores table, no verdict, with their source — the md's Table 5 and Section 4.4
    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "figure-extras.json").write_text(json.dumps({"figure-6 labelled": ["Grok 3 mini"]}), encoding="utf-8")
    paper.write_text(PAPER + "\n&#91;embedded content: Figure 6. GPQA Diamond's lanes. Each dot is one model's "
                     "best-scoring Epoch run, in the lane of its verdict. Source: Table F1.\\]\n", encoding="utf-8")
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper), "--figures", str(figs)]) == 0
    f = rows(out / "figure-6.csv")
    assert f[1][:3] == ["Masein model, High", "82.7", "2.2"] and f[1][3] == ""
    assert f[2] == ["Masein model, Medium (a reference line, no verdict)", "80.1", "2.4", "", "", "", "yes",
                    f"{build.OURS} (Table 5, Section 4.4)"]
    assert f[3][:3] == ["Masein model, Instant (a reference line, no verdict)", "75.3", "2.4"]
    assert f[4][0].startswith("Grok 3 mini") and f[4][3] == "level with"


def test_r31_the_out_of_room_caption_reads_in_high_mode_wording(paper, tmp_path):
    # r31: "…at the final limit, in High mode (thinking on), on GPQA Diamond and IFBench; …" names
    # the same benchmarks as "…at the final limit, thinking on, on GPQA Diamond and IFBench; …"
    old = "at the final limit, thinking on, on GPQA Diamond and IFBench;"
    assert PAPER.count(old) == 1
    assert build.main([str(tmp_path / "old"), str(paper)]) == 0
    paper.write_text(PAPER.replace(old, "at the final limit, in High mode (thinking on), on GPQA Diamond and "
                                        "IFBench;"), encoding="utf-8")
    assert build.main([str(tmp_path / "new"), str(paper)]) == 0
    assert rows(tmp_path / "new" / "figure-5.csv") == rows(tmp_path / "old" / "figure-5.csv")
    assert sorted(r[0] for r in rows(tmp_path / "new" / "figure-5.csv")[1:]) == ["GPQA Diamond", "IFBench"]


# ---------------------------------------------------------------------------
# r31 (masein, 16 Nov): the unnumbered figure under the abstract, from the figure script's
# figure-abstract.csv beside figure-1.csv, each row traced to Table 5 or Table F1
# ---------------------------------------------------------------------------

ABS_ROWS = [["Masein model, High", "82.7", "2.2", "78.4", "87.0", "Table 5", build.OURS],
            ["Masein model, Medium", "80.1", "2.4", "75.4", "84.8", "Table 5", build.OURS],
            ["Grok 3 mini (low)", "76.3", "3.0", "70.4", "82.2", "Table F1",
             "Epoch AI Benchmarking Hub, CC BY 4.0, imported 7 Nov 2026"],
            ["Masein model, Instant", "75.3", "2.4", "70.6", "80.0", "Table 5", build.OURS]]


def abstract_file(tmp_path, rows=ABS_ROWS) -> Path:
    f = tmp_path / "figure-abstract.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(build.ABSTRACT_HEAD)
        w.writerows(rows)
    return tmp_path / "figure-1.csv"                     # the build looks beside figure-1.csv


def test_r31_the_figure_under_the_abstract_is_traced_to_table_5_and_table_f1(paper, tmp_path):
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper), "--speeds", str(abstract_file(tmp_path))]) == 0
    got = rows(out / "figure-abstract.csv")
    assert got[0] == build.ABSTRACT_HEAD
    assert got[3] == ["Grok 3 mini (low)", "76.3", "3.0", "70.4", "82.2", "Table F1", build.EPOCH]
    assert [r[0] for r in got[1:]] == [r[0] for r in ABS_ROWS]
    ix = {r[0]: r[1:] for r in rows(out / "index.csv")}
    assert ix["figure-abstract.csv"][0] == "The figure under the abstract"


@pytest.mark.parametrize("change,says", [
    ({0: {1: "82.8"}}, "Masein model, High is drawn at 82.8 ± 2.2; Table 5 prints 82.7 ± 2.2"),
    ({2: {2: "2.9"}}, "Grok 3 mini (low) is drawn at 76.3 ± 2.9; Table F1 prints 76.3 ± 3.0"),
    ({2: {0: "Grok 4 (high)"}}, "Grok 4 (high) is neither a mode in the scores table nor a row of Table F1"),
    ({3: {4: "80.2"}}, "Masein model, Instant's bar is 70.6 to 80.2, not ±1.96 SE"),
    ({1: {5: "Table F1"}}, "Masein model, Medium says 'Table F1'"),
    ({"swap": (1, 2)}, "is out of order (highest first)"),
])
def test_r31_an_abstract_figure_row_that_isnt_the_reports_stops_the_build(paper, tmp_path, change, says):
    rs = [list(r) for r in ABS_ROWS]
    for k, v in change.items():
        if k == "swap":
            rs[v[0]], rs[v[1]] = rs[v[1]], rs[v[0]]
        else:
            for col, val in v.items():
                rs[k][col] = val
    with pytest.raises(SystemExit, match=re.escape(says)):
        build.main([str(tmp_path / "out"), str(paper), "--speeds", str(abstract_file(tmp_path, rs))])


def test_r31_with_no_abstract_figure_file_the_rest_builds_and_it_says_so(paper, tmp_path, capsys):
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper), "--speeds", str(tmp_path / "figure-1.csv")]) == 0
    assert "no figure-abstract.csv beside figure-1.csv" in capsys.readouterr().out
    assert not (out / "figure-abstract.csv").exists() and (out / "table-5.csv").exists()
