"""The report's public data repository: paper-data/ built from the paper itself, the scan
for internal terms, and the upload's checks before anything is sent (no network here)."""
import csv
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
