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
| GPQA Diamond (%), thinking on / off | 82.7 ± 2.2 / 75.3 ± 2.4 | Thinking on: level with o1 (*76.8*) |
| Writing speed on a phone (decode, tokens/s), thinking off | OnePlus 15: 16.0 (15.0 to 16.6) | No verdict |

**Table 5.** The Masein model, measured here, in both modes: score (%) ± standard error.

| Benchmark | Questions × runs | Thinking on (%) | Ran out, thinking on |
| --- | --- | --- | --- |
| GPQA Diamond | 198 × 4 | 82.7 ± 2.2 | 7 of 792 |
| IFBench | 300 × 1 | 63.7 ± 2.8 | 15 of 300 |

**Table 7.** How the thinking goes, thinking on: finished answers only.

| Benchmark | Finished answers, right / wrong | Mean thinking tokens, right / wrong |
| --- | --- | --- |
| GPQA Diamond | 655 / 130 | 15,649 / 24,995 |

**Table 8.** Answers that ran out of room at the final limit.

| Benchmark | Ran out of room | Ended in a loop |
| --- | --- | --- |
| GPQA Diamond | 7 of 792 | 5 of 792 |
| IFBench | 15 of 300 | 12 of 300 |

**Table 9.** Against the frontier models. Italic cells are Epoch AI's runs.

| Benchmark | Masein model | Gemini 4 Argon (high) |
| --- | --- | --- |
| GPQA Diamond | **82.7 ± 2.2** | *96.4 ± 1.2* |

**Table 10.** Against five models (Section 4.3). Italic models' scores are Epoch AI's runs.

| Model | GPQA Diamond |
| --- | --- |
| *o1* | level, +5.9 (−1.4 to +13.3) |
| *Claude Haiku 4.5* | higher, +11.5 (+3.8 to +19.2) ‡ |

**Table 12.** DeviceMark, all 27 rows. Italic rows are DeviceMark's published leaderboard.

| Model | Run by | Composite \\[95% range\\] |
| --- | --- | --- |
| **Masein model, thinking off** | **us (replica)** | **84.4 \\[81.4, 87.3\\]** |
| Qwen3.5-4B (int8) | *DeviceMark* | *26.1 \\[22.4, 29.9\\]* |

**Table F1.** GPQA Diamond, 2 models. Italic scores are Epoch AI's runs.

| Model (setting) | Epoch score | Verdict | Gap (95% range) |
| --- | --- | --- | --- |
| **Masein model (measured here)** | **82.7 ± 2.2** | – | – |
| Grok 3 mini (low) † | *76.3 ± 3.0* | level with ‡ | +6.4 (−0.9 to +13.7) |

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
                          "table-9.csv", "table-10.csv", "table-12.csv", "table-f1.csv",
                          "figure-3.csv", "figure-4.csv", "figure-5.csv", "figure-10.csv",
                          "index.csv"])
    # the cells as printed: a score and its SE apart, a "right / wrong" pair apart,
    # thousands separators gone, nothing re-rounded
    t5 = rows(out / "table-5.csv")
    assert t5[0][2:4] == ["Thinking on (%)", "Thinking on (%), SE"]
    assert t5[1][2:5] == ["82.7", "2.2", "7 of 792"]
    t7 = rows(out / "table-7.csv")
    assert t7[0][1:3] == ["Finished answers, right", "Finished answers, wrong"]
    assert t7[1] == ["GPQA Diamond", "655", "130", "15649", "24995"]
    # key numbers: one row per mode, and a row that isn't a score stays as printed
    k = rows(out / "key-numbers.csv")
    assert [r[1:4] for r in k[1:3]] == [["thinking on", "82.7", "2.2"], ["thinking off", "75.3", "2.4"]]
    assert k[3][2] == "OnePlus 15: 16.0 (15.0 to 16.6)"
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
    t12 = rows(out / "table-12.csv")
    assert [r[-1] for r in t12[1:]] == ["measured by Teraformer", build.DEVICEMARK]
    assert t12[0][2:5] == ["Composite", "Composite, 95% low", "Composite, 95% high"]
    f1 = rows(out / "table-f1.csv")
    assert f1[2] == ["Grok 3 mini (low)", "yes", "76.3", "3.0", "level with", "yes", "+6.4",
                     "-0.9", "+13.7", build.EPOCH]
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
    assert rows(out / "figure-10.csv")[2][2:] == ["26.1", "22.4", "29.9", build.DEVICEMARK]


def test_a_caption_that_disagrees_with_its_table_stops_the_build(paper, tmp_path):
    paper.write_text(PAPER.replace("Out of room: 5.0%", "Out of room: 5.3%"), encoding="utf-8")
    with pytest.raises(SystemExit, match="says 5.3% for IFBench"):
        build.main([str(tmp_path / "out"), str(paper)])


def test_a_figure_the_script_doesnt_know_stops_it(paper, tmp_path):
    paper.write_text(PAPER + "\n&#91;embedded content: Figure 11. Something new. A pie.\\]\n",
                     encoding="utf-8")
    with pytest.raises(SystemExit, match="Figure 11 is new"):
        build.main([str(tmp_path / "out"), str(paper)])


SPEED_ROW = "| Qwen3.5-4B, thinking off | us (replica) | 80.1 \\[76.6, 83.4\\] |\n"
SPEED_FIG = ("\n&#91;embedded content: Figure 1. Fast and right. DeviceMark composite against writing speed "
             "on a phone (Table 12). Source: Table 12.\\]\n")


def speed_paper(paper):
    """the fixture with a replica row for Qwen3.5-4B and the speed chart as Figure 1"""
    text = PAPER.replace("| Qwen3.5-4B (int8) |", SPEED_ROW + "| Qwen3.5-4B (int8) |", 1) + SPEED_FIG
    paper.write_text(text, encoding="utf-8")
    return paper


def speeds_file(tmp_path, **change):
    """figure-1.csv as the figure script writes it; change={model: {column: value}}"""
    pip = "Pipette leaderboard, Galaxy S26 Ultra (12 GB): Qwen3.5-4B q4_k_m"
    rows = [{"model": "Masein model", "mode": "thinking off", "devicemark_composite": "84.4", "ci95_low": "81.4",
             "ci95_high": "87.3", "score_source": build.OURS, "speed_tok_s": "16.0",
             "speed_source": "Teraformer's app on a OnePlus 15 (Table 1)", "speed_credit": build.OURS},
            {"model": "Qwen3.5-4B", "mode": "thinking off", "devicemark_composite": "80.1", "ci95_low": "76.6",
             "ci95_high": "83.4", "score_source": build.OURS + " (replica)", "speed_tok_s": "15",
             "speed_source": pip, "speed_credit": build.PIPETTE},
            {"model": "Qwen3.5-4B (int8)", "mode": "", "devicemark_composite": "26.1", "ci95_low": "22.4",
             "ci95_high": "29.9", "score_source": build.DEVICEMARK, "speed_tok_s": "", "speed_source": "",
             "speed_credit": ""}]
    for r in rows:
        r.update(change.get(r["model"], {}))
    f = tmp_path / "figure-1.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=build.SPEED_HEAD, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return f


def test_the_speed_chart_takes_its_scores_from_table_12_and_keeps_pipettes_notice(paper, tmp_path):
    # Figure 1 plots our DeviceMark scores against speeds from two setups: each score must
    # be the one Table 12 prints, a Pipette speed must carry the notice Pipette's licence
    # asks for, and our own speed must be the one the paper prints
    speed_paper(paper)
    out = tmp_path / "paper-data"
    assert build.main([str(out), str(paper), "--speeds", str(speeds_file(tmp_path))]) == 0
    f1 = rows(out / "figure-1.csv")
    assert f1[1][:7] == ["Masein model, thinking off", "84.4", "81.4", "87.3", build.OURS, "16.0",
                         "Teraformer's app on a OnePlus 15 (Table 1)"]
    assert f1[2][5] == "15" and f1[2][7] == build.PIPETTE
    assert f1[3][:5] == ["Qwen3.5-4B (int8)", "26.1", "22.4", "29.9", build.DEVICEMARK]
    for change, says in (({"Qwen3.5-4B": {"devicemark_composite": "80.4"}}, "Table 12 prints 80.1"),
                         ({"Qwen3.5-4B": {"speed_credit": "Pipette"}}, "must carry its notice"),
                         ({"Masein model": {"speed_tok_s": "16.5"}}, "16.5 tokens/s is not in the paper")):
        with pytest.raises(SystemExit, match=says):
            build.main([str(tmp_path / "x"), str(paper), "--speeds", str(speeds_file(tmp_path, **change))])


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
    # timings on the OnePlus 15 can replace Pipette's: a speed in figure-1.csv wins,
    # and a row without one falls back to the value the script carries
    pytest.importorskip("matplotlib")
    figs = load("make_figures")
    f = tmp_path / "figure-1.csv"
    f.write_text("model,mode,speed_tok_s,speed_source\nQwen3.5-4B,thinking off,14.2,OnePlus 15, timed\n"
                 "LFM2.5-1.2B,,,\n".replace("OnePlus 15, timed", '"OnePlus 15, timed"'), encoding="utf-8")
    got = figs.speeds(tmp_path)
    assert got["Qwen3.5-4B, thinking off"] == ("14.2", "OnePlus 15, timed")
    assert got["LFM2.5-1.2B"][0] == "68" and got["LFM2.5-1.2B"][1].startswith("Pipette")
    assert got["Masein model, thinking off"][0] == "16.0"


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
