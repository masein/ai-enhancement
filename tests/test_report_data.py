"""The report's public data repository: paper-data/ built from the paper itself, the scan
for internal terms, and the upload's checks before anything is sent (no network here)."""
import csv
import importlib.util
import json
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
