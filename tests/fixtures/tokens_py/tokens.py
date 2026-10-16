import json, os, statistics as st, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from frontier_graders import think_split
root = Path(os.environ.get("OUT_ROOT") or Path(os.environ.get("BENCH_ROOT", os.getcwd())) / "results") / "full"
TASKS = ["gpqa_diamond_epoch", "hle_text_cais", "otis_aime_epoch", "math_l5_epoch",
         "mmlupro_tiger", "simpleqa_epoch", "arc_agi2_public"]
rows = sorted(d for d in root.iterdir() if d.is_dir() and d.name.endswith("__thinking")
              and ("lda-2610" in d.name or "Q4-original-k-8" in d.name))
print("ROWS", [d.name for d in rows])
def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None
for d in rows:
    for t in TASKS:
        f = d / f"{t}_0shot" / "frontier"
        if not f.is_dir():
            continue
        recs, src = [], ""
        sm = sorted(f.glob(f"samples_{t}_*.jsonl"))
        if sm:
            src = "samples"
            with open(sm[-1], encoding="utf-8") as fh:
                for line in fh:
                    s = json.loads(line)
                    for resp, r in zip(s.get("resps") or [], s.get("frontier") or []):
                        recs.append((resp[0] if resp else "", r.get("tokens"), r.get("finish"), r.get("ok")))
        elif (f / "answers.jsonl").is_file():
            src = "answers"
            with open(f / "answers.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("error"):
                        continue
                    recs.append((r.get("answer") or "", r.get("tokens"), r.get("finish"), None))
        if not recs:
            continue
        tok, think, right, wrong, out_n = [], [], [], [], 0
        tc = ac = 0
        for text, n, fin, ok in recs:
            outside, thoughts, unclosed = think_split(text)
            th, an = sum(len(x) for x in thoughts), len(outside)
            cut = fin == "length" or unclosed
            out_n += cut
            if not isinstance(n, int):
                continue
            tc += th; ac += an
            tok.append(n)
            think.append(round(n * th / (th + an)) if th + an else 0)
            if not cut and ok is True:
                right.append(n)
            elif not cut and ok is False:
                wrong.append(n)
        mean = lambda v: round(st.mean(v)) if v else None
        print(json.dumps({"row": d.name, "task": t, "src": src, "n": len(recs), "n_tok": len(tok),
                          "ran_out": out_n, "tok_mean": mean(tok), "tok_med": q(tok, .5),
                          "tok_p90": q(tok, .9), "tok_max": max(tok) if tok else None,
                          "tok_sum": sum(tok), "think_mean": mean(think), "think_med": q(think, .5),
                          "think_share": round(tc / (tc + ac), 3) if tc + ac else None,
                          "right_n": len(right), "right_mean": mean(right),
                          "wrong_n": len(wrong), "wrong_mean": mean(wrong)}), flush=True)
print("DONE")
