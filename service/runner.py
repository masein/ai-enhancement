"""Run one submission end-to-end. This is scripts/run_benchmarks.sh ported to
Python with the same semantics, because those semantics were earned the hard way
on this exact GPU:

  * same results tree (results/full/<model>/<task>_<n>shot/...) — service runs
    and CLI runs stay comparable and resume each other's work
  * same lock (results/.run.lock, mkdir-atomic, /proc staleness) — a service run
    and a manual run can never race
  * free-VRAM gate before loading, per-model batch from the vocab logits law
  * per-task resume: a (model, task) with results is never re-run
  * OOM aborts the rest of this model (it would OOM again); a missing python
    package fails the submission with the fix in the message
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config, db, gpu
from . import served as _served
from . import devicemark as _devicemark
from .hfmeta import PreflightError, preflight

LOCK = config.RESULTS_ROOT / ".run.lock"


# ---------------------------------------------------------------------------
# running someone else's model code with less of the machine attached
#
# When a submission opts into trust_remote_code, the eval subprocess executes
# Python that arrived in an upload. It still runs on the host GPU and can still
# read the results tree — this is a smaller blast radius, not a sandbox — but
# two specific things are worth taking away from it:
#
#   root         it drops to EVAL_USER, so a stray rmtree or chmod in someone's
#                modeling file cannot reach anything that user does not own
#   the HF token lives in HF_HOME and is what an exfiltration would actually be
#                worth. The token env vars are unset and the hub is put offline;
#                a local artifact needs neither, so nothing is lost. Make the
#                token file root-owned 0600 and the drop does the rest.
# ---------------------------------------------------------------------------

def _eval_ids() -> tuple[int, int] | None:
    """(uid, gid) for EVAL_USER, or None when it is unset/unknown."""
    if not config.EVAL_USER:
        return None
    try:
        import pwd
        rec = pwd.getpwnam(config.EVAL_USER)
        return rec.pw_uid, rec.pw_gid
    except (ImportError, KeyError):
        return None


def _job_scratch(sid: int, run_as: tuple[int, int] | None) -> Path:
    """A writable scratch tree for a dropped-privilege job, on the mounted
    volume rather than the container's own filesystem.

    Dropping to EVAL_USER without this is broken in two ways, both found by a
    friend's bug report rather than by me. HOME still points at root's home, so
    every library that caches under ~/.cache writes somewhere it cannot. And
    torch creates its inductor cache under tempfile.gettempdir() at IMPORT
    time — before any model is touched — so a /tmp the job user cannot write is
    an instant failure with a traceback that names none of our code. Putting
    HOME, TMPDIR and the torch/triton caches here fixes both, and keeps the
    writes on BENCH_ROOT, where there is room and the operator can see them."""
    d = config.BENCH_ROOT / ".jobscratch" / str(sid)
    for sub in ("tmp", "home/.cache", "inductor", "triton"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    if run_as:
        for p in (d, *(q for q in d.rglob("*") if q.is_dir())):
            try:
                os.chown(p, *run_as)
            except OSError:
                pass
    return d


# Every secret the service process may hold. A submitted model's own code runs
# in a child of this process, so anything not on this list is readable by it.
# Add a new secret's variable name here in the same commit that introduces it.
SECRET_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                   "HF_API_TOKEN", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY",
                   "LLM_API_KEY", "ANTHROPIC_API_KEY", "SUBMIT_TOKEN",
                   "EXAM_API_KEY", "JUDGE_API_KEY", "OPENROUTER_API_KEY")


def _child_env(remote_code: bool, scratch: Path | None = None) -> dict:
    env = os.environ.copy()
    if not remote_code:
        return env
    for var in SECRET_ENV_VARS:
        env.pop(var, None)
    env["HF_HUB_OFFLINE"] = "1"        # the artifact is on disk; nothing to fetch
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if scratch:
        env["HOME"] = str(scratch / "home")
        env["TMPDIR"] = str(scratch / "tmp")
        env["XDG_CACHE_HOME"] = str(scratch / "home" / ".cache")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "inductor")
        env["TRITON_CACHE_DIR"] = str(scratch / "triton")
    return env


# the import chain that failed for the first real remote-code submission: it
# runs before any model is loaded, so when the job environment is broken the
# submitter gets "task failed" for a fault that has nothing to do with their
# model. Probe it once, up front, and say so in those words.
_CANARY = (
    # the import chain that failed (torch builds its inductor cache at import)
    "import os, transformers.generation.utils, torch._dynamo; "
    # and the next wall: datasets writes lock files into the HF cache, which is
    # root-owned because the service created it
    "h = os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface'); "
    "os.makedirs(h, exist_ok=True); "
    "f = os.path.join(h, '.bench-write-probe'); "
    "open(f, 'w').write('x'); os.remove(f)")


def _env_canary(env: dict, run_as: tuple[int, int] | None) -> str:
    """'' if the job environment can import the stack AND write the caches it
    needs, else the last line of what went wrong. Two failures this catches that
    otherwise surface as four identical tracebacks blamed on the model: a
    /tmp the job user cannot write (torch makes its inductor cache at import
    time) and an HF cache it cannot write (datasets takes a lock)."""
    try:
        p = subprocess.run([sys.executable, "-c", _CANARY], env=env,
                           capture_output=True, text=True, timeout=600,
                           **({"user": run_as[0], "group": run_as[1]} if run_as else {}))
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not start a job process: {e}"
    if p.returncode == 0:
        return ""
    tail = (p.stderr or p.stdout or "").strip().splitlines()
    return tail[-1] if tail else f"exit {p.returncode}"


# ---------------------------------------------------------------------------
# the shared-GPU primitives
# ---------------------------------------------------------------------------

def load_spec(hf_id: str, meta: dict) -> dict:
    """How a model loads, for a run and (12d.1) for the Playground: its path —
    local/<name> artifacts resolve to their directory, and the report
    normalizes the path back to local/<name> so ids stay consistent — the
    dtype, and its own code only at the approved commit (12h.1: that commit,
    no other)"""
    pretrained = hf_id
    if pretrained.startswith("local/"):
        pretrained = str((config.ARTIFACTS_DIR / pretrained[6:]).resolve())
    return {"pretrained": pretrained, "dtype": "bfloat16",
            "trust_remote_code": bool(meta.get("remote_code")),
            "revision": meta.get("revision") or None,
            # 12t: what lm_eval is told of a model whose limit it can't find
            "max_length": nested_limit(meta)}


def nested_limit(meta: dict) -> int | None:
    """12t: the tokens a model reads, when its config keeps that under
    text_config (Gemma 4) — None for every other model. lm_eval looks for a
    model's limit at the top of its config only, and takes 2,048 when it finds
    none: a prompt longer than that is cut from the left without a word, and a
    generative task with 2,048 tokens or more to write fails outright. Such a
    model's runs on hf are told its real limit; a model whose limit lm_eval
    finds is told nothing, and is asked exactly as before"""
    a = meta.get("archinfo") or {}
    try:
        return int(a["ctx"]) if a.get("ctx_nested") and a.get("ctx") else None
    except (TypeError, ValueError):
        return None


def model_args(spec: dict) -> str:
    """lm_eval's --model_args for a load_spec"""
    margs = f"pretrained={spec['pretrained']},dtype={spec['dtype']}"
    if spec["trust_remote_code"]:
        margs += ",trust_remote_code=True"
    if spec["revision"]:
        margs += f",revision={spec['revision']}"
    if spec.get("max_length"):                  # 12t: nested_limit
        margs += f",max_length={spec['max_length']}"
    return margs


def run_holding() -> dict | None:
    """12d.1: the run holding the GPU lock, as far as the service can say —
    {sid, hf_id, suite} (sid 0 and no model for a command-line run) — or None"""
    if not LOCK.exists():
        return None
    try:
        sid = int((LOCK / "submission").read_text().strip() or 0)
    except (OSError, ValueError):
        sid = 0
    sub = db.get(sid) if sid else None
    return {"sid": sid, "hf_id": (sub or {}).get("hf_id") or "",
            "suite": (sub or {}).get("suite") or ""}


def _everyday_settings(meta: dict) -> dict:
    """everyday.run_settings for this model: how its Everyday answers are
    generated (12d.1)"""
    import everyday as _ev          # scripts/, on sys.path in the service
    return _ev.run_settings(meta.get("archinfo"))


# the exam's own template (eval_tasks/fr/_fr_template_yaml): what it stops on
EXAM_UNTIL = ["\n\n\n"]
EXAM_MAX_GEN_TOKS = 256


def _exam_settings(meta: dict) -> dict:
    """how a Knowledge exam answer is generated, said once (12f.1): a local
    run's lm_eval command and a served model's questions both read it. 11l:
    a reasoning model thinks before it answers, and 256 tokens ran out
    inside the thinking on every question of run #60"""
    thinks = bool((meta.get("archinfo") or {}).get("reasoning_template"))
    return {"until": EXAM_UNTIL, "do_sample": False, "temperature": 0.0,
            "max_gen_toks": config.REASONING_MAX_GEN_TOKS if thinks else EXAM_MAX_GEN_TOKS}


def hours_words(s: float) -> str:
    """"5.6 h", "41 min" """
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{max(1, round(s / 60))} min"


def time_left(done: int, total: int, secs_each: float) -> str:
    """"140 of 200 · 4.1 s an answer · about 4 min left": a slow run's
    progress, from the seconds each answer has taken so far"""
    left = max(0, total - done) * secs_each
    when = (f"about {left / 3600:.1f} h left" if left >= 5400 else
            f"about {max(1, round(left / 60))} min left" if left >= 60 else "under a minute left")
    return f"{done} of {total} · {secs_each:.1f} s an answer · " + (when if done < total else "done")


def gpu_free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30)
    return int(out.stdout.splitlines()[0].strip())


def acquire_lock(sid: int) -> bool:
    """Same protocol as the CLI script; returns False while someone else runs."""
    try:
        LOCK.mkdir(parents=True)
    except FileExistsError:
        owner = ""
        try:
            owner = (LOCK / "pid").read_text().strip()
        except OSError:
            pass
        beat = LOCK / "heartbeat"
        if beat.exists():
            # 12f.3: the GGUF worker on the host holds it — its pid is not one
            # this container can see; its heartbeat says whether it's alive
            if time.time() - beat.stat().st_mtime < LOCK_BEAT_S:
                return False
        elif owner and Path(f"/proc/{owner}").is_dir():
            return False                      # a live run (CLI or us) holds it
        # stale — the holder died hard; take over
        subprocess.run(["rm", "-rf", str(LOCK)], check=False)
        try:
            LOCK.mkdir(parents=True)
        except FileExistsError:
            return False
    (LOCK / "pid").write_text(str(os.getpid()))
    (LOCK / "submission").write_text(str(sid))
    return True


# 12f.3: a worker's lock whose heartbeat is older than this is a dead worker's
LOCK_BEAT_S = 120


def gguf_holding() -> int | None:
    """12f.5: the GGUF job holding the lock while its worker is alive — its
    run id, 0 when the lock doesn't say — else None. Only the worker's lock
    has a heartbeat, and acquire_lock takes one gone quiet"""
    try:
        if time.time() - (LOCK / "heartbeat").stat().st_mtime >= LOCK_BEAT_S:
            return None
    except OSError:
        return None
    try:
        return int((LOCK / "submission").read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def wait_for_lock(sid: int) -> bool:
    """the run lock, taken for `sid`; False when it was canceled or gave up
    waiting, the row saying so. 12f.5: the GGUF worker is a queued board job,
    not a manual run — #96 gave up on it after six hours — so while it holds
    the lock the run waits it out, saying for which run and how long; the six
    hours count only while another run holds it"""
    t0 = time.time()
    while not acquire_lock(sid):
        if db.cancel_requested(sid):
            db.update(sid, status="canceled", finished_at=time.time(),
                      progress="canceled by request while waiting for the run lock")
            return False
        g = gguf_holding()
        if g is not None:
            from . import gguf as _gguf
            db.update(sid, status="waiting_lock", progress=_gguf.waiting_line(g))
            t0 = time.time()
        else:
            db.update(sid, status="waiting_lock",
                      progress="another run (service or CLI) holds the GPU lock")
            if time.time() - t0 > config.GPU_WAIT_MAX_S:
                db.update(sid, status="failed", finished_at=time.time(),
                          error="gave up waiting for the run lock — a manual run has "
                                "held the GPU for hours; resubmit later.")
                return False
        time.sleep(config.GPU_POLL_S)
    return True


def release_lock() -> None:
    subprocess.run(["rm", "-rf", str(LOCK)], check=False)


# ---------------------------------------------------------------------------
# error classification — the message a friend sees instead of a traceback
# ---------------------------------------------------------------------------

_FRIENDLY = [
    # 12q.G: PyTorch's own message says who held what (oom_line); these are the
    # words for one that isn't PyTorch's
    (r"out of memory|OutOfMemoryError",
     "ran out of GPU memory. "
     "Resubmit; the finished tasks are kept and only the missing ones re-run."),
    # 12n.2: GPQA's dataset is gated, not the model — said so, with where to accept it
    (r"(?s)Idavidrein/gpqa.{0,400}(?:gated|authenticat|401|403|GatedRepoError)"
     r"|(?:gated|authenticat|401|403|GatedRepoError).{0,400}Idavidrein/gpqa",
     config.GPQA_GATED),
    (r"GatedRepoError|401 Client",
     "the model is gated for this server's HF account — accept the license on "
     "huggingface.co and resubmit."),
    (r"ModuleNotFoundError|ImportError",
     "the server's python environment is broken (missing package) — tell the "
     "operator; this fails identically for every model."),
    (r"trust_remote_code",
     "the model needs its own modeling code executed. Upload it as an artifact "
     "and submit with allow_remote_code=true; code from "
     "the Hub is never executed here."),
    (r"No space left on device|Errno 28",
     "the server ran out of disk. Nothing to do with your model — tell the "
     "operator. (Jobs that drop privileges hit this first: ext4 keeps 5% of "
     "blocks in reserve for root, so root-run jobs keep working while these "
     "fail.)"),
    (r"no kernel image",
     "PyTorch/CUDA mismatch on the server (wrong wheel for this GPU) — operator "
     "issue, not your model."),
]


# 12q.G: who held what when the card ran out, from PyTorch's own message. The
# line above blamed the card every time; Granite-4.0-H-1B's own process had
# grown to 15.9 GiB and asked for 6 more beside a 12.9 GiB neighbour that
# hadn't moved (#147)
_OOM = re.compile(r"out of memory|OutOfMemoryError", re.I)
_SIZE = r"([\d.]+) ?(GiB|MiB|KiB|bytes)"
_OOM_GREW_GIB = 1.0          # the others count as busier when they hold this much more
# lm_eval's own warning when a prompt is longer than max_length less the answer
_CUT = re.compile(r"Left truncation applied\. Original sequence length was (\d+), "
                  r"truncating to last (\d+) tokens")


def _gib(num: str, unit: str) -> float:
    return float(num) / {"GiB": 1, "MiB": 1024, "KiB": 1024 ** 2, "bytes": 1024 ** 3}[unit]


def oom_said(text: str) -> dict | None:
    """{asked, own, others, total, spare} in GiB, from the last out-of-memory
    message in `text`; None when it isn't PyTorch's. `own` is the run's own
    process, `others` every other process on the card (what PyTorch lists, or
    what is left of the card), `spare` what the run had reserved and not used"""
    at = text.rfind("Tried to allocate")
    if at < 0:
        return None
    said = text[at:at + 1500]
    find = lambda pat: (lambda m: _gib(*m.groups()) if m else None)(re.search(pat, said))
    asked = find(rf"Tried to allocate {_SIZE}")
    total, free = find(rf"total capacity of {_SIZE}"), find(rf"of which {_SIZE} is free")
    used, spare = find(rf"memory {_SIZE} is allocated by PyTorch"), find(rf"and {_SIZE} is reserved")
    own = find(rf"this process has {_SIZE} memory in use")
    if own is None and used is not None:
        own = used + (spare or 0)
    listed = [_gib(*m) for m in re.findall(rf"Process \d+ has {_SIZE} memory in use", said)]
    others = (sum(listed) if listed else
              max(0.0, total - free - own) if None not in (total, free, own) else None)
    if asked is None or own is None:
        return None
    return {"asked": asked, "own": own, "others": others, "total": total, "spare": spare}


def oom_line(o: dict, before_mib: int | None = None) -> str:
    """"this run's own process held 15.9 GB and asked for 6 GB more; the
    card's other processes held 12.9 GB, as when the task started: the run
    grew, not the card" — `before_mib` is what the others held then. 16.2: GB
    as everywhere the board says GPU memory (1024³, PyTorch's GiB)"""
    g = lambda v: f"{v:.1f}".rstrip("0").rstrip(".") + " GB"
    line = (f"this run's own process held {g(o['own'])}"
            + (f" ({g(o['spare'])} of it reserved and unused)"
               if (o["spare"] or 0) >= _OOM_GREW_GIB else "")
            + f" and asked for {g(o['asked'])} more")
    if o["others"] is None:
        return line
    line += f"; the card's other processes held {g(o['others'])}"
    if before_mib is None:
        return line
    before = before_mib / 1024
    if o["others"] - before >= _OOM_GREW_GIB:
        return line + f", up from {g(before)} when the task started: the card got busier"
    return line + (", as when the task started" if abs(o["others"] - before) < 0.1 else
                   f" ({g(before)} when the task started)") + ": the run grew, not the card"


def others_mib() -> int | None:
    """what the card's processes hold now, before a task's own is started"""
    try:
        total = gpu_total_mib()
        return None if not total else max(0, total - gpu_free_mib())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def classify(log_tail: str) -> str:
    for pat, msg in _FRIENDLY:
        if re.search(pat, log_tail, re.I):
            # 12f.4: which package — "missing package: tiktoken" says what to install
            gone = re.findall(r"No module named '([\w.]+)'", log_tail)
            if gone and "(missing package)" in msg:
                msg = msg.replace("(missing package)", f"(missing package: {gone[-1].split('.')[0]})")
            return msg
    return "task failed — see the log link for the raw error."


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# weights that never arrived
#
# transformers does not fail when a checkpoint lacks a parameter the model class
# declares. It randomly initializes the gap, prints a report, and hands back a
# model that runs — so lm_eval scores it, exits 0, and the number reaches the
# leaderboard looking like every other number.
#
# The first custom-code submission this server ever completed did exactly that.
# A GPTNeoX checkpoint stores its output projection as embed_out.weight; the
# uploaded class called the module lm_head. 74 of 75 tensors loaded, the head
# was noise, both accuracy tasks landed within a point of chance and both
# perplexities were several times worse than the same weights under the stock
# class. Status said 'done'. Nothing anywhere said otherwise, and it was caught
# only because that upload happened to be a known model with a reference row.
#
# A crash is a safe failure: somebody fixes it. A plausible wrong number on a
# shared board is not — it gets believed, and it gets compared against. So the
# report is parsed after every task, for every submission and not just the ones
# running custom code: a renamed head or a half-saved checkpoint does this
# without any remote code involved.
#
# Only MISSING is fatal. UNEXPECTED means the checkpoint carries tensors this
# class has no slot for, which is normal when loading across task heads. Tied
# embeddings are the one plausible false positive — a model with
# tie_word_embeddings does not store lm_head.weight separately — but
# transformers resolves tying before it writes this report, and runs 31-34 here
# (one of them tied) produce no MISSING rows, so the distinction holds in
# practice on 5.15.1. If that ever changes the symptom is a clean model being
# refused, which is loud, checkable and the safe direction to be wrong in.
# ---------------------------------------------------------------------------

_MISSING_ROW = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*\|\s*MISSING\b", re.M)
_MISSING_PROSE = re.compile(r"newly initialized:\s*\[([^\]]*)\]")

_MISSING_MSG = (
    "the checkpoint does not contain weights this architecture needs: {keys}{more}. "
    "transformers filled them in with random values, so the model ran and scored "
    "but the numbers were noise — they were discarded instead of published. This "
    "is almost always a naming disagreement between the checkpoint and the model "
    "class: a GPTNeoX checkpoint stores the output projection as embed_out.weight, "
    "for instance, while a custom class that calls it lm_head silently gets a "
    "random head. Rename the module or the tensor so the two agree, confirm the "
    "load report is clean, and resubmit.")


def _missing_weights(text: str) -> list[str]:
    """Checkpoint keys transformers had to invent, or [] when the load was clean.

    Two formats are matched because the wording belongs to transformers, not to
    us, and a guard that quietly stops guarding when a library reformats its log
    is worse than no guard: the table emitted by 5.x, and the older one-line
    'newly initialized: [...]' prose."""
    keys = set(_MISSING_ROW.findall(text))
    for blob in _MISSING_PROSE.findall(text):
        keys.update(k.strip().strip("'\"") for k in blob.split(",") if k.strip())
    return sorted(keys)


def _read_from(path: Path, offset: int) -> str:
    """The log written since `offset` — one task's own output, so a report left
    by an earlier task in the same submission cannot be re-attributed here.

    Read as bytes and decoded here: `offset` is a byte count from stat(), and
    seeking a text handle to anything but a tell() cookie is undefined."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _chat_renderer(hf_id: str, meta: dict):
    """14.1: a conversation in the model's own chat template, as text — MT-Bench's
    second turn, asked by lm_eval as it stands (no chat template of its own on
    top). A Gemma model's BOS is left to lm_eval, which adds one to every Gemma
    prompt; any other keeps the template's"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, revision=meta.get("revision"),
                                        trust_remote_code=bool(meta.get("remote_code")))
    bos = tok.bos_token or ""
    gemma = "gemma" in str((meta.get("archinfo") or {}).get("arch") or hf_id).lower()

    def render(messages: list[dict]) -> str:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text[len(bos):] if gemma and bos and text.startswith(bos) else text
    return render


def include_args_for(task: str) -> list[str]:
    """--include_path for a task the harness does not ship. The perplexity
    slices live under BENCH_ROOT/eval_tasks; the permutation control ships with
    this repo (eval_tasks/mmlu_perm). One task per lm_eval invocation means
    each task gets exactly the directory it lives in — no task ever sees the
    other's yaml, so a stray file in one cannot rename a task in the other."""
    if task in config.CONTROL_TASKS:
        return ["--include_path", str(config.CONTROL_TASKS_DIR)]
    if task == config.EVERYDAY_TASK:
        return ["--include_path", str(config.EVERYDAY_TASKS_DIR)]
    if task in config.TRUST_TASKS:                  # 12k.2: trust_safety.build_tasks writes them
        return ["--include_path", str(config.TRUST_TASKS_DIR)]
    if task == config.SIMPLEQA_TASK:                # 12n.2: simpleqa.build_tasks writes it
        return ["--include_path", str(config.SIMPLEQA_TASKS_DIR)]
    if task in config.MAB_ALL:
        # 12o.3, 14.1: mobileaibench.build_tasks writes them; MT-Bench's second
        # turn is the model's own (lm_eval_cmd's include_dir)
        return ["--include_path", str(config.MAB_TASKS_DIR)]
    if task == config.MMP_TASK:                     # 14.3: mobile_mmlu.build_tasks writes it
        return ["--include_path", str(config.MMP_TASKS_DIR)]
    if task == config.MMF_TASK:                     # 14.4: and build_full_tasks the full set's
        return ["--include_path", str(config.MMF_TASKS_DIR)]
    if task in config.DM_TASKS:                     # 12q: devicemark.build_tasks writes them
        return ["--include_path", str(config.DM_TASKS_DIR)]
    if task.startswith(("exam_", "fr_")):
        return ["--include_path", str(config.JUDGED_TASKS_DIR)]
    if config.EVAL_TASKS_DIR.is_dir() and any(config.EVAL_TASKS_DIR.glob("*.yaml")):
        return ["--include_path", str(config.EVAL_TASKS_DIR)]
    return []


WHOLE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lm_eval_whole.py"


def lm_eval_cmd(model_args: str, task: str, shots: int, batch, task_out: Path, *,
                chat: bool, max_gen_toks: int | None = None, backend: str = "hf",
                samples: Path | None = None, limit: int | None = None,
                cache: Path | None = None, system: str | None = None,
                include_dir: Path | None = None, whole: bool = False) -> list[str]:
    """The lm_eval command for one task. Built here only, so that
    scripts/check_tasks.py (deploy step 4) hands the installed harness exactly
    what a run hands it. 12h.1: `backend` is "hf" or "vllm" (vLLM sizes its
    own batches and picks its own device), and `samples` a seeded subset.
    12f.1: "local-chat-completions" asks a model served elsewhere — no device
    here, one request a message, and `cache`, the answers it has, so a run
    the server stopped asks only the rest next time. 15.7: `whole`, hf's
    replies kept whole — every token, special ones too, nothing cut at the end
    of the thinking (scripts/lm_eval_whole.py's hf-whole)"""
    cmd = [*([sys.executable, str(WHOLE_SCRIPT)] if whole else ["lm_eval"]),
           "--model", "hf-whole" if whole else backend,
           "--model_args", model_args,
           "--tasks", task,
           "--num_fewshot", str(shots),
           "--batch_size", "auto" if backend == "vllm" else
           "1" if backend == _served.BACKEND else str(batch),
           "--seed", str(config.SEED),
           "--output_path", str(task_out),
           "--log_samples",
           *([] if backend in ("vllm", _served.BACKEND) else ["--device", "cuda:0"]),
           *(["--include_path", str(include_dir)] if include_dir else include_args_for(task))]
    if chat:
        cmd.append("--apply_chat_template")
    if max_gen_toks:
        cmd += ["--gen_kwargs", f"max_gen_toks={max_gen_toks}"]
    if samples:
        cmd += ["--samples", str(samples)]
    if limit:                        # scripts/trial_generative.py only: a few items
        cmd += ["--limit", str(limit)]
    if cache:
        cmd += ["--use_cache", str(cache)]
    # 12o.3: a benchmark whose prompt has a system line of its own (MobileAIBench's)
    if system:
        cmd += ["--system_instruction", system]
    return cmd


# ---------------------------------------------------------------------------
# 12h.1: IFEval, MMLU-Pro and MATH-500 — instruct models, thinking on or off,
# on vLLM where the model loads in it
# ---------------------------------------------------------------------------

def gen_thinking(sub: dict, meta: dict) -> dict:
    """{mode, on, separate, budget, think_end}: how this run asks it. Off by
    default for every model that can turn it off (a phone assistant answers
    straight away); on for one that cannot; on when a person asked, for a
    model with a switch — and that run is a row of its own, never averaged
    with the model's thinking-off row"""
    a = meta.get("archinfo") or {}
    mode = a.get("thinking") or "never"
    on = mode == "always" or (mode == "switch" and bool(sub.get("thinking")))
    return {"mode": mode, "on": on, "separate": mode == "switch" and on,
            "budget": config.GEN_THINKING_MAX_GEN_TOKS if on else config.GEN_MAX_GEN_TOKS,
            "think_end": a.get("think_end") or "</think>"}


# 16b (masein, 5 Oct): the thinking rule of the Mobile suite on every path,
# and of Instruction & maths and Frontier for a served model too — a model
# that can turn its thinking off answers with it off, unless a person asks for
# a thinking run, which is a row of its own ("· thinking"). A served model is
# asked with enable_thinking said out loud, whatever its registration says
# (one from OpenRouter has no switch: it thinks as it does). Everyday, Trust &
# safety and the exam keep the chat template's own default (decided 5 Oct)
THINKING_RULE = ("Asked with thinking off for a model that can turn it off — served models "
                 "too — unless a thinking run is asked for: that is a row of its own "
                 "(“· thinking”)")
THINKING_NAME = "thinking.json"


def suite_thinking(sub: dict, meta: dict, rec: dict | None) -> dict:
    """{mode, on, separate, budget, think_end} for a run of the Mobile suite, or
    of Instruction & maths or Frontier: gen_thinking's rule, on every path"""
    if rec and not _served.is_openrouter(rec):
        on = bool(sub.get("thinking"))
        return {"mode": "switch", "on": on, "separate": on,
                "budget": config.GEN_THINKING_MAX_GEN_TOKS if on else config.GEN_MAX_GEN_TOKS,
                "think_end": (meta.get("archinfo") or {}).get("think_end") or "</think>"}
    return gen_thinking(sub, meta)


def gen_model_args(pretrained: str, th: dict, *, backend: str, remote_code: bool = False,
                   revision: str | None = None, gpu_util: float | None = None,
                   max_length: int | None = None) -> str:
    """The model_args of a generative run: the thinking switch said out loud
    (both ways — Qwen3 thinks unless told not to, Qwen3.5 only when told), the
    end of the thinking when it thinks (the harness scores what follows it),
    the approved commit of a model that runs its own code, and vLLM's share
    of the card. 12q.G: `max_length`, on hf, is the length lm_eval is told
    instead of looking for the model's (it takes 2,048 when it finds none)"""
    parts = [f"pretrained={pretrained}", "dtype=bfloat16"]
    if th["mode"] == "switch":
        parts.append(f"enable_thinking={th['on']}")
    if th["on"]:
        parts.append(f"think_end_token={th['think_end']}")
    if remote_code:
        parts.append("trust_remote_code=True")
    if revision:
        parts.append(f"revision={revision}")
    if backend == "vllm":
        parts += [f"gpu_memory_utilization={gpu_util or 0.8}",
                  f"max_model_len={th['budget'] + 6144}"]
    elif max_length:
        parts.append(f"max_length={max_length}")
    return ",".join(parts)


def vllm_available() -> bool:
    """vLLM is in the image (12h.1 builds it in; a build without it runs
    everything on hf)"""
    import importlib.util
    return importlib.util.find_spec("vllm") is not None


def gen_backend() -> tuple[str, str]:
    """(backend, why not vLLM)"""
    if config.GEN_BACKEND == "hf":
        return "hf", "GEN_BACKEND=hf"
    if not vllm_available():
        return "hf", "vLLM is not installed in this image"
    return "vllm", ""


def gpu_util_for(free_mib: int, total_mib: int | None) -> float:
    """vLLM takes a fraction of the WHOLE card up front: a share of what is
    free, never more than 0.9 (the CLI's rule, run_benchmarks.sh)"""
    if not total_mib:
        return 0.8
    return round(max(0.1, min(0.9, free_mib * 0.85 / total_mib)), 2)


def gpu_total_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _restricted(task: str) -> str:
    """14.4.4: '' for a task whose questions may be few-shot examples; else why
    not (a restricted set's never are)"""
    _scripts()
    import restrictions
    return restrictions.never_trained([task])


def mmlu_pro_subset(n: int, seed: int = config.GEN_SUBSET_SEED) -> dict[str, list[int]]:
    """A fixed, seeded subset of MMLU-Pro: `n` items, each subject in its
    share of the 12,032, the same items for every model. Only a full run is
    comparable to published numbers, and the page says which this is"""
    import random
    subj = config.MMLU_PRO_SUBJECTS
    total = sum(subj.values())
    n = max(1, min(n, total))
    raw = {k: n * v / total for k, v in subj.items()}
    take = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - take[k], reverse=True)[:n - sum(take.values())]:
        take[k] += 1
    rng = random.Random(seed)
    return {f"mmlu_pro_{k}": sorted(rng.sample(range(subj[k]), take[k])) for k in subj}


def lm_eval_cwd(task_out: Path) -> Path:
    """Where lm_eval runs: the task's own output folder, never BENCH_ROOT.

    lm_eval 0.4.12 reads a --tasks value that names a folder in its working
    directory as a folder of task yaml files, and never looks the name up
    (lm_eval/config/evaluate_config.py, process_tasks). BENCH_ROOT/everyday
    is such a folder: run from BENCH_ROOT, `--tasks everyday` found no yaml
    directly in it and selected nothing, and all four everyday runs, #62 to
    #65, failed before asking a question. The output folder holds only what
    lm_eval writes there, a folder named after the model, never a task."""
    return task_out


def dm_cache(task_out: Path) -> Path:
    """15.1: a DeviceMark task's answers as they are written, one by one (15.3:
    any task that writes its answers here, the generative three too) —
    lm_eval's own cache (--use_cache; it adds "_rank0.db"), committed after
    each answer. A task stopped part-way — a restart, a timeout, a rented box
    stopped — asks only the rest when it runs again. Inside the task's folder,
    so whatever sets the folder aside or discards it takes its cache too"""
    return task_out / "lm-cache" / "answers"


def keep_whole(task_out: Path) -> None:
    """15.7: a DeviceMark task with the thinking on saves its replies whole
    (lm_eval_whole.py), and says so beside them (devicemark.reply_form) —
    before it starts, so answers kept part-way are known for what they are.
    Answers an earlier run kept in lm_eval's cut form are set aside, not
    mixed in: the task asks them again"""
    d = _devicemark.dm()
    cache = task_out / "lm-cache"
    if cache.is_dir() and d.reply_form(task_out) != "whole" and any(cache.glob("*.db")):
        shutil.move(str(cache), str(task_out / f"lm-cache-cut-{int(time.time())}"))
    task_out.mkdir(parents=True, exist_ok=True)
    (task_out / d.FORM_NAME).write_text(json.dumps(
        {"form": "whole", "by": "scripts/lm_eval_whole.py (hf-whole)",
         "scoring": d.SCORING}), encoding="utf-8")


def saved_cut(task_out: Path, task: str) -> bool:
    """15.7a: a DeviceMark task holding answers — whole or part-way — saved in
    lm_eval's cut form: anything but what keep_whole marks whole"""
    if not task_out.is_dir() or _devicemark.dm().reply_form(task_out) == "whole":
        return False
    return (any(task_out.rglob(f"samples_{task}_*.jsonl"))
            or any(task_out.glob("lm-cache/*.db")))


def set_aside_cut(task_out: Path) -> Path:
    """15.7a: a task's cut-form answers moved beside the tree, whole —
    results/earlier/<row>/<task>-cut-<time> — never read as its answers again"""
    dest = (config.OUT_DIR.with_name("earlier") / task_out.parent.name
            / f"{task_out.name}-cut-{int(time.time())}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task_out), str(dest))
    return dest


# ---------------------------------------------------------------------------
# 16b: answers made under the other thinking setting are never reused — a
# model's thinking-on answers kept in its own row (run #180's, a Hugging Face
# model's under its template's default) are its thinking row's; anything else
# goes beside the tree, as 15.7a's do
# ---------------------------------------------------------------------------

_THINK = re.compile(r"(?s)^\s*(?:<think>)?(.*?)</think>")


def _reply_thinks(text) -> bool:
    """a reply that holds thinking: text before a </think>, as a local run's
    has it and a served model's is put back (served.ask)"""
    m = _THINK.match(str(text or ""))
    return bool(m and m.group(1).strip())


def answered_thinking(task_out: Path, unmarked: str = "off") -> str | None:
    """the thinking setting a task's answers on disk were made with — "on",
    "off", or "default" (as a server decided: not known) — from its
    THINKING_NAME, else its replies; None when it holds none. Replies with no
    thinking in them are `unmarked`'s: "off" where a reply keeps its thinking
    (a local run's, a served one's through served.ask); a served model's
    lm_eval answers lost theirs to the server's reasoning field, so theirs is
    what it was registered with (unmarked_of). The replies only, never the
    prompt (a template that switches the thinking off writes an empty block
    into it)"""
    files = list(task_out.rglob("samples_*.jsonl")) if task_out.is_dir() else []
    if not files:
        return None
    try:
        return "on" if json.loads((task_out / THINKING_NAME).read_text())["enable_thinking"] \
            else "off"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if "</think>" not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                for r in (rec.get("filtered_resps") or []) + (rec.get("resps") or []):
                    while isinstance(r, list) and r:
                        r = r[0]
                    if _reply_thinks(r):
                        return "on"
    return unmarked


def unmarked_of(rec: dict | None, through_lm_eval: bool) -> str:
    """what a task's answers with no thinking in their replies were made with
    (answered_thinking): a served model's through lm_eval, before 16b, as it
    was registered — "the model decides" is not known"""
    if not (rec and through_lm_eval):
        return "off"
    return {"on": "on", "off": "off"}.get(rec.get("thinking"), "default")


def mark_thinking(task_out: Path, th: dict) -> None:
    """what a task's answers are asked with, beside them, before it asks"""
    task_out.mkdir(parents=True, exist_ok=True)
    (task_out / THINKING_NAME).write_text(json.dumps(
        {"enable_thinking": bool(th["on"]), "rule": THINKING_RULE, "at": time.time()}),
        encoding="utf-8")


def _verdicts_of(task: str, key: str) -> bool:
    """whether a MobileAIBench verdict ("<task>:<id>:<turn>") is on this task's
    answers: MT-Bench's two turns are one task's verdicts, by their turn"""
    if task.startswith("mab_mtbench_t"):
        return key.startswith("mab_mtbench:") and key.endswith(":" + task[-1])
    return key.startswith(task + ":")


def sort_thinking(task_out: Path, task: str, want_on: bool, thinking_row: Path | None,
                  meta_row: dict | None = None, unmarked: str = "off") -> tuple[str, Path] | None:
    """answers kept under another setting: thinking-on answers in a model's
    own row go to its thinking row (`thinking_row`) when that has none for this
    task; otherwise — the other way round, or a setting not known — beside the
    tree, results/earlier/<row>/<task>-thinking-<on|off|default>-<time>. Their
    judge's verdicts go with them. ("thinking row" | "earlier", where) — or
    None when the answers are this setting's, or there are none"""
    was = answered_thinking(task_out, unmarked)
    if was is None or was == ("on" if want_on else "off"):
        return None
    row = task_out.parent
    if was == "on" and not want_on and thinking_row is not None \
            and not (thinking_row / task_out.name).exists():
        thinking_row.mkdir(parents=True, exist_ok=True)
        if not (thinking_row / "model_meta.json").exists() and meta_row:
            (thinking_row / "model_meta.json").write_text(json.dumps(meta_row), encoding="utf-8")
        dest, kind = thinking_row / task_out.name, "thinking row"
    else:
        dest = (config.OUT_DIR.with_name("earlier") / row.name
                / f"{task_out.name}-thinking-{was}-{int(time.time())}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        kind = "earlier"
    shutil.move(str(task_out), str(dest))
    if kind == "thinking row":
        _results_say_thinking(dest)
    # the judge's verdicts on those answers go with them (they are keyed by
    # question: left here, they would be read as new answers' marks)
    if task in config.MAB_ALL:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import mobileaibench as _mab
        got = _mab.read_judged(row)
        vs = got.get("verdicts") or {}
        moving = {k: v for k, v in vs.items() if _verdicts_of(task, k)}
        if moving:
            got["verdicts"] = {k: v for k, v in vs.items() if k not in moving}
            _mab.write_judged(row, got)
            if kind == "thinking row":
                there = _mab.read_judged(dest.parent)
                there["verdicts"] = {**(there.get("verdicts") or {}), **moving}
                _mab.write_judged(dest.parent, there)
            else:
                (dest / "verdicts.json").write_text(json.dumps(moving), encoding="utf-8")
    for d in {row, dest.parent} if kind == "thinking row" else {row}:
        _mobile_scores_again(d)
    return kind, dest


def _results_say_thinking(task_out: Path) -> None:
    """answers moved into a thinking row: their results say they thought, as a
    thinking run's do (enable_thinking=True in model_args), so the report files
    them under "· thinking" and not the model's own row"""
    for f in task_out.rglob("results*.json"):
        try:
            blob = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cfg = blob.setdefault("config", {})
        args = cfg.get("model_args") or ""
        if isinstance(args, dict):
            cfg["model_args"] = {**args, "enable_thinking": True}
        elif "enable_thinking=True" not in str(args):
            cfg["model_args"] = re.sub(r",?enable_thinking=False", "", str(args)) \
                + ",enable_thinking=True"
        else:
            continue
        f.write_text(json.dumps(blob, indent=1), encoding="utf-8")


def _mobile_scores_again(row: Path) -> None:
    """a row's Mobile scores from the answers it holds now — none left, none kept"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import mobile_mmlu as _mm
    import mobileaibench as _mab
    out = _mab.mark(row)
    if out:
        _mab.write(row, out)
    else:
        (row / _mab.OUT_NAME).unlink(missing_ok=True)
    if _mm.collect(row) is None:
        for name in (_mm.PRED_FILE, _mm.FULL_PRED_FILE):
            (row / name).unlink(missing_ok=True)


def _has_results(task_out: Path) -> bool:
    return any(task_out.glob("*/results*.json")) or any(task_out.glob("results*.json"))


# ---------------------------------------------------------------------------
# which questions a task's answers answer. A task's name does not say:
# exam_law held the retired law's questions until phase 10 and the new Law's
# after it. The exam build fingerprints each task's question set; the runner
# writes the same value beside the answers, and answers count as done only
# while the two agree. Tasks the build does not write (mmlu, hellaswag…) have
# no fingerprint and resume on their name, as always.
# ---------------------------------------------------------------------------

BANK_FILE = "bank.sha256"
ANSWERED_BY = "answered_by.json"


def _scripts() -> None:
    p = str(Path(__file__).resolve().parent.parent / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


def current_fingerprint(task: str) -> str | None:
    """The question set the built task holds now, or None for a task the
    exam build does not write."""
    _scripts()
    import exam_build
    return (exam_build.current_fingerprints(config.JUDGED_TASKS_DIR) or {}).get(task)


def answers_fingerprint(task_out: Path, task: str) -> str | None:
    """Which question set the answers in `task_out` were given on: the file
    the runner wrote beside them, or — for answers from before it wrote one
    — what the answers themselves say they answered."""
    try:
        v = (task_out / BANK_FILE).read_text(encoding="utf-8").strip()
        if v:
            return v
    except OSError:
        pass
    _scripts()
    import judge
    return judge.answered_fingerprint(task_out.parent, task)


def _dm_answered(task_out: Path, task: str) -> bool:
    """A DeviceMark task whose saved answers cover every item the built task
    holds: done, as a task with lm_eval's results file is. Its scoring reads
    those answers alone (devicemark.records_from_samples), so a run the
    service restarted under (#167, at the 12z deploy) resumes at the task it
    was on instead of asking the battery again from its first item"""
    want = _devicemark.dm().task_keys(config.DM_TASKS_DIR, task)
    return bool(want) and want <= _devicemark.dm().answered_keys(task_out, task)


def _task_done(task_out: Path, task: str | None = None) -> bool:
    if not _has_results(task_out):
        return bool(task in config.DM_TASKS and _dm_answered(task_out, task))
    want = current_fingerprint(task) if task else None
    return want is None or answers_fingerprint(task_out, task) == want


def _answered_by(task_out: Path) -> int | None:
    try:
        return int(json.loads((task_out / ANSWERED_BY).read_text(encoding="utf-8"))["submission"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _set_aside(task_out: Path, task: str) -> Path:
    """Answers to another question set are not overwritten and not deleted:
    they move out of the results tree, whole, beside it — results/earlier/
    <model>/<task>_<n>shot-<fingerprint>-<time> — so nothing reads them as
    this task's answers again and nobody loses them."""
    fp = answers_fingerprint(task_out, task) or "unfingerprinted"
    dest = (config.OUT_DIR.with_name("earlier") / task_out.parent.name
            / f"{task_out.name}-{fp[:12]}-{int(time.time())}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task_out), str(dest))
    return dest


def reuse_note(reused: dict[str, int | None], tasks: list[str]) -> str:
    """What a judged row says when it answered nothing it had answered
    before: "answers reused from #46 (same questions) · re-graded". A reader
    of the queue should not have to infer it from GPU —."""
    exam = [t for t in tasks if t.startswith("exam_")]
    got = [t for t in exam if t in reused]
    if not got:
        return ""
    sids = sorted({reused[t] for t in got if reused[t]})
    src = f"from {', '.join(f'#{s}' for s in sids)}" if sids else "from an earlier run"
    what = "answers" if len(got) == len(exam) else f"answers for {len(got)} of {len(exam)} topics"
    return f"{what} reused {src} (same questions) · re-graded"


CANCELED = -15
TIMED_OUT = -3                # 15.3: the task ran past its limit
# 15.3: each run's limit for the task in hand, (seconds, why) — set by the run
# before it starts lm_eval and read by _run_task, so the many tests that stand
# in for _run_task keep their own signature
_TASK_LIMIT: dict[int, tuple[float, str]] = {}
PACE_NAME = "pace.json"


def model_pace(hf_id: str) -> dict | None:
    """15.3: the model's measured generation pace, {tok_s, sid, task, …} — its
    last task that wrote its answers in one go here (record_pace)"""
    try:
        p = json.loads((config.OUT_DIR / hf_id.replace("/", "__") / PACE_NAME)
                       .read_text(encoding="utf-8"))
        return p if float(p.get("tok_s") or 0) > 0 else None
    except (OSError, ValueError, TypeError):
        return None


def task_limit(n_left: int, cap: int, pace: dict | None) -> tuple[float, str] | None:
    """15.3: the limit a task that writes its answers is given — (seconds,
    why): the answers still to write, each at the cap, at the model's
    measured pace (else PACE_GUESS_TOK_S), with LIMIT_HEADROOM, never under
    TASK_TIMEOUT_S. None when TASK_TIMEOUT_S is 0: no limit"""
    if config.TASK_TIMEOUT_S <= 0:
        return None
    tok_s = float(pace["tok_s"]) if pace else config.PACE_GUESS_TOK_S
    src = (f"measured on #{pace['sid']}, {pace.get('task')}" if pace else
           "a guess: no run here has measured this model yet")
    sized = config.LIMIT_HEADROOM * max(0, n_left) * cap / tok_s
    why = (f"{max(0, n_left):,} answers at the {cap:,}-token cap at {tok_s:.0f} tokens a second "
           f"({src}), with {config.LIMIT_HEADROOM:g}× headroom")
    if sized < config.TASK_TIMEOUT_S:
        return float(config.TASK_TIMEOUT_S), why + f", and never under {hours_words(config.TASK_TIMEOUT_S)}"
    return sized, why


def _items_of(task: str, sub: dict) -> int:
    """15.3: how many answers a task that writes them asks: DeviceMark's from
    its built task, the generative three's from the board's counts"""
    if task in config.DM_TASKS:
        return len(_devicemark.dm().task_keys(config.DM_TASKS_DIR, task))
    if task == "mmlu_pro":
        n = sum(config.MMLU_PRO_SUBJECTS.values())
        return min(int(sub.get("subset") or 0), n) or n
    return _served.GEN_ITEMS.get(task, 0)


def cached_answers(task_out: Path) -> int:
    """the answers a task's cache holds (dm_cache): what a stopped task keeps"""
    import sqlite3
    n = 0
    for f in task_out.glob("lm-cache/*.db"):
        try:
            with sqlite3.connect(f"file:{f}?mode=ro", uri=True, timeout=5) as c:
                n += c.execute("SELECT COUNT(*) FROM unnamed").fetchone()[0]
        except sqlite3.Error:
            continue
    return n


def _answer_tokens(hf_id: str, revision: str | None, texts: list[str]) -> tuple[int, bool]:
    """the answers' tokens, counted with the model's tokenizer — or, when it
    can't be loaded, at four characters a token: (tokens, counted)"""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        return sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts), True
    except Exception:                                   # noqa: BLE001 — characters, then
        return sum(len(t) for t in texts) // 4, False


def record_pace(sid: int, hf_id: str, revision: str | None, task: str, task_out: Path,
                text: str) -> dict | None:
    """15.3: the model's pace from a task it answered in one go: its answers'
    tokens over the seconds lm_eval's bar took"""
    bar = _tqdm_last(text)
    files = sorted(task_out.rglob(f"samples_{task}_*.jsonl"))
    if not bar or not bar[0] or bar[0] != bar[1] or not bar[2] or not files:
        return None
    texts = []
    for line in files[-1].read_text(encoding="utf-8").splitlines():
        try:
            texts.append(((json.loads(line).get("resps") or [[""]])[0] or [""])[0] or "")
        except ValueError:
            continue
    if len(texts) != bar[0]:
        return None
    tokens, counted = _answer_tokens(hf_id, revision, texts)
    out = {"tok_s": round(tokens / bar[2], 2), "sid": sid, "task": task, "tokens": tokens,
           "seconds": bar[2], "answers": bar[0], "counted": counted, "at": time.time()}
    p = config.OUT_DIR / hf_id.replace("/", "__") / PACE_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    return out


def _run_task(sid: int, cmd: list[str], lf, env: dict, run_as, cwd: Path,
              on_poll=None) -> int:
    """One lm_eval task, as a child we watch: its exit code, -1 on timeout, or
    CANCELED when someone asked the queue to stop this run. Polled every two
    seconds, so a cancel costs at most that plus a clean shutdown. `cwd` is
    lm_eval_cwd(), never BENCH_ROOT. `on_poll()` is called at each poll: a
    served run's progress, read from the harness's own progress bar. 12m.3:
    what it returns, when it isn't empty, is why the child stops now (a model
    from OpenRouter at the month's AI limit)"""
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd,
                            env=env, **({"user": run_as[0], "group": run_as[1]}
                                        if run_as else {}))
    t0 = time.time()
    # 15.3: the limit the run sized for this task, else TASK_TIMEOUT_S; none
    # at 0 (15.1: a rented GPU has nothing else queued)
    limit = _TASK_LIMIT.get(sid)
    limit_s = limit[0] if limit else (config.TASK_TIMEOUT_S if config.TASK_TIMEOUT_S > 0 else 0)
    while True:
        try:
            return proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        stop = (on_poll() or "") if on_poll else ""
        took = time.time() - t0
        timed = bool(limit_s) and took > limit_s
        why = ("canceled by request" if db.cancel_requested(sid)
               else stop or (f"timed out after {hours_words(took)}: its limit, "
                             f"{hours_words(limit_s)}, was "
                             + (limit[1] if limit else "TASK_TIMEOUT_S") if timed else ""))
        if why:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            lf.write(f"\n[service] {why}\n")
            return (CANCELED if why.startswith("canceled") else TIMED_OUT if timed and not stop
                    else -1)


# ---------------------------------------------------------------------------
# 12f.1: a model served elsewhere
# ---------------------------------------------------------------------------

def _ask_served(sid: int, rec: dict, meta: dict, task: str, task_out: Path, label: str,
                log_path: Path, everyday: bool, asked: list[str] | None, safety: bool = False,
                meter: _served.Meter | None = None, evd_dir: Path | None = None,
                trust_dir: Path | None = None, sq_dir: Path | None = None,
                mab_dir: Path | None = None, odd: dict | None = None,
                th: dict | None = None):
    """one Everyday, exam or Trust & safety task, asked over the model's
    server. (status, stopped): 0 or CANCELED, and a ServerStopped when it
    stopped answering — what it answered before that is written and kept.
    12k.2: Do-Not-Answer and XSTest are asked as Everyday is, as typed.
    12m.3: a model from OpenRouter's questions go through its run's meter,
    and its progress says the running total. 12s: `odd` is the run's count
    of questions the server failed on — asked the raw way, or with no answer
    — and each is a line of the log"""
    if safety:
        items = (trust_dir or config.TRUST_TASKS_DIR) / f"{task}.jsonl"      # build_tasks'
        everyday = True
    elif task == config.SIMPLEQA_TASK:
        # 12n.2: asked as typed, with the Everyday settings, as Trust & safety is
        items = (sq_dir or config.SIMPLEQA_TASKS_DIR) / f"{task}.jsonl"      # build_tasks'
        everyday, safety = True, True
    elif task in (config.MMP_TASK, config.MMF_TASK):
        # 14.3: Mobile-MMLU-Pro, asked the authors' way — a letter, as typed.
        # 14.4: and the full Mobile-MMLU, from build_full_tasks'
        items = (config.MMP_TASKS_DIR if task == config.MMP_TASK
                 else config.MMF_TASKS_DIR) / f"{task}_ask.jsonl"
        everyday, safety = True, True
    elif task in config.MAB_ALL:
        # 12o.3: MobileAIBench's prompt as typed, its system line as the system
        # message, with the Everyday settings. 14.1: MT-Bench's second turn
        # from the model's own folder (build_turn2), its conversation so far
        # asked as messages
        items = (mab_dir or config.MAB_TASKS_DIR) / f"{task}.jsonl"        # build_tasks'
        everyday, safety = True, True
    elif everyday:
        items = (evd_dir or config.EVERYDAY_TASKS_DIR) / f"{config.EVERYDAY_TASK}.jsonl"
    else:
        items = config.JUDGED_TASKS_DIR / f"{task}.jsonl"                     # exam_build's
    docs = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if everyday and not safety and asked is not None:
        docs = [d for d in docs if d.get("id") in set(asked)]
    if task in (config.MMP_TASK, config.MMF_TASK):
        # 14.4: only the questions the model has no pick for (one run, two scores)
        import mobile_mmlu as _mmp
        docs = [docs[i] for i in _mmp.to_ask(task_out.parent, task) if i < len(docs)]
    s = _served.settings_for(rec, meta, everyday)
    # 16b: the Mobile suite and Frontier's SimpleQA — the switch, said out loud
    if th is not None and th.get("mode") == "switch" and not _served.is_openrouter(rec):
        s["chat_template_kwargs"] = {"enable_thinking": bool(th["on"])}
    if task in config.MAB_ALL:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import mobileaibench as _mab
        s["system"] = _mab.SYSTEM[task]
        if task in config.MAB_JUDGED_TASKS:
            # 14.1: MT-Bench's answers get FastChat's room, 1,024 tokens
            s["max_tokens"] = max(int(s["max_tokens"]), _mab.MTB_MAX_GEN_TOKS)
    with open(log_path, "a") as lf:
        lf.write(f"\n===== [{sid}] {task} · served: {rec['name']} at {rec['base_url']} · "
                 f"{rec['pin'].get('version') or rec['pin'].get('file')} · {len(docs)} "
                 f"question(s), {_served.concurrency(rec)} at a time · {json.dumps(s)} =====\n")

    def progress(done: int, total: int, each: float) -> None:
        db.update(sid, status="running", progress=f"{label} · {time_left(done, total, each)}"
                  + (f" · {meter.line()}" if meter else ""))
    here: dict = {}
    try:
        status = _served.answer_task(rec, task, docs, task_out, s, everyday,
                                     on_progress=progress,
                                     canceled=lambda: db.cancel_requested(sid), meter=meter,
                                     odd=here)
        return status, None
    except _served.ServerStopped as e:
        e.task = task
        with open(log_path, "a") as lf:
            lf.write(f"\n[service] {task}: {e} ({e.why}); the {e.done} answered are kept\n")
        return 0, e
    finally:
        if here.get("lines"):
            with open(log_path, "a") as lf:
                lf.write("".join(f"[service] {line}\n" for line in here["lines"]))
        if odd is not None:
            for k in ("raw_fallback", "errors"):
                odd[k] = odd.get(k, 0) + here.get(k, 0)


_TQDM = re.compile(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<")


def _tqdm_last(text: str) -> tuple[int, int, float] | None:
    """the harness's last progress bar in `text`: (done, total, seconds so far)"""
    m = None
    for m in _TQDM.finditer(text):
        pass
    if not m:
        return None
    h, mi, se = int(m.group(3) or 0), int(m.group(4)), int(m.group(5))
    return int(m.group(1)), int(m.group(2)), float(h * 3600 + mi * 60 + se)


def _served_poll(sid: int, label: str, log_path: Path, mark: int,
                 meter: _served.Meter | None = None) -> str:
    """a served generative task's progress, from lm_eval's own bar — 12m.3:
    with the running total of a model from OpenRouter, and, once its meter
    has stopped at the month's AI limit, why lm_eval stops now"""
    try:
        size = log_path.stat().st_size
        got = _tqdm_last(_read_from(log_path, max(mark, size - 4096)))
    except OSError:
        got = None
    if got and got[0]:
        done, total, secs = got
        db.update(sid, progress=f"{label} · {time_left(done, total, secs / done)}"
                  + (f" · {meter.line()}" if meter else ""))
    elif meter:
        db.update(sid, progress=f"{label} · {meter.line()}")
    return "stopped at this month's AI limit" if meter and meter.stopped else ""


_SERVER_GONE = re.compile(r"ConnectionError|Connection refused|ClientConnectorError|"
                          r"ServerDisconnected|RetryError|Cannot connect|ConnectTimeout|"
                          r"ReadTimeout|RemoteDisconnected")


def _served_stopped_in_log(text: str) -> _served.ServerStopped | None:
    """lm_eval gave up on the server: at which answer, from its bar"""
    if not _SERVER_GONE.search(text):
        return None
    got = _tqdm_last(text) or (0, 0, 0.0)
    return _served.ServerStopped(got[0], got[1], "lm_eval could not reach it")


def run_submission(sub: dict) -> None:
    sid = sub["id"]
    everyday = sub["suite"] == "everyday"
    generative = sub["suite"] == "generative"
    safety = sub["suite"] == "safety"            # 12k.2: Do-Not-Answer and XSTest
    # 12n.2: GPQA Diamond's chain of thought, asked as the generative three are,
    # and SimpleQA Verified, asked as Trust & safety is and graded by the judge
    shared = sub["suite"] == "shared"
    # 12o.3: MobileAIBench's HotpotQA and SQL, asked as SimpleQA is and scored
    # by MobileAIBench's own metrics, with no judge
    mobile = sub["suite"] == "mobile"
    # 12q: DeviceMark's battery — a served setup asked over its server
    # (service/devicemark.py), a Hugging Face model as three tasks on hf
    devicemark = sub["suite"] == "devicemark"
    # 17: the Frontier benchmarks — a served model's, asked over its server
    # (service/frontier.py), on the board or on a rented GPU
    frontier = sub["suite"] == "frontier"
    dm_plan = None                       # 12q.G: a Hugging Face model's (hf_plan)
    # 12f.0: a run that can't save doesn't start — in the status dot's words
    from . import disk
    why = disk.blocks_run()
    if why:
        db.update(sid, status="failed", finished_at=time.time(), error=why)
        return
    # 16.5: a judged run queued before the Knowledge exam was switched off: its
    # tasks are all the exam's, so it skips them all and says so
    if sub.get("suite") == "judged" and not config.KNOWLEDGE_EXAM:
        db.update(sid, status="failed", finished_at=time.time(),
                  error=config.EXAM_OFF + ". Its exam tasks were skipped: nothing was run.")
        return
    # 12p.1: nor an Everyday run without its hidden set — in the red banner's words
    if everyday:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import everyday as _everyday
        why = _everyday.hidden_status()["why"]
        if why:
            db.update(sid, status="failed", finished_at=time.time(), error=why)
            return

    # 12f.1: a model served elsewhere — asked over its server, never loaded here
    srv = _served.is_served(sub["hf_id"])

    # -- preflight: metadata only, no GPU, seconds --------------------------------
    try:
        if srv:
            # its registration, the suites it can sit, and the file its server
            # serves now against the one registered: a different file stops here
            meta = _served.preflight(sub)
            if frontier and _served.is_openrouter(meta.get("served")):
                raise PreflightError(config.FRONTIER_NOT_OPENROUTER)
        elif frontier:
            raise PreflightError(config.FRONTIER_SERVED_ONLY)
        else:
            # the submitter's kind is passed in: 'auto' is resolved here, and
            # refused when it is genuinely ambiguous rather than guessed. 12a:
            # the pilot asks through the chat template whatever kind the board
            # lists — a model with one is asked as a person would ask it, a
            # model without one cannot be
            meta = preflight(sub["hf_id"], "instruct" if everyday else sub["kind"],
                             allow_remote_code=bool(sub.get("allow_remote_code")))
            if everyday and not meta.get("has_template"):
                raise PreflightError(config.NO_CHAT_TEMPLATE)
            # 12h.1: asked through the chat template and scored on what it
            # writes, so only an instruct model can sit them fairly
            if generative and meta["kind"] != "instruct":
                raise PreflightError(config.GEN_INSTRUCT_ONLY)
            if safety and meta["kind"] != "instruct":
                raise PreflightError(config.SAFETY_INSTRUCT_ONLY)
            if shared and meta["kind"] != "instruct":
                raise PreflightError(config.SHARED_INSTRUCT_ONLY)
            if devicemark and meta["kind"] != "instruct":
                raise PreflightError(config.DM_INSTRUCT_ONLY)
            if devicemark:
                # 12q.G: the length lm_eval is told, and the batch and memory
                # of answers 4,096 tokens long — or a refusal, when the model
                # reads fewer tokens than the protocol needs
                dm_plan = _devicemark.hf_plan(sub, meta, gen_thinking(sub, meta),
                                              load_spec(sub["hf_id"], meta)["pretrained"])
                meta["batch"], meta["need_gb"] = dm_plan["batch"], dm_plan["need_gb"]
    except PreflightError as e:
        # 16.8: what a person needn't read in the run's line — an address, an
        # error code — goes in its log
        if getattr(e, "detail", ""):
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.LOGS_DIR / f"service_{sid}_{sub['hf_id'].replace('/', '__')}.log",
                      "a", encoding="utf-8") as lf:
                lf.write(f"preflight: {e.detail}\n")
        db.update(sid, status="failed", error=str(e), finished_at=time.time())
        return
    rec = meta.get("served")
    kind = meta["kind"]
    remote_code = bool(meta.get("remote_code"))
    # 12m.3: a model from OpenRouter — each answer's cost counted as it lands,
    # against the month's AI limit
    est = meta.get("estimate") or {}
    meter = _served.Meter(rec, sid, est.get("usd", 0.0)) if _served.is_openrouter(rec) else None
    # 12m.3: and a lane of its own — it never touches the GPU, so it neither
    # waits for nor holds the run lock, and writes its tasks in a folder of its
    # own: a GPU run beside it reads its own, never rewritten under it
    remote = meter is not None
    evd_dir = config.BENCH_ROOT / "remote" / "everyday-tasks" if remote else config.EVERYDAY_TASKS_DIR
    trust_dir = config.BENCH_ROOT / "remote" / "trust-tasks" if remote else config.TRUST_TASKS_DIR
    sq_dir = config.BENCH_ROOT / "remote" / "simpleqa-tasks" if remote else config.SIMPLEQA_TASKS_DIR
    mab_dir = config.BENCH_ROOT / "remote" / "mab-tasks" if remote else config.MAB_TASKS_DIR
    db.update(sid, kind=kind, params=meta["params"], vocab=meta["vocab"],
              batch=meta["batch"], need_gb=meta["need_gb"],
              arch=json.dumps(meta.get("archinfo") or {}),
              progress=f"preflight ok · batch={meta['batch']} · "
                       f"needs ~{meta['need_gb']:g} GB" if not rec else
                       f"preflight ok · via OpenRouter: {rec['pin']['version']} · this run "
                       f"{est.get('line', '')}" if meter else
                       f"preflight ok · served elsewhere: {rec['pin'].get('file') or rec['name']}")

    tasks = config.tasks_for_suite(sub["suite"], bbq_all=bool(sub.get("bbq_all")),
                                   part=sub.get("part") or "")
    judged = set(config.judged_tasks())
    # a judged run narrowed to one topic: the same suite, fewer tasks. The
    # judge below grades only these, so a person can sit one topic in minutes
    # instead of the whole exam
    only = [t for t in json.loads(sub.get("tasks") or "[]") if t in tasks]
    if only:
        tasks = only
    safe = sub["hf_id"].replace("/", "__")
    log_path = config.LOGS_DIR / f"service_{sid}_{safe}.log"
    # 12h.1: a thinking-on run of a model that can turn thinking off is a row
    # of its own, "Qwen3.5-2B · thinking": its answers live apart, so nothing
    # ever averages them with the thinking-off ones
    th = suite_thinking(sub, meta, rec) if generative or shared or mobile or frontier else None
    if devicemark:
        # 12q: thinking off unless asked, said out loud either way — a served
        # setup's too, whatever it was registered with — and a thinking-on
        # run a row of its own; the cap, thinking included, is the protocol's
        th = {**({"mode": "switch", "on": bool(sub.get("thinking")),
                  "separate": bool(sub.get("thinking")), "think_end": "</think>"}
                 if rec else gen_thinking(sub, meta)), "budget": _devicemark.dm().CAP}
    row_safe = safe + "__thinking" if th and th["separate"] else safe
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # drop the model's shape next to its results, so the report (live OR the
    # frozen single-file kind) can show architecture/hidden/layers/vocab without
    # any access to the service database
    meta_dir = config.OUT_DIR / safe
    meta_dir.mkdir(parents=True, exist_ok=True)
    # …except that the pilot's kind is not the board's: it applies the chat
    # template to every model it asks, and a model listed as base stays base
    if not (everyday and (meta_dir / "model_meta.json").exists()):
        (meta_dir / "model_meta.json").write_text(json.dumps(
            {"model": sub["hf_id"], "kind": kind, "params": meta["params"],
             "kind_reason": meta.get("kind_reason"),
             **(meta.get("archinfo") or {})}), encoding="utf-8")
    if row_safe != safe:
        (config.OUT_DIR / row_safe).mkdir(parents=True, exist_ok=True)
        (config.OUT_DIR / row_safe / "model_meta.json").write_text(json.dumps(
            {"model": sub["hf_id"] + " · thinking", "base_model": sub["hf_id"], "kind": kind,
             "params": meta["params"], "kind_reason": meta.get("kind_reason"),
             **(meta.get("archinfo") or {})}), encoding="utf-8")
    backend, not_vllm = (gen_backend() if (generative or shared or devicemark) and not rec else
                         (_served.BACKEND, "") if generative or shared else ("hf", ""))
    fell_back = ""
    # 12q.E: the batch a generative task runs at on hf, once one ran out of GPU
    # memory and was tried again at half: the run's later tasks start there
    gen_batch = None
    if everyday:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import everyday as _everyday
        _everyday.build_task(evd_dir)
    if set(tasks) & set(config.TRUST_TASKS):
        # 12k.2: BBQ, Do-Not-Answer and XSTest, from the pinned files
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import trust_safety as _trust
        _trust.build_tasks(trust_dir)
    if config.SIMPLEQA_TASK in tasks:
        # 12n.2: SimpleQA Verified, from the pinned file
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import simpleqa as _sq
        _sq.build_tasks(sq_dir)
    if set(tasks) & set(config.MAB_ALL):
        # 12o.3: MobileAIBench's samples, from the pinned files (14.1: and
        # MT-Bench's first turn; its second is built from the model's answers)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import mobileaibench as _mab
        _mab.build_tasks(mab_dir)
        # 14.2: Privacy Leakage is fetched at deploy: without it the trust part waits
        missing = [t for t in tasks if t in _mab.TRUST and _mab.available(t)]
        if missing:
            db.update(sid, status="failed", finished_at=time.time(),
                      error=_mab.available(missing[0]) + ". Nothing was asked.")
            return
    if config.MMP_TASK in tasks or config.MMF_TASK in tasks:
        # 14.3: Mobile-MMLU-Pro, from the file the data step fetched — every
        # question, never our key. 14.4: the full Mobile-MMLU from its 80, and
        # Pro's wording of the question the two word apart
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import mobile_mmlu as _mmp
        try:
            why = _mmp.available()
            if not why and config.MMF_TASK in tasks:
                why = _mmp.full_available()
            if not why:
                _mmp.build_tasks(config.MMP_TASKS_DIR)
                if config.MMF_TASK in tasks:
                    _mmp.build_full_tasks(config.MMF_TASKS_DIR)
        except (OSError, ValueError) as e:
            why = str(e)
        if why:
            db.update(sid, status="failed", finished_at=time.time(),
                      error=why + ". Nothing was asked.")
            return
    if devicemark and not rec:
        # 12q: the battery's questions from the pinned datasets, as three tasks
        try:
            items = _devicemark.dm().load_items(config.DM_ITEMS)
            # 15.6: the battery the repo expects, or nothing is asked
            why = _devicemark.dm().items_differ(items, config.DM_ITEMS_SHA256)
            if not why:
                # 15.5: on a rented GPU run as a shard, its share of each task alone
                _devicemark.dm().build_tasks(config.DM_TASKS_DIR, items,
                                             shard=_devicemark.dm().parse_shard(config.DM_SHARD))
        except Exception as e:                          # noqa: BLE001 — said on the row
            db.update(sid, status="failed", finished_at=time.time(),
                      error=f"the battery's questions could not be read: {e}")
            return
        if why:
            db.update(sid, status="failed", finished_at=time.time(), error=why)
            return
    if remote_code:      # the code that produced the scores is part of the record
        db.update(sid, progress=f"preflight ok · custom model code · batch={meta['batch']}")
    # 12h.1: a Hub model on the approved list runs its own code offline, as
    # an uploaded one does: its pinned commit is fetched here, before the job
    # drops privileges and loses the network
    if meta.get("revision") and not sub["hf_id"].startswith("local/"):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(sub["hf_id"], revision=meta["revision"])
        except Exception as e:                          # noqa: BLE001 — said on the row
            db.update(sid, status="failed", finished_at=time.time(),
                      error=f"could not fetch {sub['hf_id']} at the approved commit "
                            f"{meta['revision'][:12]}: {e}")
            return

    # -- one run at a time: wait for the shared lock ------------------------------
    # 12d.1: the Playground lets go of the GPU first — its replies there stop,
    # its models unload — within CHAT_YIELD_WAIT_S; chat never takes the lock
    held = False
    if not remote:
        try:
            from . import chat as _chat
            _chat.ENGINE.yield_gpu()
        except Exception:                              # noqa: BLE001 — a run never waits on chat
            pass
        if not wait_for_lock(sid):
            return
        held = True
    relay = None          # 12m.3: a model from OpenRouter's generative three go through it

    # 12d.3: the lock is this run's, so a new chat message to the model it
    # tests waits (chat.Engine.place); the replies already streaming to it
    # stop first, so no chat request is interleaved with a scored one
    if rec:
        try:
            from . import chat as _chat
            _chat.ENGINE.yield_served(sub["hf_id"])
        except Exception:                              # noqa: BLE001 — a run never waits on chat
            pass

    try:
        # 12q: a served setup's devicemark run — the battery, the pilot, the
        # parity check or the speed test — asked over its server, in the lock
        if devicemark and rec:
            try:
                status, line = _devicemark.run(sid, sub, rec, log_path)
            except Exception as e:                      # noqa: BLE001 — said on the row
                status, line = "failed", f"devicemark: {e}"
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] devicemark: {e!r}\n")
            db.update(sid, status=status, finished_at=time.time(), progress=line,
                      error="" if status == "done" else line)
            return
        # 17: a served model's Frontier run — each question asked over its
        # server, the answers kept as they land, and scored when all are in
        if frontier:
            from . import frontier as _frontier
            try:
                status, line = _frontier.run(sid, sub, rec, th, config.OUT_DIR / row_safe,
                                             log_path)
            except Exception as e:                      # noqa: BLE001 — said on the row
                status, line = "failed", f"frontier: {e}"
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] frontier: {e!r}\n")
            db.update(sid, status=status, finished_at=time.time(), progress=line,
                      error="" if status == "done" else line)
            return
        # -- wait for GPU memory, then run the missing tasks -----------------------
        need_mib = int(meta["need_gb"] * 1024) + config.FREE_MARGIN_MIB
        need_b = gpu.mib_to_bytes(need_mib)
        t0 = time.time()
        # 12f.1: a served model's memory is its server's: nothing to wait for here.
        # 16.2: in GB, "GPU memory", as everywhere; and a card that can't be read
        # is said in one line, never an internal error
        while not rec:
            try:
                free = gpu_free_mib()
            except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
                db.update(sid, status="failed", finished_at=time.time(),
                          error=f"{gpu.WHY_NONE} (nvidia-smi: {str(e)[:120]}). Nothing was run.")
                return
            if free >= need_mib:
                break
            if db.cancel_requested(sid):
                db.update(sid, status="canceled", finished_at=time.time(),
                          progress="canceled by request while waiting for GPU memory")
                return
            db.update(sid, status="waiting_gpu",
                      progress=gpu.waiting_line(need_b, gpu.mib_to_bytes(free)))
            if time.time() - t0 > config.GPU_WAIT_MAX_S:
                db.update(sid, status="failed", finished_at=time.time(),
                          error=f"gave up after {config.GPU_WAIT_MAX_S // 3600} h waiting for "
                                f"{gpu.gb_text(need_b)} of GPU memory.")
                return
            time.sleep(config.GPU_POLL_S)

        # resolve the job identity ONCE, before any task: a broken sandbox should
        # fail the submission with a service error, not four identical tracebacks
        # blamed on the submitter's model
        run_as = _eval_ids() if remote_code else None
        if remote_code and not run_as and os.getuid() == 0:
            # fail closed: the whole point of the gate is that uploaded code does
            # NOT run as root, so an unresolvable EVAL_USER stops the job rather
            # than quietly becoming the thing we were guarding against
            db.update(sid, status="failed", finished_at=time.time(),
                      error=f"refusing to run custom model code as root: "
                            f"EVAL_USER={config.EVAL_USER!r} does not resolve to an "
                            f"account on this machine.")
            release_lock()
            return
        scratch = _job_scratch(sid, run_as) if remote_code else None
        job_env = _child_env(remote_code, scratch)
        if dm_plan:
            # 12q.G: PyTorch's own advice in its out-of-memory message, for a
            # run whose big tensors come and go (Granite's, layer after layer):
            # what it freed is given back in pieces the next one can use
            job_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if remote_code:
            broken = _env_canary(job_env, run_as)
            if broken:
                db.update(sid, status="failed", finished_at=time.time(),
                          error=f"the job environment is broken, not your model: "
                                f"{broken}. Nothing of yours was loaded — this fails "
                                f"identically for every custom-code submission. Tell "
                                f"the operator (SERVICE.md § custom model code).")
                release_lock()
                shutil.rmtree(scratch, ignore_errors=True)
                return

        gpu_seconds = 0.0
        failed_tasks: list[str] = []
        canceled = False
        reused: dict[str, int | None] = {}
        asked: list[str] | None = None
        stopped: _served.ServerStopped | None = None       # 12f.1: the server stopped
        odd: dict = {}             # 12s: the questions its server failed on, counted
        if (generative or shared) and meter:
            relay = _served.Relay(rec, meter).__enter__()
        elif (generative or shared) and rec and th and th["mode"] == "switch":
            # 16b: lm_eval sends no chat_template_kwargs: the relay adds the switch
            relay = _served.Relay(rec, extra={"chat_template_kwargs": {
                "enable_thinking": bool(th["on"])}}).__enter__()
        for i, task in enumerate(tasks, 1):
            if canceled or db.cancel_requested(sid):
                canceled = True
                break
            # 12n.2: a task asked as the generative three are — GPQA's chain of thought.
            # 12o.1: and in a thinking run, SimpleQA too: its answers are the
            # thinking row's, what follows the thinking
            gen_task = generative or devicemark or (
                shared and (task == config.GPQA_COT or th["separate"]))
            shots = config.NFEWSHOT.get(task, 0)
            if shots and _restricted(task):
                # 14.4.4: a restricted set's questions are never few-shot examples
                failed_tasks.append(task)
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] {task}: {_restricted(task)} — asked 0-shot only; "
                             f"not asked\n")
                continue
            task_out = config.OUT_DIR / row_safe / f"{task}_{shots}shot"
            label = f"{i}/{len(tasks)} · {task} ({shots}-shot)"
            turn2_dir = None
            if task == config.MAB_PRIVACY and not rec:
                # 14.2: each question's own system line, in the model's chat
                # template as text, in a folder of this model's
                turn2_dir = config.BENCH_ROOT / "mobileaibench" / "rendered" / row_safe
                try:
                    _mab.build_rendered(turn2_dir, task, _chat_renderer(sub["hf_id"], meta))
                except Exception as e:                  # noqa: BLE001 — said on the row
                    failed_tasks.append(task)
                    with open(log_path, "a") as lf:
                        lf.write(f"\n[service] {task}: the questions could not be built: {e!r}\n")
                    continue
            if task == config.MAB_MTB2:
                # 14.1: the second turn after the model's own first answer — as
                # messages to a server, or in the model's own chat template as
                # text for lm_eval, in a folder of this model's
                turn2_dir = config.BENCH_ROOT / "mobileaibench" / "turn2" / row_safe
                try:
                    _mab.build_turn2(turn2_dir, config.OUT_DIR / row_safe,
                                     None if rec else _chat_renderer(sub["hf_id"], meta))
                except Exception as e:                  # noqa: BLE001 — said on the row
                    failed_tasks.append(task)
                    with open(log_path, "a") as lf:
                        lf.write(f"\n[service] {task}: the second turns could not be built: "
                                 f"{e!r}\n")
                    continue
            if everyday:
                # 12a.5: the model is asked only what it has no answer to on
                # today's words; the answers it gave before are kept (beside
                # its marks, first) and marked again, with no GPU
                todo = _everyday.unanswered(config.OUT_DIR / safe)
                asked = [q["id"] for q in todo]
                if not todo:
                    with open(log_path, "a") as lf:
                        lf.write(f"\n[service] {task}: every question is answered already; "
                                 f"marking them again\n")
                    db.update(sid, status="running",
                              progress=f"{label} — no new questions, marking again")
                    continue
                _everyday.build_task(evd_dir, only=asked)
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] {task}: asking the {len(todo)} question(s) it has "
                             f"no answer to on today's words\n")
                if _has_results(task_out):
                    # the last run's folder moves beside the tree, whole; its
                    # answers are kept already
                    moved = _set_aside(task_out, task)
                    with open(log_path, "a") as lf:
                        lf.write(f"\n[service] {task}: the last run's folder is kept at "
                                 f"{moved}\n")
            if (devicemark and not rec and backend == "hf" and th and th.get("on")
                    and saved_cut(task_out, task)):
                # 15.7a: answers saved before 15.7, cut by lm_eval at the end of
                # the thinking (or with their markers dropped), are no answers
                # to keep: set aside, and asked again, whole
                moved = set_aside_cut(task_out)
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] {task}: its replies were saved before 15.7, cut at "
                             f"the end of the thinking — kept at {moved}, asking again with "
                             f"each reply whole\n")
            if th is not None and th["mode"] == "switch" and (
                    mobile or (rec and (generative or shared))):
                # 16b: answers made under the other thinking setting are never reused
                meta_think = None
                if not th["on"]:
                    meta_think = {"model": sub["hf_id"] + " · thinking",
                                  "base_model": sub["hf_id"], "kind": kind,
                                  "params": meta.get("params"),
                                  "kind_reason": meta.get("kind_reason"),
                                  **(meta.get("archinfo") or {})}
                unmarked = unmarked_of(rec, gen_task)
                was = answered_thinking(task_out, unmarked)
                got = sort_thinking(task_out, task, bool(th["on"]),
                                    None if th["on"] else config.OUT_DIR / (safe + "__thinking"),
                                    meta_think, unmarked)
                if got:
                    with open(log_path, "a") as lf:
                        lf.write(f"\n[service] {task}: its answers on disk were made "
                                 + ("as its server decided" if was == "default"
                                    else f"with thinking {was}")
                                 + f", and this run asks with it {'on' if th['on'] else 'off'} "
                                 f"— moved to "
                                 f"{'its thinking row' if got[0] == 'thinking row' else 'earlier'} "
                                 f"({got[1]}), asking again\n")
            mm_left = None
            if task in (config.MMP_TASK, config.MMF_TASK):
                # 14.4: one run, two scores — a Mobile-MMLU task asks only the
                # questions the model has no pick for, by its runs of either set
                mm_left = _mmp.to_ask(config.OUT_DIR / row_safe, task)
                if not mm_left:
                    with open(log_path, "a") as lf:
                        lf.write(f"\n[service] {task}: every question answered already, by "
                                 f"this model's Mobile-MMLU runs; not asked again\n")
                    db.update(sid, status="running", progress=f"{label} — already done")
                    continue
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] {task}: {len(mm_left):,} question(s) to ask\n")
            if mm_left is None and _task_done(task_out, task):
                if current_fingerprint(task):
                    reused[task] = _answered_by(task_out)
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] {task}: answered already ({task_out.name}); not "
                             f"asked again\n")
                db.update(sid, status="running", progress=f"{label} — already done")
                continue
            if mm_left is None and _has_results(task_out):
                # answers on disk, to a question set the task no longer holds
                moved = _set_aside(task_out, task)
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] {task}: the answers on disk are to other questions "
                             f"than the task holds now; kept at {moved}, answering again\n")
            db.update(sid, status="running", progress=label)
            if th is not None and th["mode"] == "switch" and (
                    mobile or (rec and (generative or shared))):
                mark_thinking(task_out, th)              # 16b

            if rec and not gen_task:
                # 12f.1: Everyday and the exam, asked over the server with the
                # settings a local run uses, into the files lm_eval writes
                t_task = time.time()
                status, stopped = _ask_served(sid, rec, meta, task, task_out, label, log_path,
                                              everyday, asked, safety=safety, meter=meter,
                                              evd_dir=evd_dir, trust_dir=trust_dir,
                                              sq_dir=sq_dir, mab_dir=turn2_dir or mab_dir,
                                              odd=odd, th=th if (mobile or shared) else None)
                gpu_seconds += time.time() - t_task
                db.update(sid, gpu_seconds=gpu_seconds)
                if status == CANCELED:
                    canceled = True
                    break
                if status == 0 and not stopped and current_fingerprint(task):
                    (task_out / BANK_FILE).write_text(current_fingerprint(task) + "\n",
                                                      encoding="utf-8")
                    (task_out / ANSWERED_BY).write_text(json.dumps(
                        {"submission": sid, "at": time.time()}), encoding="utf-8")
                if stopped:
                    break          # every task after this one asks the same server
                continue

            # 12d.1: how the model loads, said once (load_spec) — the Playground
            # loads through the same decision
            spec = load_spec(sub["hf_id"], meta)
            pretrained = spec["pretrained"]
            margs = model_args(spec)
            # 16b: the Mobile suite asked as a chat — the switch said out loud,
            # as the generative three's (gen_model_args). Mobile-MMLU is asked
            # with no chat template, the paper's way, where it switches nothing:
            # said all the same, so a thinking run's results are its row's
            if mobile and th and th["mode"] == "switch":
                margs += f",enable_thinking={th['on']}"
                if th["on"]:
                    margs += f",think_end_token={th['think_end']}"
            # 11l: a reasoning model thinks before it answers, and 256 tokens
            # ran out inside the thinking on every question of run #60. Only
            # its judged answers get more room, and only when its template is
            # the one that thinks; lm_eval records the override in its results
            sq = task == config.SIMPLEQA_TASK or task in config.MAB_ALL
            thinks = ((kind == "instruct" and task in judged or everyday or safety or sq)
                      and (meta.get("archinfo") or {}).get("reasoning_template"))
            # 12a.4: its everyday answers get more room still — 12d.1: said
            # by everyday.run_settings, the function the Playground reads too.
            # 12k.2: Trust & safety is asked with the Everyday settings
            room = (_everyday_settings(meta)["max_gen_toks"] if everyday or safety or sq
                    else _exam_settings(meta)["max_gen_toks"]) if thinks else None
            # 14.1: MT-Bench's second turn is the conversation already in the
            # model's chat template: asked as it stands
            # 14.3: Mobile-MMLU-Pro as the paper ran lm-evaluation-harness, with no
            # chat template; 14.4: the full set too, and only what's left to ask
            cmd = lm_eval_cmd(margs, task, shots, meta["batch"], task_out,
                              chat=(kind == "instruct" or everyday)
                              and not (task == config.MAB_MTB2 or turn2_dir is not None)
                              and task not in (config.MMP_TASK, config.MMF_TASK),
                              max_gen_toks=room, include_dir=turn2_dir,
                              samples=_mmp.samples_file(task_out, task, mm_left)
                              if mm_left else None,
                              system=(_mab.SYSTEM.get(task) or None) if task in config.MAB_ALL
                              else None)
            if gen_task:
                def gen_cmd(be, batch=None):
                    samples = None
                    if task == "mmlu_pro" and int(sub.get("subset") or 0) > 0:
                        samples = task_out / "subset.json"
                        samples.parent.mkdir(parents=True, exist_ok=True)
                        samples.write_text(json.dumps(mmlu_pro_subset(int(sub["subset"]))),
                                           encoding="utf-8")
                    if rec:
                        # 12f.1: through lm_eval's local-chat-completions; the
                        # answers it has are kept in its cache, by what was served
                        return lm_eval_cmd(
                            _served.lm_eval_model_args(rec, relay), task, shots, 1, task_out,
                            chat=True, max_gen_toks=th["budget"], backend=be,
                            samples=samples,
                            cache=_served.cache_path(rec, task, th["on"]
                                                     if th["mode"] == "switch" else None))
                    return lm_eval_cmd(
                        gen_model_args(pretrained, th, backend=be, remote_code=remote_code,
                                       revision=meta.get("revision"),
                                       gpu_util=gpu_util_for(gpu_free_mib(), gpu_total_mib())
                                       if be == "vllm" else None,
                                       # 12q.G: a DeviceMark run's own; 12t: else
                                       # the limit lm_eval can't find, if any
                                       max_length=dm_plan["max_length"] if dm_plan
                                       else spec["max_length"]),
                        task, shots, batch or gen_batch or meta["batch"], task_out,
                        chat=True, max_gen_toks=th["budget"], backend=be, samples=samples,
                        # 15.1: a DeviceMark task resumes per answer; 15.3: any
                        # task that writes its answers here
                        cache=dm_cache(task_out),
                        # 15.7: a DeviceMark reply with the thinking on, kept whole
                        whole=bool(devicemark and be == "hf" and th.get("on")))
                cmd = gen_cmd(backend)
                if devicemark and not rec and backend == "hf" and th.get("on"):
                    keep_whole(task_out)

            t_task = time.time()
            # the dropped-privilege child cannot create its own output dir under
            # a root-owned tree, so make it here and hand over ownership
            task_out.mkdir(parents=True, exist_ok=True)
            if run_as:
                try:
                    os.chown(task_out, *run_as)
                except OSError:
                    pass
            with open(log_path, "a") as lf:
                lf.write(f"\n===== [{sid}] {task} ({shots}-shot) =====\n")
                if room and not gen_task:
                    lf.write(f"[reasoning model] answers get {room} tokens, not "
                             f"{512 if everyday or safety or sq else 256}: the chat template "
                             f"writes its reasoning before the answer\n")
                if gen_task:
                    lf.write(f"[generative] thinking {'on' if th['on'] else 'off'} "
                             f"({th['mode']}) · {th['budget']} tokens per answer · on {backend}"
                             + (f" ({not_vllm})" if not_vllm else "")
                             + (f" · MMLU-Pro subset of {sub['subset']}"
                                if task == "mmlu_pro" and int(sub.get("subset") or 0) > 0
                                else "") + "\n")
                if dm_plan and backend == "hf":
                    lf.write(_devicemark.hf_plan_lines(dm_plan, gen_batch))
                elif spec["max_length"] and not rec and (not gen_task or backend == "hf"):
                    lf.write(f"[service] lm_eval is told max_length={spec['max_length']}: the "
                             f"model's limit is under text_config in its config, where lm_eval "
                             f"doesn't look (it would take 2,048)\n")
                if remote_code:
                    lf.write(f"[trust_remote_code] running as "
                             f"{config.EVAL_USER or 'root (EVAL_USER unset!)'}, "
                             f"hub offline, token withheld\n")
                lf.flush()
                mark = log_path.stat().st_size      # this task's output starts here
                if rec:
                    # 16b: the relay says the switch, lm_eval can't
                    said = relay is not None and not relay.meter
                    lf.write(f"[served] {rec['name']} at {rec['base_url']} · "
                             f"{rec['pin'].get('version') or rec['pin'].get('file')} · "
                             + (f"thinking {'on' if th['on'] else 'off'}, said by the board's "
                                f"relay (lm_eval sends no switch)" if said else
                                "thinking as the server does (lm_eval sends no switch)")
                             + f" · {_served.concurrency(rec)} at a time"
                             + (" · through the board's relay, which counts each answer's cost"
                                if relay and relay.meter else "") + "\n")
                    lf.flush()
                # 12f.1: a served run's key, and its progress from lm_eval's bar;
                # a local run is called as it always was. 12m.3: and a model from
                # OpenRouter's running total, and its stop at the AI limit
                served_kw = ({"on_poll": lambda: _served_poll(sid, label, log_path, mark, meter)}
                             if rec else {})
                # 12q.G: what the card's other processes hold before this task
                # loads anything: an out-of-memory error is set against it
                others_held = None if rec else others_mib()
                # 15.3: a task that writes its answers here gets a limit sized for
                # it — the answers still to write, at the cap, at the model's pace
                n_task = _items_of(task, sub) if gen_task and not rec else 0
                kept_at_start = cached_answers(task_out) if n_task else 0
                limit = (task_limit(n_task - kept_at_start, th["budget"], model_pace(sub["hf_id"]))
                         if n_task else None)
                if limit:
                    _TASK_LIMIT[sid] = limit
                    lf.write(f"[service] {task}: its limit is {hours_words(limit[0])} — "
                             f"{limit[1]}\n")
                    lf.flush()
                    mark = log_path.stat().st_size
                else:
                    _TASK_LIMIT.pop(sid, None)
                status = _run_task(sid, cmd, lf,
                                   _served.job_env(job_env, rec) if rec else job_env, run_as,
                                   cwd=lm_eval_cwd(task_out), **served_kw)
                if status == CANCELED:
                    canceled = True
                # 12h.1: a model vLLM cannot load runs on the harness's own
                # loader instead, and the result says which models fell back
                if (gen_task and backend == "vllm" and status not in (0, CANCELED)
                        and config.GEN_BACKEND != "vllm"):
                    fell_back = f"vLLM could not run it (exit {status}); ran on hf"
                    backend = "hf"
                    lf.write(f"\n[service] {fell_back}\n")
                    lf.flush()
                    status = _run_task(sid, gen_cmd("hf"), lf, job_env, run_as,
                                       cwd=lm_eval_cwd(task_out))
                    if status == CANCELED:
                        canceled = True
                # 12q.E: a generative task that ran out of GPU memory on hf is tried
                # again at half the batch, down to 1. The batch is sized for scoring
                # short prompts (hfmeta.estimate); 4,096 generated tokens on MMLU-Pro's
                # long ones need far more (Granite-4.0-H-1B, #132 and #144)
                batch = gen_batch or meta.get("batch")
                while (gen_task and not rec and backend == "hf" and isinstance(batch, int)
                       and batch > 1 and status not in (0, CANCELED)
                       and _OOM.search(_read_from(log_path, mark))):
                    oom = oom_said(_read_from(log_path, mark))
                    batch = gen_batch = max(1, batch // 2)
                    lf.write(f"\n[service] {task} ran out of GPU memory: again at batch "
                             f"{batch}" + (f" ({oom_line(oom, others_held)})" if oom else "") + "\n")
                    lf.flush()
                    mark = log_path.stat().st_size      # the next try's output starts here
                    others_held = others_mib()
                    db.update(sid, status="running",
                              progress=f"{label} · out of memory, again at batch {batch}")
                    status = _run_task(sid, gen_cmd("hf", batch), lf, job_env, run_as,
                                       cwd=lm_eval_cwd(task_out))
                    if status == CANCELED:
                        canceled = True
            gpu_seconds += time.time() - t_task
            db.update(sid, gpu_seconds=gpu_seconds)
            if canceled:
                break

            # checked before the exit code, because this failure has a zero exit
            # code: lm_eval did its job perfectly on a model that was partly
            # random. The scores are already on disk by now — lm_eval writes them
            # before we get to look — so removing them is the whole point. Left
            # there they would show on the leaderboard AND be counted as finished
            # work by _task_done() when the submitter resubmits the fix.
            missing = _missing_weights(_read_from(log_path, mark))
            if missing:
                shutil.rmtree(task_out, ignore_errors=True)
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] discarded results for {task}: the "
                             f"checkpoint is missing {', '.join(missing)} and "
                             f"transformers initialized them randomly\n")
                failed_tasks.append(task)
                db.update(sid, load_missing=json.dumps(missing),
                          error=_MISSING_MSG.format(
                              keys=", ".join(missing[:6]),
                              more=f" (+{len(missing) - 6} more)"
                                   if len(missing) > 6 else ""))
                break          # every remaining task would load the same model

            if rec and status == 0:
                _served.adopt_lm_eval_results(task_out, rec)
            if rec and status not in (0, CANCELED):
                got = _served_stopped_in_log(_read_from(log_path, mark))
                if meter and meter.stopped:
                    # 12m.3: stopped at the month's AI limit — where, from lm_eval's bar
                    bar = _tqdm_last(_read_from(log_path, mark)) or (0, 0, 0.0)
                    got = _served.LimitReached(bar[0], bar[1], meter)
                if got:
                    stopped = got
                    failed_tasks.append(task)
                    db.update(sid, error=f"{task}: {got}" + _served.KEPT_FOR_NEXT)
                    break

            # 15.3: the model's pace, from a task it answered in one go here
            if status == 0 and gen_task and not rec and n_task and not kept_at_start:
                try:
                    record_pace(sid, sub["hf_id"], meta.get("revision"), task, task_out,
                                _read_from(log_path, mark))
                except Exception as e:                  # noqa: BLE001 — a limit, not a result
                    with open(log_path, "a") as lf:
                        lf.write(f"\n[service] the pace couldn't be recorded: {e!r}\n")
            if status == 0 and current_fingerprint(task):
                # which questions these answers answer, and which row asked
                (task_out / BANK_FILE).write_text(current_fingerprint(task) + "\n",
                                                  encoding="utf-8")
                (task_out / ANSWERED_BY).write_text(json.dumps(
                    {"submission": sid, "at": time.time()}), encoding="utf-8")

            # 12q.G: a prompt lm_eval cut isn't the protocol's question. The room
            # is sized from the battery's longest, so this is a model whose
            # tokenizer couldn't be counted with, and wrote more tokens than
            # its characters said
            cut = (_CUT.findall(_read_from(log_path, mark))
                   if dm_plan and status == 0 else [])
            if cut:
                was, to = max(int(a) for a, _ in cut), int(cut[0][1])
                shutil.rmtree(task_out, ignore_errors=True)
                failed_tasks.append(task)
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] discarded results for {task}: lm_eval cut "
                             f"{len(cut)} prompt(s) to {to:,} tokens\n")
                db.update(sid, error=f"{task}: a prompt of {was:,} tokens was cut to the "
                                     f"{to:,} this run left for it, so its answers aren't the "
                                     f"protocol's and none were kept. Set DM_PROMPT_TOKENS to "
                                     f"{was:,} or more and resubmit.")
                break                # the other tasks were given the same room

            if status == TIMED_OUT:
                # 15.3: it says it timed out, what its limit was and why, and what it kept
                failed_tasks.append(task)
                kept = cached_answers(task_out)
                db.update(sid, error=f"{task} timed out: its limit was "
                                     f"{hours_words(limit[0]) if limit else 'TASK_TIMEOUT_S'}"
                                     + (f" — {limit[1]}" if limit else "") + ". "
                                     + (f"The {kept:,} answers it wrote are kept: resubmit and it "
                                        f"asks only the other {max(0, n_task - kept):,}."
                                        if kept else "Nothing it wrote was kept."))
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] {task}: {kept:,} of {n_task:,} answers kept\n")
                continue
            if status != 0:
                tail = _tail(log_path)
                failed_tasks.append(task)
                friendly = classify(tail)
                # 12q.G: an out-of-memory error says who held what, and which grew
                oom = (oom_said(_read_from(log_path, mark))
                       if not rec and _OOM.search(tail) else None)
                if oom:
                    at = gen_batch or meta.get("batch") if gen_task else meta.get("batch")
                    line = oom_line(oom, others_held)
                    friendly = ("ran out of GPU memory"
                                + (f" at batch {at}" if isinstance(at, int) and backend == "hf"
                                   else "")
                                + f": {line}. "
                                + ("Resubmit when the card is quieter; the finished tasks are "
                                   "kept and only the missing ones re-run."
                                   if line.endswith("busier") else
                                   "There is no smaller batch to try: the model needs more "
                                   "than the card has free." if at == 1 and backend == "hf" else
                                   "Resubmit; the finished tasks are kept and only the "
                                   "missing ones re-run."))
                db.update(sid, error=f"{task}: {friendly}")
                if _OOM.search(tail):
                    break            # will OOM again for this model — stop here
                if re.search(r"ModuleNotFoundError|ImportError", tail, re.I):
                    break            # environment — fails for every task
                if re.search(r"No space left on device|Errno 28", tail, re.I):
                    break            # operator fault — every task fails the same

        # the judge is an API batch now: SUBMIT it here (seconds, no GPU) and
        # let the poller finish it — a judge that waits on a provider must
        # never hold the card. With JUDGE_MODEL=stub the file is written now.
        judge_note = ""
        if canceled:
            db.update(sid, status="canceled", finished_at=time.time(),
                      progress=f"canceled by request after {len(tasks) and i - 1} of "
                               f"{len(tasks)} tasks — nothing half-written was kept")
            return
        note = reuse_note(reused, tasks) if sub["suite"] == "judged" else ""
        if note:
            db.update(sid, reuse_note=note)
        # 12a: marked straight after the answers, in the same run. Only the
        # TL;DR waits on the judge; the row says so until it lands
        if everyday and not failed_tasks:
            db.update(sid, status="running", progress="marking the answers")
            try:
                ev = _everyday.start(config.OUT_DIR / safe, submission=sid, asked=asked)
                judge_note = _everyday.summary(ev)
                # 12a.4: the wording this run answered, on the run itself
                ver = ev.get("version") or {}
                db.update(sid, bank_version=ver.get("hash") or "")
                with open(log_path, "a") as lf:
                    lf.write(f"\n===== [{sid}] everyday: {_everyday.summary(ev)}"
                             + f" · questions of {ver.get('date') or 'an earlier wording'}"
                             + f" ({ver.get('hash')})"
                             + (f" · judge batch {ev['batch_id']}" if ev.get("batch_id") else "")
                             + (f" · the judge could not be asked: {ev['error']}"
                                if ev.get("error") else "") + " =====\n")
            except Exception as e:                      # noqa: BLE001 — the answers are on disk
                failed_tasks.append("marking")
                db.update(sid, error=f"marking: {e}")
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] everyday tasks could not be marked: {e!r}\n")
        # 12k.2: Do-Not-Answer and XSTest, marked by the judge on the rubric —
        # submitted here (seconds, no GPU); the poller lands the marks
        if safety and not failed_tasks:
            db.update(sid, status="running", progress="sending the answers to the judge")
            try:
                ts = _trust.start(config.OUT_DIR / safe, submission=sid)
                judge_note = _trust.summary(ts)
                with open(log_path, "a") as lf:
                    lf.write(f"\n===== [{sid}] trust & safety: {judge_note}"
                             + (f" · judge batch {ts['batch_id']}" if ts.get("batch_id") else "")
                             + (f" · the judge could not be asked: {ts['error']}"
                                if ts.get("error") else "") + " =====\n")
            except Exception as e:                      # noqa: BLE001 — the answers are on disk
                failed_tasks.append("marking")
                db.update(sid, error=f"marking: {e}")
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] Do-Not-Answer and XSTest could not be marked: {e!r}\n")
        # 12n.2: SimpleQA Verified, graded by the judge with the dataset's grader —
        # submitted here (seconds, no GPU); the poller lands the grades
        if shared and config.SIMPLEQA_TASK in tasks and not failed_tasks:
            db.update(sid, status="running", progress="sending the answers to the judge")
            try:
                sr = _sq.start(config.OUT_DIR / row_safe, submission=sid)
                judge_note = _sq.summary(sr)
                with open(log_path, "a") as lf:
                    lf.write(f"\n===== [{sid}] SimpleQA Verified: {judge_note}"
                             + (f" · judge batch {sr['batch_id']}" if sr.get("batch_id") else "")
                             + (f" · the judge could not be asked: {sr['error']}"
                                if sr.get("error") else "") + " =====\n")
            except Exception as e:                      # noqa: BLE001 — the answers are on disk
                failed_tasks.append("grading")
                db.update(sid, error=f"grading: {e}")
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] SimpleQA Verified could not be graded: {e!r}\n")
        # 12o.3: MobileAIBench's sets, scored by its own metrics — no judge, no
        # GPU. 14.1: the judged part's answers go to the judge as a step of its
        # own: an offline judge leaves them "awaiting judge", and the run is done
        if mobile and not failed_tasks and (sub.get("part") or "") in ("mmlu", "mmlu_full"):
            # 14.3: every pick kept, and scored against the key as it stands.
            # 14.4: one run, two scores — a full run's picks give Pro's score
            # too, each set scored on its own wording
            try:
                got = _mmp.collect(config.OUT_DIR / row_safe)
                judge_note = _mmp.summary(_mmp.score(got))
                if sub.get("part") == "mmlu_full":
                    full = _mmp.full_predictions(config.OUT_DIR / row_safe)
                    judge_note = (_mmp.summary(_mmp.score(full, _mmp.full_key()), _mmp.FULL_NAME)
                                  + " · " + judge_note)
                with open(log_path, "a") as lf:
                    lf.write(f"\n===== [{sid}] {judge_note} =====\n")
            except Exception as e:                      # noqa: BLE001 — the answers are on disk
                failed_tasks.append("reading")
                db.update(sid, error=f"reading the answers: {e}")
        elif mobile and not failed_tasks:
            db.update(sid, status="running", progress="scoring the answers")
            try:
                if (sub.get("part") or "") in ("judged", "trust"):
                    out = _mab.start_judge(config.OUT_DIR / row_safe, sid)
                else:
                    out = _mab.mark(config.OUT_DIR / row_safe)
                    if out:
                        _mab.write(config.OUT_DIR / row_safe, out)
                judge_note = _mab.summary(out) + (f" · {out['note']}" if (out or {}).get("note")
                                                  else "")
                with open(log_path, "a") as lf:
                    lf.write(f"\n===== [{sid}] MobileAIBench: {judge_note} =====\n")
            except Exception as e:                      # noqa: BLE001 — the answers are on disk
                failed_tasks.append("scoring")
                db.update(sid, error=f"scoring: {e}")
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] MobileAIBench could not be scored: {e!r}\n")
        # 12h.1: the three read again, in this board's words — the letter an
        # answer settles on, the final answer compared as maths, the harness's
        # own IFEval verdicts — and the answers that ran out of room counted
        if generative and not failed_tasks:
            db.update(sid, status="running", progress="reading the answers")
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
                import generative as _gen
                n_sub = int(sub.get("subset") or 0)
                out = _gen.mark(config.OUT_DIR / row_safe, extra={
                    "thinking": {"mode": th["mode"], "on": th["on"], "budget": th["budget"]},
                    "backend": backend, "fell_back": fell_back or None,
                    "not_vllm": not_vllm or None,
                    "subset": ({"n": min(n_sub, sum(config.MMLU_PRO_SUBJECTS.values())),
                                "of": sum(config.MMLU_PRO_SUBJECTS.values()),
                                "seed": config.GEN_SUBSET_SEED} if n_sub > 0 else None)})
                if out:
                    _gen.write(config.OUT_DIR / row_safe, out)
                    judge_note = " · " + _gen.summary(out)
                    with open(log_path, "a") as lf:
                        lf.write(f"\n===== [{sid}] generative: {_gen.summary(out)} =====\n")
            except Exception as e:                      # noqa: BLE001 — the answers are on disk
                failed_tasks.append("reading")
                db.update(sid, error=f"reading the answers: {e}")
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] the answers could not be read: {e!r}\n")
        # 12q: DeviceMark's battery scored by its protocol — no judge, no GPU
        if devicemark and not failed_tasks and not config.DM_SCORE_AFTER_RUN:
            # 15.1: on a rented GPU — scored on the server, from the bundle
            judge_note = "answered: scored where the bundle is imported"
        elif devicemark and not failed_tasks:
            db.update(sid, status="running", progress="devicemark · scoring the answers")
            try:
                judge_note = _devicemark.mark_hf(sid, sub, meta, config.OUT_DIR / row_safe, th)
                with open(log_path, "a") as lf:
                    lf.write(f"\n===== [{sid}] devicemark: {judge_note} =====\n")
            except Exception as e:                      # noqa: BLE001 — the answers are on disk
                failed_tasks.append("scoring")
                db.update(sid, error=f"scoring: {e}")
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] the devicemark answers could not be scored: {e!r}\n")
        if sub["suite"] == "judged" and not failed_tasks:
            db.update(sid, status="running", progress="submitting the answers to the judge")
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
            import judge as _judge
            try:
                # the row is passed in so the batch id lands on it at submit
                # time: the queue shows THIS run's batch, never the model's
                # newest, and the poller can say on the row when it lands
                jr = _judge.start_run(config.OUT_DIR / safe, config.OUT_DIR, only=only,
                                      submission=sid)
                if jr.get("skipped"):
                    judge_note = f" · {jr['skipped']}"
                elif jr.get("batch_id"):
                    judge_note = (f" · judge batch {jr['batch_id']} submitted ({jr['n']} answers); "
                                  f"judge.json lands when it completes")
                elif jr.get("written"):
                    judge_note = " · judged"
                with open(log_path, "a") as lf:
                    lf.write(f"\n===== [{sid}] judge: {json.dumps(jr)} =====\n")
            except Exception as e:                      # noqa: BLE001 — the answers are on disk
                failed_tasks.append("judge")
                db.update(sid, error=f"judge: {e}")
                with open(log_path, "a") as lf:
                    lf.write(f"\n[service] judge could not be submitted: {e!r}\n")

        # 12m.3: what the run cost on OpenRouter, on its row
        def spent(line: str) -> str:
            # 12s: and the questions its server failed on, after the run's own line
            line = " · ".join(x for x in (line, _served.odd_line(odd).strip(" ·")) if x)
            return " · ".join(x for x in (line, f"{_served.usd(meter.spent)} on OpenRouter")
                              if x) if meter else line
        if failed_tasks:
            db.update(sid, status="failed", finished_at=time.time(),
                      progress=spent(f"failed on: {', '.join(failed_tasks)}"))
        elif stopped:
            # 12f.1: a partial result, and it says so — what was answered is
            # marked (and judged) like any answer; the rest is asked next time
            db.update(sid, status="failed", finished_at=time.time(),
                      error=f"{stopped.task}: {stopped} · the {stopped.done} answered are kept "
                            f"and marked",
                      progress=spent((judge_note or "").strip(" ·")))
        else:
            what = (f"all {len(tasks)} tasks" if not only
                    else f"{', '.join(t.replace('exam_', '') for t in tasks)}")
            db.update(sid, status="done", finished_at=time.time(),
                      progress=spent(judge_note if everyday or safety or shared or mobile
                                     or devicemark
                                     else f"{what} done" + (f" · {note}" if note else "")
                                     + judge_note), error="")
    finally:
        _TASK_LIMIT.pop(sid, None)
        if relay is not None:
            relay.__exit__(None, None, None)
        if held:                   # 12m.3: never a GPU run's lock, taken while a remote one ran
            release_lock()
        if remote_code:
            shutil.rmtree(config.BENCH_ROOT / ".jobscratch" / str(sid),
                          ignore_errors=True)
