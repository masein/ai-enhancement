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

from . import config, db
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
                   "EXAM_API_KEY", "JUDGE_API_KEY")


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
        if owner and Path(f"/proc/{owner}").is_dir():
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


def release_lock() -> None:
    subprocess.run(["rm", "-rf", str(LOCK)], check=False)


# ---------------------------------------------------------------------------
# error classification — the message a friend sees instead of a traceback
# ---------------------------------------------------------------------------

_FRIENDLY = [
    (r"out of memory|OutOfMemoryError",
     "ran out of GPU memory — the card was busier than when the run started. "
     "Resubmit; the finished tasks are kept and only the missing ones re-run."),
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


def classify(log_tail: str) -> str:
    for pat, msg in _FRIENDLY:
        if re.search(pat, log_tail, re.I):
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

def include_args_for(task: str) -> list[str]:
    """--include_path for a task the harness does not ship. The perplexity
    slices live under BENCH_ROOT/eval_tasks; the permutation control ships with
    this repo (eval_tasks/mmlu_perm). One task per lm_eval invocation means
    each task gets exactly the directory it lives in — no task ever sees the
    other's yaml, so a stray file in one cannot rename a task in the other."""
    if task in config.CONTROL_TASKS:
        return ["--include_path", str(config.CONTROL_TASKS_DIR)]
    if task.startswith(("exam_", "fr_")):
        return ["--include_path", str(config.JUDGED_TASKS_DIR)]
    if config.EVAL_TASKS_DIR.is_dir() and any(config.EVAL_TASKS_DIR.glob("*.yaml")):
        return ["--include_path", str(config.EVAL_TASKS_DIR)]
    return []


def _task_done(task_out: Path) -> bool:
    return any(task_out.glob("*/results*.json")) or any(task_out.glob("results*.json"))


def run_submission(sub: dict) -> None:
    sid = sub["id"]

    # -- preflight: metadata only, no GPU, seconds --------------------------------
    try:
        # the submitter's kind is passed in: 'auto' is resolved here, and refused
        # when it is genuinely ambiguous rather than guessed
        meta = preflight(sub["hf_id"], sub["kind"],
                         allow_remote_code=bool(sub.get("allow_remote_code")))
    except PreflightError as e:
        db.update(sid, status="failed", error=str(e), finished_at=time.time())
        return
    kind = meta["kind"]
    remote_code = bool(meta.get("remote_code"))
    db.update(sid, kind=kind, params=meta["params"], vocab=meta["vocab"],
              batch=meta["batch"], need_gb=meta["need_gb"],
              arch=json.dumps(meta.get("archinfo") or {}),
              progress=f"preflight ok · batch={meta['batch']} · "
                       f"needs ~{meta['need_gb']:g} GB")

    tasks = config.tasks_for_suite(sub["suite"])
    # a judged run narrowed to one topic: the same suite, fewer tasks. The
    # judge below grades only these, so a person can sit one topic in minutes
    # instead of the whole exam
    only = [t for t in json.loads(sub.get("tasks") or "[]") if t in tasks]
    if only:
        tasks = only
    safe = sub["hf_id"].replace("/", "__")
    log_path = config.LOGS_DIR / f"service_{sid}_{safe}.log"
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # drop the model's shape next to its results, so the report (live OR the
    # frozen single-file kind) can show architecture/hidden/layers/vocab without
    # any access to the service database
    meta_dir = config.OUT_DIR / safe
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "model_meta.json").write_text(json.dumps(
        {"model": sub["hf_id"], "kind": kind, "params": meta["params"],
         "kind_reason": meta.get("kind_reason"),
         **(meta.get("archinfo") or {})}), encoding="utf-8")
    if remote_code:      # the code that produced the scores is part of the record
        db.update(sid, progress=f"preflight ok · custom model code · batch={meta['batch']}")

    # -- one run at a time: wait for the shared lock ------------------------------
    t0 = time.time()
    while not acquire_lock(sid):
        db.update(sid, status="waiting_lock",
                  progress="another run (service or CLI) holds the GPU lock")
        if time.time() - t0 > config.GPU_WAIT_MAX_S:
            db.update(sid, status="failed", finished_at=time.time(),
                      error="gave up waiting for the run lock — a manual run has "
                            "held the GPU for hours; resubmit later.")
            return
        time.sleep(config.GPU_POLL_S)

    try:
        # -- wait for VRAM, then run the missing tasks ----------------------------
        need_mib = int(meta["need_gb"] * 1024) + config.FREE_MARGIN_MIB
        t0 = time.time()
        while (free := gpu_free_mib()) < need_mib:
            db.update(sid, status="waiting_gpu",
                      progress=f"waiting for VRAM: need {need_mib} MiB, "
                               f"{free} MiB free")
            if time.time() - t0 > config.GPU_WAIT_MAX_S:
                db.update(sid, status="failed", finished_at=time.time(),
                          error=f"gave up after {config.GPU_WAIT_MAX_S // 3600}h "
                                f"waiting for {need_mib} MiB of free VRAM.")
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
        for i, task in enumerate(tasks, 1):
            shots = config.NFEWSHOT.get(task, 0)
            task_out = config.OUT_DIR / safe / f"{task}_{shots}shot"
            label = f"{i}/{len(tasks)} · {task} ({shots}-shot)"
            if _task_done(task_out):
                db.update(sid, status="running", progress=f"{label} — already done")
                continue
            db.update(sid, status="running", progress=label)

            # local/<name> artifacts resolve to their on-disk directory; the report
            # normalizes the path back to local/<name> so ids stay consistent
            pretrained = sub["hf_id"]
            if pretrained.startswith("local/"):
                pretrained = str((config.ARTIFACTS_DIR / pretrained[6:]).resolve())
            margs = f"pretrained={pretrained},dtype=bfloat16"
            if remote_code:
                margs += ",trust_remote_code=True"
            cmd = ["lm_eval",
                   "--model", "hf",
                   "--model_args", margs,
                   "--tasks", task,
                   "--num_fewshot", str(shots),
                   "--batch_size", str(meta["batch"]),
                   "--seed", str(config.SEED),
                   "--output_path", str(task_out),
                   "--log_samples",
                   "--device", "cuda:0",
                   *include_args_for(task)]
            if kind == "instruct":
                cmd.append("--apply_chat_template")

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
                if remote_code:
                    lf.write(f"[trust_remote_code] running as "
                             f"{config.EVAL_USER or 'root (EVAL_USER unset!)'}, "
                             f"hub offline, token withheld\n")
                lf.flush()
                mark = log_path.stat().st_size      # this task's output starts here
                try:
                    proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                          cwd=config.BENCH_ROOT,
                                          env=job_env,
                                          **({"user": run_as[0], "group": run_as[1]}
                                             if run_as else {}),
                                          timeout=config.TASK_TIMEOUT_S)
                    status = proc.returncode
                except subprocess.TimeoutExpired:
                    status = -1
                    lf.write(f"\n[service] killed after {config.TASK_TIMEOUT_S}s timeout\n")
            gpu_seconds += time.time() - t_task
            db.update(sid, gpu_seconds=gpu_seconds)

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

            if status != 0:
                tail = _tail(log_path)
                failed_tasks.append(task)
                friendly = classify(tail)
                db.update(sid, error=f"{task}: {friendly}")
                if re.search(r"out of memory|OutOfMemoryError", tail, re.I):
                    break            # will OOM again for this model — stop here
                if re.search(r"ModuleNotFoundError|ImportError", tail, re.I):
                    break            # environment — fails for every task
                if re.search(r"No space left on device|Errno 28", tail, re.I):
                    break            # operator fault — every task fails the same

        # the judge is an API batch now: SUBMIT it here (seconds, no GPU) and
        # let the poller finish it — a judge that waits on a provider must
        # never hold the card. With JUDGE_MODEL=stub the file is written now.
        judge_note = ""
        if sub["suite"] == "judged" and not failed_tasks:
            db.update(sid, status="running", progress="submitting the answers to the judge")
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
            import judge as _judge
            try:
                jr = _judge.start_run(config.OUT_DIR / safe, config.OUT_DIR, only=only)
                if jr.get("skipped"):
                    judge_note = f" · {jr['skipped']}"
                elif jr.get("batch_id"):
                    # on the row, so the queue shows THIS run's batch and the
                    # poller can say on the row when it lands
                    db.update(sid, judge_batch=jr["batch_id"])
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

        if failed_tasks:
            db.update(sid, status="failed", finished_at=time.time(),
                      progress=f"failed on: {', '.join(failed_tasks)}")
        else:
            what = (f"all {len(tasks)} tasks" if not only
                    else f"{', '.join(t.replace('exam_', '') for t in tasks)}")
            db.update(sid, status="done", finished_at=time.time(),
                      progress=f"{what} done{judge_note}", error="")
    finally:
        release_lock()
        if remote_code:
            shutil.rmtree(config.BENCH_ROOT / ".jobscratch" / str(sid),
                          ignore_errors=True)
