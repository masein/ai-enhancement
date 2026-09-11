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
import subprocess
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


def _child_env(remote_code: bool) -> dict:
    env = os.environ.copy()
    if not remote_code:
        return env
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                "HF_API_TOKEN", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY"):
        env.pop(var, None)
    env["HF_HUB_OFFLINE"] = "1"        # the artifact is on disk; nothing to fetch
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return env


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
# the run
# ---------------------------------------------------------------------------

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

        include_args = []
        if config.EVAL_TASKS_DIR.is_dir() and any(config.EVAL_TASKS_DIR.glob("*.yaml")):
            include_args = ["--include_path", str(config.EVAL_TASKS_DIR)]

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
                   *include_args]
            if kind == "instruct":
                cmd.append("--apply_chat_template")

            t_task = time.time()
            # the dropped-privilege child cannot create its own output dir under
            # a root-owned tree, so make it here and hand over ownership
            run_as = _eval_ids() if remote_code else None
            if remote_code and not run_as and os.getuid() == 0:
                # fail closed: the whole point of the gate is that uploaded code
                # does NOT run as root, so an unresolvable EVAL_USER stops the job
                # rather than quietly becoming the thing we were guarding against
                db.update(sid, status="failed", finished_at=time.time(),
                          error=f"refusing to run custom model code as root: "
                                f"EVAL_USER={config.EVAL_USER!r} does not resolve to "
                                f"an account on this machine.")
                release_lock()
                return
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
                try:
                    proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                          cwd=config.BENCH_ROOT,
                                          env=_child_env(remote_code),
                                          **({"user": run_as[0], "group": run_as[1]}
                                             if run_as else {}),
                                          timeout=config.TASK_TIMEOUT_S)
                    status = proc.returncode
                except subprocess.TimeoutExpired:
                    status = -1
                    lf.write(f"\n[service] killed after {config.TASK_TIMEOUT_S}s timeout\n")
            gpu_seconds += time.time() - t_task
            db.update(sid, gpu_seconds=gpu_seconds)

            if status != 0:
                tail = _tail(log_path)
                failed_tasks.append(task)
                friendly = classify(tail)
                db.update(sid, error=f"{task}: {friendly}")
                if re.search(r"out of memory|OutOfMemoryError", tail, re.I):
                    break            # will OOM again for this model — stop here
                if re.search(r"ModuleNotFoundError|ImportError", tail, re.I):
                    break            # environment — fails for every task

        if failed_tasks:
            db.update(sid, status="failed", finished_at=time.time(),
                      progress=f"failed on: {', '.join(failed_tasks)}")
        else:
            db.update(sid, status="done", finished_at=time.time(),
                      progress=f"all {len(tasks)} tasks done", error="")
    finally:
        release_lock()
