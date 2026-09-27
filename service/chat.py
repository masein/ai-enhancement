"""12d.1: the Playground's chat engine — the models on this server, streaming
their replies to the page, inside the service.

- **The loader is the runs' own.** A model loads through `runner.load_spec`
  (its path, bfloat16, the approved commit) and lm_eval's HF loader, the one a
  run's `--model hf` uses; its reply goes through its own chat template.
- **Runs come first.** While a run holds the GPU lock nothing loads on the
  GPU; a model under CHAT_CPU_MAX_PARAMS_B answers on the CPU meanwhile. A run
  about to take the lock calls `yield_gpu()`: replies on the GPU stop (kept,
  "cut short"), GPU models unload, and the run waits CHAT_YIELD_WAIT_S at most.
  Chat never takes the lock, and a lock taken from the command line is seen
  between tokens.
- **One reply per loaded model at a time**; a second waits and says so.
- **Stop**, and a closed tab, end a reply at once. A model idle for
  CHAT_IDLE_UNLOAD_S unloads.
- **A model that runs its own code is not offered:** runs execute such code
  only in a sandboxed subprocess (runner._child_env, EVAL_USER), and chat runs
  in the service.

The engine never touches a score: what it writes is the chat's own record.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config, db, runner

CUT_RUN = "a run started"
WAIT_WORDS = "answering another message, yours is next"
WAIT_GPU = "waiting for the GPU"
CPU_NOTE = "on the CPU while a run uses the GPU — slower"
SUITE_WORDS = {"full": "Standard tests", "quick": "quick tests", "control": "control tests",
               "judged": "Knowledge exam", "everyday": "Everyday tasks",
               "generative": "instruction and maths tests"}


def now() -> float:
    """the engine's clock — a test replaces it"""
    return time.time()


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class FakeBackend:
    """A model that streams canned text, word by word: the tests' model, and
    CHAT_BACKEND=fake. `responder(model, messages, settings)` gives the text;
    a model that thinks writes its thinking first, in <think> tags."""
    name = "fake"
    delay_s = 0.01
    responder = None
    loaded: list = []

    def load(self, model_id: str, spec: dict, device: str):
        type(self).loaded.append((model_id, device))
        return {"model": model_id, "device": device}

    def unload(self, handle) -> None:
        pass

    def generate(self, handle, messages: list[dict], settings: dict, should_stop, emit) -> None:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = (type(self).responder(handle["model"], messages, settings)
                if type(self).responder else
                f"You asked: {last[:60]}. This is a short reply from "
                f"{handle['model'].split('/')[-1]}, streamed a word at a time.")
        if settings.get("thinking_on"):
            text = "<think>\nLet me think this through, one step at a time.\n</think>\n\n" + text
        for piece in re.findall(r"\S+\s*|\s+", text):
            if should_stop():
                return
            emit(piece)
            if type(self).delay_s:
                time.sleep(type(self).delay_s)


class HFBackend:
    """transformers through lm_eval's HFLM — the loader `--model hf` uses — and
    a TextIteratorStreamer"""
    name = "hf"

    def load(self, model_id: str, spec: dict, device: str):
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=spec["pretrained"], revision=spec["revision"] or "main",
                  dtype=spec["dtype"], trust_remote_code=False, device=device, batch_size=1)
        return {"model": lm.model, "tok": lm.tokenizer, "device": device}

    def unload(self, handle) -> None:
        handle.clear()
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass

    def generate(self, handle, messages: list[dict], settings: dict, should_stop, emit) -> None:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        tok, model = handle["tok"], handle["model"]
        # no enable_thinking unless it was changed: the run passes none, so the
        # template's own default is what was scored
        kw = {} if settings.get("thinking") is None else {"enable_thinking": settings["thinking"]}
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, **kw)
        ids, mask = enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        class Stop(StoppingCriteria):
            def __call__(self, *a, **k):
                return bool(should_stop())
        temp = float(settings.get("temperature") or 0.0)
        gen = dict(input_ids=ids, attention_mask=mask, max_new_tokens=int(settings["max_gen_toks"]),
                   streamer=streamer,
                   stopping_criteria=StoppingCriteriaList([Stop()]), do_sample=temp > 0)
        if temp > 0:
            gen["temperature"] = temp
        t = threading.Thread(target=lambda: torch.inference_mode()(model.generate)(**gen),
                             daemon=True)
        t.start()
        for piece in streamer:
            emit(piece)
        t.join()


def backend():
    return FakeBackend() if config.CHAT_BACKEND == "fake" else HFBackend()


def gpu_free_bytes() -> int | None:
    """free GPU memory, or None: no GPU this process can use"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return int(free)
    except Exception:                                       # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# the models on this server
# ---------------------------------------------------------------------------

def _meta(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def board_models() -> list[dict]:
    """every model on the board: {id, name, kind, params, archinfo, chat,
    why_not}. `chat` is whether the Playground offers it"""
    from . import catalog
    approved = set(catalog.approved())
    out = []
    root = config.OUT_DIR
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        m = _meta(d / "model_meta.json")
        if not m or m.get("base_model") or not m.get("model"):
            continue                     # a "· thinking" row is its model's, not another
        if m.get("served"):
            continue                     # 12f.1: served elsewhere, never loaded here
        mid = m["model"]
        arch = {k: v for k, v in m.items() if k not in ("model", "kind", "params", "kind_reason")}
        templ = bool(arch.get("tmpl_sha")) or m.get("kind") == "instruct"
        why = ("" if templ else "base") if mid not in approved else "own code"
        out.append({"id": mid, "name": mid.split("/")[-1], "kind": m.get("kind") or "",
                    "params": m.get("params"), "archinfo": arch,
                    "source": "artifact" if mid.startswith("local/") else "hub",
                    "chat": not why, "why_not": why})
    return out


def model_row(model_id: str) -> dict | None:
    return next((m for m in board_models() if m["id"] == model_id), None)


def scored_settings(row: dict) -> dict:
    """what the Everyday run generates with — everyday.run_settings, the one
    function both read — in the Playground's four fields"""
    import everyday as ev
    s = ev.run_settings(row.get("archinfo"))
    return {"system": s["system"], "thinking": s["thinking"], "temperature": s["temperature"],
            "max_gen_toks": s["max_gen_toks"], "thinking_on": s["thinking_on"],
            "can_think": s["can_think"]}


FIELDS = ("system", "thinking", "temperature", "max_gen_toks")


def is_scored(settings: dict, row: dict) -> bool:
    base = scored_settings(row)
    return all(settings.get(k, base[k]) == base[k] for k in FIELDS)


def effective(settings: dict, row: dict) -> dict:
    """the settings a reply is generated with: the scored ones, with the
    chat's own changes over them"""
    base = scored_settings(row)
    s = {**base, **{k: settings[k] for k in FIELDS if k in settings}}
    s["thinking_on"] = base["thinking_on"] if s["thinking"] is None else bool(s["thinking"])
    return s


def context_words(row: dict, settings: dict) -> int:
    """about how many words of conversation fit beside the reply budget — the
    reply keeps at most half the context, so a budget as long as the context
    (a reasoning model's 4,096 on a 4k model) leaves room to ask"""
    ctx = int((row.get("archinfo") or {}).get("ctx") or 4096)
    return max(0, int((ctx - min(int(settings["max_gen_toks"]), ctx // 2)) * 0.75))


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

@dataclass
class Loaded:
    model: str
    device: str
    handle: object
    last_used: float
    busy: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Stream:
    id: str
    chat_id: str
    model: str
    events: list = field(default_factory=list)
    done: bool = False
    stop: threading.Event = field(default_factory=threading.Event)
    cut: str = ""
    cond: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, ev: dict) -> None:
        with self.cond:
            self.events.append(ev)
            if ev.get("t") in ("done", "refused", "error"):
                self.done = True
            self.cond.notify_all()


class Engine:
    def __init__(self):
        self.loaded: dict[str, Loaded] = {}
        self.streams: dict[str, Stream] = {}
        self.lock = threading.RLock()
        self.yielding = False
        self.be = None

    def _backend(self):
        if self.be is None or getattr(self.be, "name", "") != (
                "fake" if config.CHAT_BACKEND == "fake" else "hf"):
            self.be = backend()
        return self.be

    # -- where a model can answer ---------------------------------------------
    def place(self, row: dict) -> tuple[str | None, str]:
        """("cuda" | "cpu", a note) — or (None, why not now), in one line"""
        params = float(row.get("params") or 0)
        small = 0 < params < config.CHAT_CPU_MAX_PARAMS_B * 1e9
        run = runner.run_holding()
        if run or self.yielding:
            if small:
                return "cpu", CPU_NOTE
            return None, gpu_busy_line(run)
        free = gpu_free_bytes()
        need = params * 2 + config.CHAT_GPU_MARGIN_GB * 1e9     # bfloat16: two bytes a weight
        held = self.loaded.get(row["id"])
        if held and held.device == "cuda":
            return "cuda", ""
        if free is None or free < need:
            if small:
                return "cpu", ("on the CPU: no GPU is free for chat — slower" if free is None
                               else "on the CPU: the GPU's memory is taken — slower")
            if free is None:
                return None, "No GPU is free for chat on this server."
            return None, (f"{row['name']} needs about {need / 1e9:.0f} GB of GPU memory and "
                          f"{free / 1e9:.1f} GB is free now.")
        return "cuda", ""

    def _load(self, row: dict, device: str) -> Loaded:
        with self.lock:
            held = self.loaded.get(row["id"])
            if held and held.device == device:
                held.last_used = now()
                return held
            if held:
                self._unload(held)
        from . import hfmeta
        meta = hfmeta.preflight(row["id"], row.get("kind") or "instruct")
        if meta.get("remote_code"):
            raise RuntimeError(f"{row['name']} runs its own code, which chat does not run.")
        handle = self._backend().load(row["id"], runner.load_spec(row["id"], meta), device)
        ld = Loaded(row["id"], device, handle, now())
        with self.lock:
            self.loaded[row["id"]] = ld
        return ld

    def _unload(self, ld: Loaded) -> None:
        with self.lock:
            self.loaded.pop(ld.model, None)
        try:
            self._backend().unload(ld.handle)
        except Exception:                                   # noqa: BLE001
            pass

    # -- runs come first -----------------------------------------------------
    def yield_gpu(self, wait_s: float | None = None, sleep=time.sleep) -> float:
        """A run is about to take the lock: stop the replies on the GPU (kept,
        cut short), unload the GPU models; at most CHAT_YIELD_WAIT_S. Returns
        how long it took. Never raises: a run is never broken by chat"""
        t0 = now()
        limit = config.CHAT_YIELD_WAIT_S if wait_s is None else wait_s
        self.yielding = True
        try:
            for st in list(self.streams.values()):
                ld = self.loaded.get(st.model)
                if not st.done and ld and ld.device == "cuda":
                    st.cut = CUT_RUN
                    st.stop.set()
            while now() - t0 < limit:
                busy = [ld for ld in self.loaded.values()
                        if ld.device == "cuda" and ld.busy.locked()]
                if not busy:
                    break
                sleep(0.05)
            for ld in [x for x in self.loaded.values() if x.device == "cuda"]:
                if not ld.busy.locked():
                    self._unload(ld)
        except Exception:                                   # noqa: BLE001
            pass
        finally:
            self.yielding = False
        return now() - t0

    def sweep(self) -> list[str]:
        """unload what has had no message for CHAT_IDLE_UNLOAD_S"""
        gone = []
        for ld in list(self.loaded.values()):
            if not ld.busy.locked() and now() - ld.last_used >= config.CHAT_IDLE_UNLOAD_S:
                self._unload(ld)
                gone.append(ld.model)
        return gone

    def status(self) -> dict:
        return {"loaded": [{"model": ld.model, "name": ld.model.split("/")[-1],
                            "device": ld.device, "idle_s": round(now() - ld.last_used)}
                           for ld in self.loaded.values()],
                "run": gpu_busy_line(runner.run_holding()) if runner.run_holding() else ""}

    # -- a reply -------------------------------------------------------------
    def start(self, chat: dict, messages: list[dict], settings: dict, row: dict,
              on_done, stream_id: str | None = None, after: Stream | None = None) -> Stream:
        """a reply, streamed; `after` (12d.2): a reply that must finish first —
        two models that don't both fit answer one after the other"""
        st = Stream(stream_id or uuid.uuid4().hex, chat["id"], row["id"])
        self.streams[st.id] = st
        threading.Thread(target=self._run, args=(st, messages, settings, row, on_done, after),
                         daemon=True, name=f"chat-{st.id[:6]}").start()
        return st

    def fits_both(self, a: dict, b: dict) -> bool:
        """12d.2: can two models answer at once? Both on the GPU, with room for
        both (what is loaded already needs nothing more), or both on the CPU"""
        pa, pb = self.place(a)[0], self.place(b)[0]
        if not pa or not pb:
            return False
        if pa == "cpu" or pb == "cpu":
            return pa == pb == "cpu"
        need = sum(float(r.get("params") or 0) * 2 + config.CHAT_GPU_MARGIN_GB * 1e9
                   for r in (a, b) if not (self.loaded.get(r["id"])
                                           and self.loaded[r["id"]].device == "cuda"))
        free = gpu_free_bytes()
        return free is not None and free >= need

    def stop(self, stream_id: str) -> bool:
        st = self.streams.get(stream_id)
        if not st:
            return False
        st.stop.set()
        return True

    def _run(self, st: Stream, messages, settings, row, on_done, after=None) -> None:
        reply = {"text": "", "thinking": "", "cut": "", "device": "", "secs": 0.0}
        try:
            if after is not None and not after.done:
                st.emit({"t": "wait", "why": WAIT_GPU})
                while not after.done:
                    if st.stop.is_set():
                        on_done(None, "stopped before it began")
                        st.emit({"t": "done", "reply": None, "stopped": True})
                        return
                    time.sleep(0.05)
                # the first one's model gives its memory back for this one
                first = self.loaded.get(after.model)
                if first and first.device == "cuda" and not first.busy.locked() \
                        and first.model != row["id"]:
                    self._unload(first)
            device, why = self.place(row)
            if not device:
                on_done(None, why)                  # the message waits: the page asks again
                st.emit({"t": "refused", "why": why})
                return
            st.emit({"t": "place", "device": device, "note": why})
            ld = self._load(row, device)
            if not ld.busy.acquire(blocking=False):
                st.emit({"t": "wait", "why": WAIT_WORDS})
                while not ld.busy.acquire(timeout=0.2):
                    if st.stop.is_set():
                        on_done(None, "stopped before it began")
                        st.emit({"t": "done", "reply": None, "stopped": True})
                        return
            try:
                ld.last_used = now()
                acc = []
                t0 = time.time()
                sent = {"think": 0, "text": 0}
                thinks = bool(settings.get("thinking_on"))

                def should_stop():
                    if st.stop.is_set():
                        return True
                    if ld.device == "cuda" and runner.LOCK.exists():
                        st.cut = st.cut or CUT_RUN
                        st.stop.set()
                        return True
                    return False

                def emit(piece: str) -> None:
                    acc.append(piece)
                    think, text = split("".join(acc), thinks)
                    if len(think) < sent["think"] or len(text) < sent["text"]:
                        st.emit({"t": "reset", "think": think, "text": text})
                    else:
                        if len(think) > sent["think"]:
                            st.emit({"t": "think", "d": think[sent["think"]:]})
                        if len(text) > sent["text"]:
                            st.emit({"t": "text", "d": text[sent["text"]:]})
                    sent.update(think=len(think), text=len(text))

                self._backend().generate(ld.handle, messages, settings, should_stop, emit)
                raw = "".join(acc)
                think, text = final_split(raw)
                secs = max(0.001, time.time() - t0)
                words = len(text.split())
                reply.update(text=text, thinking=think, device=device,
                             secs=round(max(secs, 0.01), 2),
                             words=words, wps=round(words / secs, 1),
                             cut=st.cut or ("stopped" if st.stop.is_set() else ""))
            finally:
                ld.last_used = now()
                ld.busy.release()
            st.emit({"t": "done", "reply": on_done(reply)})
        except Exception as e:                              # noqa: BLE001 — said on the page
            try:
                on_done(None, str(e)[:300])
            except Exception:                               # noqa: BLE001
                pass
            st.emit({"t": "error", "why": str(e)[:300]})
        finally:
            threading.Timer(600, lambda: self.streams.pop(st.id, None)).start()


def gpu_busy_line(run: dict | None) -> str:
    """"The GPU is running Qwen3.5-2B's Standard tests. Chat starts when it's
    done." — the run gives no time left today, so it says "when it's done" """
    if not run:
        return ""
    name = (run.get("hf_id") or "").split("/")[-1]
    what = SUITE_WORDS.get(run.get("suite") or "", "tests")
    if not name:
        return "The GPU is running a test. Chat starts when it's done."
    return f"The GPU is running {name}'s {what}. Chat starts when it's done."


def split(raw: str, thinks: bool) -> tuple[str, str]:
    """(thinking, reply) of a reply still streaming. A model that thinks
    writes its thinking first, often with the opening tag in the prompt: until
    its closing tag arrives, all of it is thinking"""
    import judge
    p = judge.split_reasoning(raw)
    if p["had_reasoning"]:
        return p["reasoning_text"], p["answer_text"]
    if thinks:
        return raw, ""
    return "", raw


def final_split(raw: str) -> tuple[str, str]:
    """the finished reply, split as a run's answer is (judge.split_reasoning):
    what is shown and copied is the answer"""
    import judge
    p = judge.split_reasoning(raw)
    return (p["reasoning_text"] if p["had_reasoning"] else ""), p["answer_text"]


ENGINE = Engine()
_janitor: threading.Thread | None = None


def start_janitor() -> None:
    """unload idle models every half minute"""
    global _janitor
    if _janitor and _janitor.is_alive():
        return

    def loop():
        while True:
            time.sleep(30)
            try:
                ENGINE.sweep()
            except Exception:                               # noqa: BLE001
                pass
    _janitor = threading.Thread(target=loop, daemon=True, name="chat-janitor")
    _janitor.start()


# ---------------------------------------------------------------------------
# chats: kept per person, by the name in masein ▾
# ---------------------------------------------------------------------------

def _who(name: str) -> str:
    return (name or "").strip().casefold()


def title_of(text: str) -> str:
    w = (text or "").split()
    return " ".join(w[:6]) + ("…" if len(w) > 6 else "")


def new_chat(model: str, by: str, settings: dict | None = None, model2: str = "") -> dict:
    """a chat with one model, or (12d.2) two answering the same messages"""
    for mid in (model, model2) if model2 else (model,):
        row = model_row(mid)
        if not row or not row["chat"]:
            raise ValueError("that model isn't offered here" + (
                ": base models have no chat format" if row and row["why_not"] == "base" else ""))
    if model2 == model:
        raise ValueError("compare two different models")
    c = {"id": uuid.uuid4().hex[:12], "who": _who(by), "by": by, "model": model,
         "model2": model2 or "",
         "settings": {k: v for k, v in (settings or {}).items() if k in FIELDS},
         "title": "", "messages": [], "created_at": now(), "updated_at": now()}
    db.chat_put(c)
    return c


def get_chat(chat_id: str, by: str) -> dict:
    """a chat of this person's — never another's, whatever the id"""
    c = db.chat_get(chat_id)
    if not c or c["who"] != _who(by):
        raise KeyError(chat_id)
    return c


def list_chats(by: str) -> list[dict]:
    return [{"id": c["id"], "title": c["title"] or "New chat", "model": c["model"],
             "model2": c.get("model2") or "",
             # 12d.2: "Qwen3-0.6B vs SmolLM2-360M"
             "names": " vs ".join(m.split("/")[-1] for m in (c["model"], c.get("model2")) if m),
             "updated_at": c["updated_at"]}
            for c in db.chat_list(_who(by), config.CHAT_LIST_N)]
