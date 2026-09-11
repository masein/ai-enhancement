"""Hub-metadata preflight: everything we can learn about a model BEFORE spending
GPU on it, from a few KB of metadata instead of a download.

What it decides, and why:

  * exists / gated — fail in seconds with a message a friend can act on, not an
    hour later inside a traceback.
  * trust_remote_code — a config.json with an `auto_map` needs the repo's own
    Python executed to load. On a shared box running other people's jobs, we do
    not execute submitted code. Rejected, with the reason.
  * params — size cap (weights alone for a >4B bf16 model crowd the shared card),
    and the dashboard's scaling axis.
  * vocab -> batch -> VRAM need — the logits law from run_benchmarks.sh:
        memory ≈ batch × seq_len × vocab × 4 bytes × ~2.5
    measured on this card: gemma-3-270m (262K vocab) tried 11.6 GiB at batch 8.
    We pick the largest batch in {8,4,2,1} whose estimate fits the job budget.
  * chat template — presence is evidence, not proof: a checkpoint inherits its
    parent's template through the tokenizer, so 'auto' resolves here only when
    the name corroborates it, and refuses otherwise. The template's hash and the
    reason for the decision are recorded on the run, because applying a template
    to the wrong kind of model moves scores by tens of points.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path

from . import config


class PreflightError(Exception):
    """Human-readable rejection; goes verbatim into the submission's error field."""


SEQ_LEN = 2048          # few-shot prompts on these tasks approach the ctx window
LOGITS_FACTOR = 4 * 2.5  # fp32 logits plus softmax working copies, empirically ~2.5x
OVERHEAD_GB = 1.5        # CUDA context, activations, KV for generation tasks


def estimate(vocab: int, params: int | None) -> tuple[int, float]:
    """Pick (batch, need_gb): largest batch in {8,4,2,1} that fits MAX_JOB_GB."""
    weights_gb = (params * 2 / 1e9) if params else 1.0     # bf16; unknown -> assume small
    for batch in (8, 4, 2, 1):
        logits_gb = batch * SEQ_LEN * vocab * LOGITS_FACTOR / 1e9
        need = logits_gb + weights_gb + OVERHEAD_GB
        if need <= config.MAX_JOB_GB or batch == 1:
            return batch, round(need, 2)
    return 1, round(weights_gb + SEQ_LEN * vocab * LOGITS_FACTOR / 1e9 + OVERHEAD_GB, 2)


LOCAL_PREFIX = "local/"


def _arch_from_config(cfg: dict) -> dict:
    """The model's shape, from its config.json — architecture name, hidden size,
    layer count, head count, context length, vocab. Handles the two naming eras
    (hidden_size/num_hidden_layers vs GPT-2's n_embd/n_layer) and multimodal
    wrappers that nest the text model under text_config."""
    tc = cfg.get("text_config") or {}
    pick = lambda *keys: next((v for src in (cfg, tc) for k in keys
                               if (v := src.get(k)) is not None), None)
    return {
        "arch": (cfg.get("architectures") or tc.get("architectures") or [None])[0],
        "hidden": pick("hidden_size", "n_embd", "d_model"),
        "layers": pick("num_hidden_layers", "n_layer", "num_layers"),
        "heads": pick("num_attention_heads", "n_head"),
        "ctx": pick("max_position_embeddings", "n_positions", "n_ctx"),
        "vocab": pick("vocab_size"),
    }


# ---------------------------------------------------------------------------
# chat-template policy
#
# Applying a chat template to a base model (or withholding it from an instruct
# model) moves multiple-choice scores by tens of points, so the decision has to
# be recorded, not just made. Detection alone is not enough: a checkpoint saved
# from a tokenizer that was initialized from an instruct model INHERITS that
# model's chat template, so "a template exists" does not mean "this is a chat
# model".
#
# Where that inheritance actually happens decides how hard we push back:
#
#   local/<name>  An uploaded checkpoint. Inheriting a parent's template is the
#                 NORMAL case here and the person submitting knows what they
#                 trained, so 'auto' + template + no name evidence is REFUSED.
#                 This is the case that produced a three-to-seven point error on
#                 a colleague's model before anyone noticed.
#   org/model     A Hub repo. Publishing a chat template usually does mean the
#                 model is a chat model — Qwen3-0.6B ships one and says nothing
#                 in its name. Refusing those was friction with no safety win,
#                 so detection stands and the run is flagged as unconfirmed.
# ---------------------------------------------------------------------------

_INSTRUCT_WORDS = ("instruct", "instruction", "chat", "sft", "dpo", "orpo",
                   "rlhf", "tulu", "zephyr", "assistant")


def _looks_instruct(hf_id: str) -> bool:
    """Does the model's NAME claim to be instruction-tuned? ('-it' is a whole
    token, so 'gemma-3-270m-it' matches but 'bit-net' does not.)"""
    name = hf_id.split("/")[-1].lower()
    tokens = set(re.split(r"[^a-z0-9]+", name))
    return "it" in tokens or any(w in name for w in _INSTRUCT_WORDS)


def _safetensors_params(files: list[Path]) -> tuple[int, str] | None:
    """(exact parameter count, stored dtype) read from the safetensors headers.

    A safetensors file starts with an 8-byte little-endian header length and
    that many bytes of JSON naming every tensor's dtype and shape, so the true
    count costs one small read instead of loading the weights. Worth doing
    rather than dividing file size by 2: that assumes bf16, and an fp32
    checkpoint would read as twice its real size — which also inflates it
    against the parameter cap and could reject a model that fits. Tied weights
    (a shared embedding/output head) are stored once, so summing is correct."""
    total, dtypes = 0, {}
    try:
        for f in files:
            with open(f, "rb") as fh:
                raw = fh.read(8)
                if len(raw) < 8:
                    return None
                n = struct.unpack("<Q", raw)[0]
                if not 0 < n < 100_000_000:        # not a safetensors header
                    return None
                hdr = json.loads(fh.read(n))
            for key, spec in hdr.items():
                if key == "__metadata__" or not isinstance(spec, dict):
                    continue
                numel = 1
                for d in spec.get("shape") or []:
                    numel *= int(d)
                total += numel
                dt = str(spec.get("dtype", "?"))
                dtypes[dt] = dtypes.get(dt, 0) + numel
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return None
    if not total:
        return None
    return total, max(dtypes, key=dtypes.get)      # the dtype most weights use


def _code_shas(d: Path) -> list[str]:
    """sha256 (12 hex) of every .py in the artifact, sorted by filename.

    Two jobs: provenance — what code produced this score is as much a part of
    the run as the n-shot count — and the allowlist, since a hash is the only
    honest way to say 'this exact file was reviewed'."""
    out = []
    for f in sorted(d.rglob("*.py")):
        try:
            out.append(f"{f.relative_to(d)}:"
                       + hashlib.sha256(f.read_bytes()).hexdigest()[:12])
        except OSError:
            out.append(f"{f.relative_to(d)}:unreadable")
    return out


def _moe_params(cfg: dict, total: int | None) -> dict:
    """Total vs active-per-token parameters.

    A sparse model loads every expert (so VRAM and the size cap follow the
    total) but routes each token through a few (so capability comparisons
    follow the active count). Reporting only one of the two numbers makes a
    MoE look either unfairly large or unfairly small next to a dense model."""
    tc = cfg.get("text_config") or {}
    g = lambda *keys: next((v for src in (cfg, tc) for k in keys
                            if (v := src.get(k)) is not None), None)
    n_exp = g("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
    per_tok = g("num_experts_per_tok", "num_experts_per_token", "moe_topk", "top_k")
    declared = g("num_active_params", "active_params", "num_activated_params")
    if declared:
        return {"experts": n_exp, "experts_per_tok": per_tok,
                "active_params": int(declared), "active_src": "config"}
    if not (n_exp and per_tok and total and n_exp > 1):
        return {"experts": n_exp, "experts_per_tok": per_tok,
                "active_params": None, "active_src": None}
    # estimate: the expert MLPs are the only part that is sparse, so scale the
    # expert block by the routing fraction and leave everything else whole
    hidden, layers = g("hidden_size", "n_embd", "d_model"), g("num_hidden_layers", "n_layer")
    inter = g("moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
    if not (hidden and layers and inter):
        return {"experts": n_exp, "experts_per_tok": per_tok,
                "active_params": None, "active_src": None}
    per_expert = 3 * hidden * inter                  # gate/up/down
    expert_total = per_expert * n_exp * layers
    active = total - expert_total + per_expert * per_tok * layers
    return {"experts": n_exp, "experts_per_tok": per_tok,
            "active_params": int(max(active, 0)) or None, "active_src": "estimated"}


def _template_id(text: str | None, src: str | None) -> dict:
    """A stable short hash of the template text — the thing that has to match
    for two runs to be comparable. The text itself is too long for a table."""
    if not text:
        return {"tmpl_sha": None, "tmpl_src": None}
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return {"tmpl_sha": sha, "tmpl_src": src}


def resolve_kind(hf_id: str, requested: str,
                 has_template: bool) -> tuple[str, str, bool]:
    """(kind, reason, unconfirmed). Refuses only the case where guessing has
    actually cost us a wrong number: an UPLOADED checkpoint that ships a
    template with nothing in its name to corroborate it."""
    claims = _looks_instruct(hf_id)
    if requested in ("base", "instruct"):
        conflict = (requested == "instruct" and not has_template)
        return requested, ("submitter said %s%s" % (
            requested, "; note: no chat template found in the repo, so none is "
                       "applied" if conflict else "")), False
    if not has_template:
        return "base", ("no chat template in the repo"
                        + ("; name suggests instruct, but a template cannot be "
                           "invented — evaluated as base" if claims else "")), False
    if claims:
        return "instruct", "chat template present and the name says instruct", False
    if hf_id.startswith(LOCAL_PREFIX):
        raise PreflightError(
            f"{hf_id} is ambiguous: this uploaded checkpoint ships a chat "
            f"template, but nothing in its name says it is instruction-tuned. A "
            f"checkpoint saved from an instruct model's tokenizer inherits that "
            f"template even when the weights are a base model, and applying it "
            f"moves multiple-choice scores by tens of points. Resubmit with "
            f"kind=\"base\" (raw completions — right for a pretrained "
            f"checkpoint) or kind=\"instruct\" (apply the template) so the run "
            f"is comparable and the choice is on the record.")
    # Hub repo: take the repo's word for it, but mark the run so the report can
    # say the template policy rests on detection alone
    return "instruct", ("chat template present; the repo name does not say "
                        "instruct, so this is the Hub's word for it — submit "
                        "kind=base explicitly if these weights are a pretrained "
                        "checkpoint"), True


def _preflight_local(name: str, allow_remote_code: bool = False) -> dict:
    """An uploaded artifact: same decisions as the Hub path, answered from disk."""
    d = config.ARTIFACTS_DIR / name
    if not d.is_dir():
        raise PreflightError(f"no uploaded artifact named {name!r} — upload it first "
                             f"(POST /api/artifacts/{name}) or check the name.")
    try:
        cfg = json.loads((d / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise PreflightError(f"artifact {name!r} has no readable config.json — not a "
                             f"loadable checkpoint.") from e
    code_shas, remote_code = [], False
    if cfg.get("auto_map"):
        if not allow_remote_code:
            raise PreflightError(
                f"artifact {name!r} carries an auto_map, so loading it executes the "
                f"Python shipped in the upload. That is off by default. Resubmit "
                f"with allow_remote_code=true (the request needs the team's "
                f"X-Token), and the server must be running with "
                f"ALLOW_REMOTE_CODE=1, SUBMIT_TOKEN and EVAL_USER set — see "
                f"SERVICE.md § custom model code.")
        blocked = config.remote_code_blocked()
        if blocked:
            raise PreflightError(f"artifact {name!r} needs custom model code, but "
                                 f"{blocked}.")
        code_shas = _code_shas(d)
        if not code_shas:
            raise PreflightError(
                f"artifact {name!r} declares an auto_map but ships no .py files — "
                f"the modeling/configuration modules it points at are missing.")
        if config.REMOTE_CODE_SHAS:
            unknown = [c for c in code_shas
                       if c.split(":")[-1] not in config.REMOTE_CODE_SHAS]
            if unknown:
                raise PreflightError(
                    f"artifact {name!r} ships code that is not on this server's "
                    f"allowlist: {', '.join(unknown)}. Register the hash in "
                    f"REMOTE_CODE_SHAS after reviewing the file, or clear the "
                    f"allowlist to accept any team upload.")
        remote_code = True
    # pickled weights execute arbitrary code on load; only safetensors are evaluated
    if list(d.glob("*.bin")):
        raise PreflightError(
            f"artifact {name!r} contains pickle-format weights (*.bin), which execute "
            f"code when loaded. Re-save with save_pretrained(..., safe_serialization=True) "
            f"— the default in modern transformers — and re-upload.")
    st = list(d.glob("*.safetensors"))
    if not st:
        raise PreflightError(f"artifact {name!r} has no *.safetensors weights.")
    vocab = cfg.get("vocab_size") or (cfg.get("text_config") or {}).get("vocab_size")
    if not vocab:
        raise PreflightError(f"could not read vocab_size from {name!r}'s config.json.")
    exact = _safetensors_params(st)
    if exact:
        params, stored_dtype, params_src = exact[0], exact[1], "safetensors header"
    else:   # not readable as safetensors headers — fall back to the old estimate
        params, stored_dtype, params_src = (
            int(sum(f.stat().st_size for f in st) / 2), None, "file size (bf16 assumed)")
    if params / 1e9 > config.MAX_PARAMS_B:
        raise PreflightError(f"artifact {name!r} is {params / 1e9:.2f}B params "
                             f"({params_src}); the cap is {config.MAX_PARAMS_B:g}B.")
    tok_cfg = {}
    if (d / "tokenizer_config.json").exists():
        try:
            tok_cfg = json.loads((d / "tokenizer_config.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
    tmpl, tmpl_src = tok_cfg.get("chat_template"), "tokenizer_config.json"
    if not tmpl and (d / "chat_template.jinja").exists():
        try:
            tmpl, tmpl_src = (d / "chat_template.jinja").read_text(), "chat_template.jinja"
        except OSError:
            tmpl = None
    if isinstance(tmpl, list):        # multi-template repos: hash the whole set
        tmpl = json.dumps(tmpl, sort_keys=True)
    batch, need = estimate(int(vocab), params)
    return {"params": params, "vocab": int(vocab), "batch": batch, "need_gb": need,
            "kind_detected": "instruct" if tmpl else "base",
            "has_template": bool(tmpl),
            "remote_code": remote_code,
            "architectures": cfg.get("architectures") or [],
            "archinfo": {**_arch_from_config(cfg),
                         **_template_id(tmpl, tmpl_src if tmpl else None),
                         **_moe_params(cfg, params),
                         # the dtype the WEIGHTS are stored in, which is not the
                         # dtype we evaluate in — transformers loads a checkpoint
                         # in its stored precision, and an fp16 checkpoint
                         # full-fine-tuned without a scaler is how a loss goes NaN
                         "stored_dtype": stored_dtype, "params_src": params_src,
                         "model_type": cfg.get("model_type"),
                         "remote_code": remote_code,
                         "code_sha": code_shas or None}}


def preflight(hf_id: str, requested_kind: str = "auto",
              allow_remote_code: bool = False) -> dict:
    """Metadata + the resolved template policy. `requested_kind` is the
    submitter's choice; 'auto' may be refused as ambiguous (see resolve_kind)."""
    meta = _preflight(hf_id, allow_remote_code)
    kind, reason, unconfirmed = resolve_kind(
        hf_id, requested_kind, meta.get("has_template", False))
    meta["kind"] = kind
    meta["kind_reason"] = reason
    if unconfirmed:      # rides in archinfo so it reaches the report and exports
        meta.setdefault("archinfo", {})["kind_unconfirmed"] = True
    return meta


def _preflight(hf_id: str, allow_remote_code: bool = False) -> dict:
    if hf_id.startswith(LOCAL_PREFIX):            # uploaded artifact — never touches the Hub
        return _preflight_local(hf_id[len(LOCAL_PREFIX):], allow_remote_code)
    if os.environ.get("STUB_PREFLIGHT") == "1":   # offline tests
        batch, need = estimate(50304, 14_000_000)
        return {"params": 14_000_000, "vocab": 50304, "batch": batch,
                "need_gb": need, "kind_detected": "base", "has_template": False,
                "architectures": ["stub"],
                "archinfo": {"arch": "StubForCausalLM", "hidden": 128, "layers": 6,
                             "heads": 4, "ctx": 2048, "vocab": 50304,
                             "tmpl_sha": None, "tmpl_src": None}}

    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError,
                                            RepositoryNotFoundError)
    except ImportError as e:
        raise PreflightError(f"server env is missing huggingface_hub: {e}") from e

    api = HfApi()
    try:
        info = api.model_info(hf_id)
    except GatedRepoError as e:
        raise PreflightError(
            f"{hf_id} is gated and this server's HF account has not accepted its "
            f"license. Accept it in a browser at https://huggingface.co/{hf_id} "
            f"(the gate is on the account, not the machine), or submit an ungated "
            f"mirror.") from e
    except RepositoryNotFoundError as e:
        raise PreflightError(f"{hf_id} does not exist on the Hub (typo? private repo "
                             f"this server's token cannot see?)") from e

    params = getattr(getattr(info, "safetensors", None), "total", None)

    def fetch_json(filename):
        try:
            return json.loads(Path(hf_hub_download(hf_id, filename)).read_text())
        except (EntryNotFoundError, OSError, json.JSONDecodeError):
            return {}

    cfg = fetch_json("config.json")
    if not cfg:
        raise PreflightError(f"{hf_id} has no readable config.json — not a loadable "
                             f"transformers checkpoint.")
    if cfg.get("auto_map"):
        # Hub repos stay refused even when remote code is enabled: the gate is
        # "a teammate uploaded this to our box", and a Hub id carries no such
        # signal — anyone could point us at any repo. Upload it as an artifact.
        raise PreflightError(
            f"{hf_id} requires trust_remote_code=True (custom modeling code in the "
            f"repo). Code from the Hub is never executed here, whatever the "
            f"server settings. If this is your model, upload it as an artifact "
            f"(POST /api/artifacts/<name>) and submit it with "
            f"allow_remote_code=true.")

    # vocab_size sometimes lives under text_config for multimodal wrappers
    vocab = cfg.get("vocab_size") or (cfg.get("text_config") or {}).get("vocab_size")
    if not vocab:
        raise PreflightError(f"could not read vocab_size from {hf_id}'s config.json — "
                             f"unusual architecture; run it manually if you trust it.")

    if params and params / 1e9 > config.MAX_PARAMS_B:
        raise PreflightError(
            f"{hf_id} has {params / 1e9:.1f}B parameters; this service caps at "
            f"{config.MAX_PARAMS_B:g}B to keep the shared card usable "
            f"(MAX_PARAMS_B raises it).")

    siblings = {s.rfilename for s in (info.siblings or [])}
    tok_cfg = fetch_json("tokenizer_config.json") if "tokenizer_config.json" in siblings else {}
    tmpl, tmpl_src = tok_cfg.get("chat_template"), "tokenizer_config.json"
    if not tmpl and "chat_template.jinja" in siblings:
        try:
            tmpl = Path(hf_hub_download(hf_id, "chat_template.jinja")).read_text()
            tmpl_src = "chat_template.jinja"
        except (EntryNotFoundError, OSError):
            tmpl = None
    if isinstance(tmpl, list):        # multi-template repos: hash the whole set
        tmpl = json.dumps(tmpl, sort_keys=True)

    batch, need = estimate(int(vocab), params)
    return {
        "params": params,
        "vocab": int(vocab),
        "batch": batch,
        "need_gb": need,
        "kind_detected": "instruct" if tmpl else "base",
        "has_template": bool(tmpl),
        "architectures": cfg.get("architectures") or [],
        "archinfo": {**_arch_from_config(cfg),
                     **_template_id(tmpl, tmpl_src if tmpl else None)},
    }
