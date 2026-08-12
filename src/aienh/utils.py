"""Small shared helpers: devices, seeding, run names, config hashing."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def pick_device(requested: str = "auto") -> torch.device:
    """cuda > mps (Apple Silicon) > cpu.

    On a MacBook, `mps` uses the GPU cores and is typically 3-10x faster than cpu
    for these small models. It does not support every op — if you hit a
    NotImplementedError, set PYTORCH_ENABLE_MPS_FALLBACK=1 or pass device=cpu.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    """
    bf16 on modern NVIDIA GPUs: same exponent range as fp32 so no loss scaling is
    needed, half the memory, roughly double the throughput. fp32 elsewhere —
    MPS bf16 support is uneven and CPU bf16 is usually slower, not faster.
    """
    if requested != "auto":
        return {"fp32": torch.float32, "float32": torch.float32,
                "bf16": torch.bfloat16, "fp16": torch.float16}[requested]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def cosine_lr(step: int, total_steps: int, base_lr: float,
              warmup_frac: float = 0.05, min_ratio: float = 0.1) -> float:
    """
    Linear warmup then cosine decay to base_lr * min_ratio.

    warmup: at step 0 the weights are random (or, in fine-tuning, the data is new),
            so gradients are large and badly conditioned; a full-size step there
            can wreck the run. Ramp in.
    cosine: big steps early to travel, small steps late to settle. The nonzero
            floor avoids a dead tail where nothing changes.
    """
    import math as _math
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)


def set_seed(seed: int) -> None:
    """Same seed + same code + same device == same numbers. Reproducibility is
    not a nice-to-have on an eval team: if a run is not reproducible you cannot
    attribute a score change to a model change."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_ADJ = ["brisk", "calm", "dense", "eager", "fluent", "glad", "keen", "lucid",
        "mild", "nimble", "plain", "quick", "sharp", "steady", "terse", "warm"]
_NOUN = ["falcon", "harbor", "lantern", "meadow", "onyx", "pillar", "quartz",
         "ridge", "summit", "thicket", "vector", "willow", "anchor", "beacon"]


def run_name(prefix: str, config_hash: str, seed: int = 0) -> str:
    """
    Readable, unique-ish, and sortable-by-nothing on purpose:  sft-keen-onyx-3f9a

    Why not just a timestamp: humans have to say these out loud in standups and
    type them into filters. Why include the config hash: two runs with the same
    name and different configs is the single most expensive mistake on a
    benchmarking team, because it silently corrupts comparisons.
    """
    rng = random.Random(config_hash + str(seed))
    return f"{prefix}-{rng.choice(_ADJ)}-{rng.choice(_NOUN)}-{config_hash[:4]}"


def config_hash(*configs: Any) -> str:
    """Stable short hash of everything that could change the numbers."""
    blob = json.dumps([_jsonable(c) for c in configs], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def _jsonable(c: Any):
    if hasattr(c, "to_dict"):
        return c.to_dict()
    if hasattr(c, "__dict__"):
        return dict(c.__dict__)
    return c


def git_sha() -> str | None:
    """Record the code version alongside the numbers, or the numbers are hearsay."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def human(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def env_info() -> dict:
    dev = pick_device()
    return {
        "torch": torch.__version__,
        "device": dev.type,
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "git_sha": git_sha(),
    }


def load_config_file(path: str | Path) -> dict:
    """YAML if PyYAML is available, otherwise JSON. Keeps the repo dependency-light."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                f"{p} is YAML but PyYAML is not installed (pip install pyyaml), "
                "or pass a .json config instead"
            ) from e
        return yaml.safe_load(text) or {}
    return json.loads(text)
