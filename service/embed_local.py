"""12o.1: the question builder's duplicate check embeds on this server. A small
embedding model, BAAI/bge-small-en-v1.5 (MIT), on the CPU, in the image and
pinned to one commit (the Dockerfile downloads it at build time and checks its
sha256), so no question — the hidden half included — leaves the server to be
embedded. OpenRouter's embeddings stay a choice (QB_EMBED_MODEL=openrouter),
said on the page to send every question out.

Loaded at the first duplicate check, not at startup; the vectors are cached by
the builder, as OpenRouter's were."""

from __future__ import annotations

import threading
from pathlib import Path

from . import config

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
WEIGHTS_SHA256 = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt")
# what a cached vector is keyed by: this model at this commit
IDENT = f"{MODEL}@{REVISION[:12]}"
MAX_TOKENS = 512
BATCH = 32

_lock = threading.Lock()
_loaded: dict = {}


def model_dir() -> Path:
    return Path(config.QB_EMBED_DIR)


def available() -> bool:
    """in this image: every file it needs"""
    d = model_dir()
    return all((d / f).is_file() for f in FILES)


def _load():
    with _lock:
        if not _loaded:
            import torch
            from transformers import AutoModel, AutoTokenizer
            d = str(model_dir())
            tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
            mod = AutoModel.from_pretrained(d, local_files_only=True).to("cpu").eval()
            _loaded.update(torch=torch, tok=tok, model=mod)
    return _loaded


def embed(texts: list[str]) -> list[list[float]] | None:
    """One unit vector per text — the [CLS] token's, as bge is trained to be
    read — or None when the model isn't in this image: the caller keeps its
    13-gram check alone"""
    if not texts:
        return []
    if not available():
        return None
    m = _load()
    torch, tok, mod = m["torch"], m["tok"], m["model"]
    out: list[list[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_TOKENS,
                      return_tensors="pt")
            cls = mod(**enc).last_hidden_state[:, 0]
            out.extend(torch.nn.functional.normalize(cls, p=2, dim=1).tolist())
    return out
