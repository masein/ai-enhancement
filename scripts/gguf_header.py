#!/usr/bin/env python3
"""16.1: what a GGUF file says about itself, from its header alone — never its
weights, never run. Standard library only: the host's GGUF worker reads it
(scripts/gguf_worker.py), and nothing from the file is executed.

    python scripts/gguf_header.py model.gguf

The header (GGUF v2 and v3, little-endian): the magic "GGUF", a version, the
number of tensors and of metadata entries, the entries (a key, a type, a
value), then each tensor's name, shape and type. From those:

- params: every tensor's elements, summed — the model's total size;
- active_params: for a mixture of experts, what one token uses — every tensor
  but the experts', and the experts' share (expert_used_count of
  expert_count); None for a dense model;
- arch, name, size_label, file_type, context_length, and whether it carries a
  chat template.

A file that isn't a GGUF, or that is cut or corrupt, gives None: an unknown
size, never a guess. Bounds keep a hostile header from asking for much."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

MAGIC = b"GGUF"
MAX_COUNT = 1_000_000            # tensors, entries, array items: far above any model's
MAX_STR = 64 * 1024 * 1024       # one string (a chat template is a few KB)
MAX_DIMS = 8

# value types (ggml's gguf_type)
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
           11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
# general.file_type: llama.cpp's llama_ftype, the names people use
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
              10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
              16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
              22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
              28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
              37: "TQ2_0"}


class NotGguf(ValueError):
    pass


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def take(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise NotGguf("the header is cut short")
        return b

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def count(self) -> int:
        n = self.unpack("<Q")
        if n > MAX_COUNT:
            raise NotGguf(f"a count of {n} is more than any model has")
        return n

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > MAX_STR:
            raise NotGguf("a string longer than any header holds")
        return self.take(n).decode("utf-8", "replace")

    def value(self, t: int, keep: bool = True):
        if t in _SCALAR:
            return self.unpack(_SCALAR[t])
        if t == _STRING:
            return self.string()
        if t == _ARRAY:
            et, n = self.unpack("<I"), self.count()
            if et in _SCALAR and not keep:
                self.take(n * struct.calcsize(_SCALAR[et]))      # a vocabulary's scores: skipped
                return None
            items = [self.value(et, keep) for _ in range(n)]
            return items if keep else None
        raise NotGguf(f"an unknown value type {t}")


def read(path: str | Path) -> dict | None:
    """{arch, name, size_label, params, active_params, expert_count,
    expert_used_count, file_type, context_length, chat_template} from the
    file's header, or None when it isn't a GGUF one can read"""
    try:
        with open(path, "rb") as fh:
            return _read(_Reader(fh))
    except (OSError, NotGguf, struct.error, UnicodeDecodeError, MemoryError):
        return None


def _read(r: _Reader) -> dict:
    if r.take(4) != MAGIC:
        raise NotGguf("not a GGUF file")
    version = r.unpack("<I")
    if version not in (2, 3):
        raise NotGguf(f"GGUF version {version}: only 2 and 3 are read")
    n_tensors, n_kv = r.count(), r.count()
    meta: dict = {}
    for _ in range(n_kv):
        key, t = r.string(), r.unpack("<I")
        # the tokenizer's lists are long and not needed: read past them
        keep = not key.startswith("tokenizer.ggml.")
        meta[key] = r.value(t, keep)
    total = experts = 0
    for _ in range(n_tensors):
        name = r.string()
        dims = r.unpack("<I")
        if dims > MAX_DIMS:
            raise NotGguf(f"a tensor of {dims} dimensions")
        n = 1
        for _ in range(dims):
            n *= r.unpack("<Q")
        r.unpack("<I")                                       # its type
        r.unpack("<Q")                                       # its offset
        total += n
        if "_exps" in name:                                  # ffn_{gate,up,down}_exps
            experts += n
    arch = str(meta.get("general.architecture") or "")
    n_exp = meta.get(f"{arch}.expert_count")
    used = meta.get(f"{arch}.expert_used_count")
    active = None
    if experts and isinstance(n_exp, int) and isinstance(used, int) and 0 < used <= n_exp:
        active = (total - experts) + experts * used // n_exp
    ft = meta.get("general.file_type")
    return {"arch": arch or None, "name": meta.get("general.name"),
            "size_label": meta.get("general.size_label"),
            "params": total or None, "active_params": active,
            "expert_count": n_exp, "expert_used_count": used,
            "file_type": FILE_TYPES.get(ft, ft) if ft is not None else None,
            "context_length": meta.get(f"{arch}.context_length"),
            "chat_template": isinstance(meta.get("tokenizer.chat_template"), str),
            "version": version}


def main(argv: list[str] | None = None) -> int:
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        print(__doc__.split("\n\n")[1])
        return 2
    bad = 0
    for p in paths:
        got = read(p)
        bad += got is None
        print(json.dumps({"file": p, **(got or {"error": "not a GGUF file this can read"})}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
