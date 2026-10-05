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


# 16b review: an untrusted file's header keeps only what the board reads. A
# string is kept to MAX_KEEP characters, an array never; everything else is
# skipped by seeking, never read into memory
MAX_KEEP = 256
MAX_DEPTH = 3                    # arrays of arrays: no model's goes deeper
_KEEP = {"general.architecture", "general.name", "general.size_label", "general.file_type"}
_KEEP_SUFFIX = (".expert_count", ".expert_used_count", ".context_length")
_PRESENT = "tokenizer.chat_template"     # whether it is there, never what it says


class _Reader:
    def __init__(self, fh):
        self.fh = fh
        fh.seek(0, 2)
        self.size = fh.tell()
        fh.seek(0)

    def take(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise NotGguf("the header is cut short")
        return b

    def skip(self, n: int) -> None:
        if n < 0 or self.fh.tell() + n > self.size:
            raise NotGguf("the header is cut short")
        self.fh.seek(n, 1)

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def count(self) -> int:
        n = self.unpack("<Q")
        if n > MAX_COUNT:
            raise NotGguf(f"a count of {n} is more than any model has")
        return n

    def string(self, keep: bool = True) -> str | None:
        n = self.unpack("<Q")
        if n > MAX_STR:
            raise NotGguf("a string longer than any header holds")
        if not keep:
            self.skip(n)
            return None
        k = min(n, MAX_KEEP)
        s = self.take(k).decode("utf-8", "replace")
        self.skip(n - k)
        return s

    def value(self, t: int, keep: bool = True, depth: int = 0):
        if t in _SCALAR:
            return self.unpack(_SCALAR[t])
        if t == _STRING:
            return self.string(keep)
        if t == _ARRAY:
            if depth >= MAX_DEPTH:
                raise NotGguf("arrays nested deeper than any model's")
            et, n = self.unpack("<I"), self.count()
            if et in _SCALAR:
                self.skip(n * struct.calcsize(_SCALAR[et]))     # a vocabulary's scores
            else:
                for _ in range(n):
                    self.value(et, False, depth + 1)
            return None                                         # an array is never kept
        raise NotGguf(f"an unknown value type {t}")


def read(path: str | Path) -> dict | None:
    """{arch, name, size_label, params, active_params, expert_count,
    expert_used_count, file_type, context_length, chat_template} from the
    file's header, or None when it isn't a GGUF one can read — a malformed
    one included, whatever it trips"""
    try:
        with open(path, "rb") as fh:
            return _read(_Reader(fh))
    except (OSError, NotGguf, struct.error, UnicodeDecodeError, MemoryError, RecursionError,
            TypeError, ValueError, OverflowError):
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
        if key == _PRESENT:
            r.value(t, keep=False)
            meta[key] = t == _STRING
            continue
        want = key in _KEEP or key.endswith(_KEEP_SUFFIX)
        v = r.value(t, keep=want)
        if want:
            meta[key] = v
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
    ctx = meta.get(f"{arch}.context_length")
    return {"arch": arch or None, "name": meta.get("general.name"),
            "size_label": meta.get("general.size_label"),
            "params": total or None, "active_params": active,
            "expert_count": n_exp if isinstance(n_exp, int) else None,
            "expert_used_count": used if isinstance(used, int) else None,
            "file_type": FILE_TYPES.get(ft, ft) if isinstance(ft, int) else None,
            "context_length": ctx if isinstance(ctx, int) else None,
            "chat_template": meta.get(_PRESENT) is True,
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
