"""
Tokenizers: turning text into integers.

Two implementations, deliberately:

CharTokenizer   one integer per character. Vocab ~90. Trivially debuggable, and
                the sequences are long (1 token per char). Use it to learn.

ByteBPETokenizer  byte-level Byte-Pair Encoding, the same family GPT-2/GPT-4,
                Llama and Qwen use. Starts from the 256 raw bytes (so it can
                encode ANY input, no <unk> ever) and repeatedly merges the most
                frequent adjacent pair into a new token.

Why this matters for your job: tokenization decides your sequence lengths, which
decides your token budget, which decides your compute cost. It also silently
decides what your model finds hard. A tokenizer that splits "1234" into "123"+"4"
makes arithmetic harder than one that splits it "1"+"2"+"3"+"4". Every eval
number you produce is downstream of this file.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# GPT-2-style pre-tokenization. BPE merges are never allowed to cross these
# boundaries, which is what stops the tokenizer learning a single token for
# " the cat sat". ASCII-only classes here to avoid a regex-module dependency;
# real tokenizers use \p{L}/\p{N} so they behave on non-English text.
_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)

EOS = "<|endoftext|>"
PAD = "<|pad|>"


# Always-present characters, even if the training corpus never used them.
#
# Why: a vocabulary derived purely from the pretraining corpus cannot represent
# anything outside it. Fine-tune on a prompt template containing "Q:" after
# pretraining on digits only, and every letter silently vanishes at encode time —
# you get a broken model and no error message. Real tokenizers are byte-level so
# this is impossible by construction; this constant is the char-level equivalent.
BASE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \n\t.,;:!?'\"()[]{}<>-_+=*/\\|@#$%^&~`"
)


class CharTokenizer:
    """One token per character. Vocab = BASE_ALPHABET + whatever the corpus used."""

    def __init__(self, chars: Iterable[str]):
        self.specials = [EOS, PAD]
        self.chars = sorted(set(chars))
        self.itos = self.chars + self.specials
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.eos_id = self.stoi[EOS]
        self.pad_id = self.stoi[PAD]

    # -- construction ----------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], include_base: bool = True, **_ignored) -> "CharTokenizer":
        chars: set[str] = set(BASE_ALPHABET) if include_base else set()
        for t in texts:
            chars.update(t)
        return cls(chars)

    # -- api -------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        # Unknown characters are dropped rather than crashing: a char tokenizer
        # genuinely cannot represent them. This is one concrete reason to prefer
        # byte-level BPE for anything real.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"kind": "char", "chars": self.chars}), encoding="utf-8"
        )


class ByteBPETokenizer:
    """
    Byte-level BPE.

    Vocabulary layout:
        0..255            the raw bytes  (so nothing is ever unrepresentable)
        256..256+n_merges the learned merges
        last two          EOS, PAD

    Training is the classic algorithm: count adjacent byte-pair frequencies over
    the corpus, merge the most frequent pair, repeat until you hit vocab_size.
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._rebuild()

    def _rebuild(self) -> None:
        # rank[(a, b)] = merge order. Lower rank == merged earlier.
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        # id -> bytes, needed for decoding
        self.id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.id_to_bytes[256 + i] = self.id_to_bytes[a] + self.id_to_bytes[b]
        self.eos_id = 256 + len(self.merges)
        self.pad_id = self.eos_id + 1
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + 2  # +2 for EOS and PAD

    # -- training --------------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 1024, verbose: bool = False):
        n_merges = max(0, vocab_size - 258)  # 256 bytes + EOS + PAD

        # Count pre-tokenized "words" once; merging then operates on the
        # weighted word list instead of the whole corpus. Same result,
        # dramatically less work.
        word_freq: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for chunk in _SPLIT_PATTERN.findall(text):
                word_freq[tuple(chunk.encode("utf-8"))] += 1

        words = {w: list(w) for w in word_freq}
        merges: list[tuple[int, int]] = []

        for step in range(n_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for word, freq in word_freq.items():
                syms = words[word]
                for a, b in zip(syms, syms[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best, count = pair_counts.most_common(1)[0]
            if count < 2:  # nothing worth merging left
                break
            new_id = 256 + step
            merges.append(best)
            for word in words:
                words[word] = _merge_once(words[word], best, new_id)
            if verbose and step % 100 == 0:
                print(f"  merge {step:5d}  {best} -> {new_id}  (count={count})")

        return cls(merges)

    # -- api -------------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        syms = list(chunk.encode("utf-8"))
        while len(syms) >= 2:
            # Greedily apply the lowest-rank (earliest-learned) available merge.
            candidate = min(
                (p for p in zip(syms, syms[1:]) if p in self.rank),
                key=lambda p: self.rank[p],
                default=None,
            )
            if candidate is None:
                break
            syms = _merge_once(syms, candidate, 256 + self.rank[candidate])
        self._cache[chunk] = syms
        return syms

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        # Handle special tokens as atomic units wherever they appear verbatim.
        for piece in re.split(f"({re.escape(EOS)}|{re.escape(PAD)})", text):
            if piece == EOS:
                out.append(self.eos_id)
            elif piece == PAD:
                out.append(self.pad_id)
            elif piece:
                for chunk in _SPLIT_PATTERN.findall(piece):
                    out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            if i == self.eos_id:
                buf.extend(EOS.encode())
            elif i == self.pad_id:
                continue
            elif i in self.id_to_bytes:
                buf.extend(self.id_to_bytes[i])
        # errors="replace": a truncated multi-byte character is normal when you
        # cut a generation short, and must not raise.
        return buf.decode("utf-8", errors="replace")

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"kind": "bpe", "merges": self.merges}), encoding="utf-8"
        )


def _merge_once(syms: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `syms` with new_id."""
    out: list[int] = []
    i = 0
    while i < len(syms):
        if i < len(syms) - 1 and (syms[i], syms[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out


def tokenizer_to_dict(tok) -> dict:
    if isinstance(tok, CharTokenizer):
        return {"kind": "char", "chars": tok.chars}
    return {"kind": "bpe", "merges": [list(m) for m in tok.merges]}


def tokenizer_from_dict(blob: dict):
    if blob["kind"] == "char":
        return CharTokenizer(blob["chars"])
    return ByteBPETokenizer([tuple(m) for m in blob["merges"]])


def load_tokenizer(path: str | Path):
    return tokenizer_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_tokenizer(kind: str, texts: Iterable[str], vocab_size: int = 1024):
    texts = list(texts)
    if kind == "char":
        return CharTokenizer.train(texts)
    if kind == "bpe":
        return ByteBPETokenizer.train(texts, vocab_size=vocab_size)
    raise ValueError(f"unknown tokenizer kind: {kind!r} (expected 'char' or 'bpe')")


if __name__ == "__main__":  # quick manual check: python -m aienh.tokenizer
    sample = ["the cat sat on the mat. " * 50, "12 + 34 = 46\n" * 50]
    for kind in ("char", "bpe"):
        tok = build_tokenizer(kind, sample, vocab_size=400)
        ids = tok.encode("the cat sat on the mat.")
        print(f"{kind:5s} vocab={tok.vocab_size:5d} len={len(ids):3d} "
              f"roundtrip_ok={tok.decode(ids) == 'the cat sat on the mat.'}")
