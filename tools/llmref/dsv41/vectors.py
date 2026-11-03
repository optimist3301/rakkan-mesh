#!/usr/bin/env python3
"""Test vectors for the Go implementations of what the reference computes
with Python libraries, for the real model.

The node must import and run DeepSeek-V4.1 without Python, so everything
the reference takes from tokenizers, numpy and sympy is reimplemented in
Go. This records the reference's answers for the real tokenizer and
config, so the Go code can be checked without them:

  token_map      Engram's compressed token map (tokenizers normalizers):
                 all 129280 ids, and its size 99092
  primes         Engram bucket primes (sympy.isprime, drawn in order)
  offsets        Engram row offsets per (n-gram size, head)
  multipliers    Engram hash multipliers (numpy PCG64 via SeedSequence,
                 bounded int64): for the model's layers and for layers
                 0..63, which exercises the generator on many seeds
  hashes         Engram hash ids of the tokens of every text below
  tokenize       token ids of texts covering the pre-tokenizer's splits,
                 CJK, digits, whitespace runs, accents, emoji, special
                 tokens and invalid UTF-8 boundaries (tokenizers library)

usage: vectors.py OUT.json.gz
"""
import gzip
import json
import sys

import numpy as np
import torch

import fetch
import run

TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and   inner   spaces\tand\ttabs\n\nnew lines\r\n",
    "Numbers: 1234567890, 3.14159, -42, 1e10, 0x1F",
    "Привет, мир! Ёжик в тумане.",
    "日本語のテキストとカタカナ、ひらがな。中文测试一下。",
    "Ünïcödé àccénts, ß, İstanbul, ǅ, ﬁ ligature, ①②③, ｆｕｌｌｗｉｄｔｈ",
    "emoji 👍🏽 🇺🇦 👨‍👩‍👧 and symbols ∑∫√ ≠ ≤",
    "code: def f(x):\n    return x**2  # comment\n",
    "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜><think>",
    "mixedCASE_and-dashes/slashes\\back 'quotes' \"double\"",
    "a" * 50 + " " + "ab" * 20,
    "",
    " ",
    "\n",
]


def main():
    out = sys.argv[1]
    m = run.setup()
    import engram as E
    from tokenizers import Tokenizer

    with open(fetch.fetch("tokenizer.json")) as f:
        tok_json = f.read()
    tok = Tokenizer.from_str(tok_json)
    sub = run.SubsetTokenizer(fetch.fetch("tokenizer.json"))
    token_map, compressed = E.build_compressed_token_map(sub)

    args = run.load_args(m, run.os.path.join(run.UPSTREAM, "config.json"), max_batch_size=1, max_seq_len=512)
    assert compressed == args.engram_compressed_vocab_size
    layout = E.EngramLayout.from_args(args)
    hstate = E.NgramHashState(args, layout, sub)

    tokenize, hashes = [], []
    for text in TEXTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        tokenize.append({"text": text, "ids": ids})
        if ids:
            h = hstate(torch.tensor([ids]), 0)  # [1, L, layers, cols]
            hashes.append({"ids": ids, "hash_ids": h[0].tolist()})

    mults = E.compute_hash_multipliers(tuple(range(64)), args.engram_max_ngram_size, compressed)
    vectors = {
        "revision": fetch.pins()["revision"],
        "vocab_size": len(sub),
        "compressed_vocab_size": compressed,
        "pad_id": int(hstate.pad_id),
        "token_map": token_map,
        "engram_layer_ids": list(layout.layer_ids),
        "primes": [[list(p) for p in layer] for layer in layout.primes],
        "offsets": hstate.offsets.tolist(),
        "multipliers": hstate.multipliers.tolist(),
        "multipliers_by_layer": {
            "max_ngram_size": args.engram_max_ngram_size,
            "vocab_size": compressed,
            "bound": int(max(1, (np.iinfo(np.int64).max // compressed) // 2)),
            "values": mults.tolist(),  # [layer 0..63][lookback]
        },
        "hashes": hashes,
        "tokenize": tokenize,
    }
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False)
    print(f"{len(token_map)} token ids -> {compressed}; {len(tokenize)} texts; written {out}")


if __name__ == "__main__":
    main()
