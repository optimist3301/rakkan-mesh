"""Load the unmodified reference model (upstream/inference) on the CPU.

Importing this module makes `kernel` resolve to kernel_cpu and puts the
upstream inference directory on the path, so `import model` gets the
reference exactly as published. Pass cuda=True to keep the real tilelang
kernels instead (check_kernels.py does, on a GPU machine).

It also replaces torch's top-k with a stable one: among equal scores the
lower index wins, and selected entries come out by descending score. The
reference leaves ties to torch.topk, which promises no order; the engine
uses this rule, so dumps made here are what it must reproduce. Ties are not
rare: the indexer's ReLU makes many scores exactly zero on short contexts.
"""

import collections
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream", "inference")

_TopK = collections.namedtuple("topk", ["values", "indices"])


def stable_topk(self, k, dim=-1, largest=True, sorted=True):
    values, indices = torch.sort(self, dim=dim, descending=largest, stable=True)
    return _TopK(values.narrow(dim, 0, k), indices.narrow(dim, 0, k))


def setup(cuda=False):
    """Make `import model` load the reference; returns the module."""
    if not cuda:
        import kernel_cpu

        sys.modules["kernel"] = kernel_cpu
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    torch.Tensor.topk = stable_topk
    torch.topk = lambda input, k, dim=-1, largest=True, sorted=True: stable_topk(input, k, dim, largest, sorted)
    import model

    return model


class SubsetTokenizer:
    """The real tokenizer restricted to its first `n` ids, as much of the
    transformers tokenizer interface as engram's token map needs."""

    def __init__(self, path, n=None):
        from tokenizers import Tokenizer

        self.backend_tokenizer = Tokenizer.from_file(path)
        self.n = n or self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    def __len__(self):
        return self.n


def load_args(model_mod, path, **overrides):
    with open(path) as f:
        cfg = json.load(f)
    cfg.update(overrides)
    return model_mod.ModelArgs(**cfg)
