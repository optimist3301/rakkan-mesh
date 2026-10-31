#!/usr/bin/env python3
"""Reference forward pass for Hugging Face LLaMA-family models, in numpy.

Written from the transformers library's definition of Llama, Qwen2 and
Qwen3, independently of the Go engine: rotate-half rotary embeddings on the
weights as published (the engine permutes them at import), the whole
sequence at once under a causal mask (the engine runs token by token with a
cache), matrix products by BLAS (the engine sums in its own fixed order).
Agreement of the logits therefore checks the engine's architecture, its
import and its arithmetic together.

usage: hf_reference.py MODEL_DIR TOKENS OUT
  TOKENS  comma-separated token IDs
  OUT     file receiving the logits of every position, float32 little-endian
"""
import glob
import json
import os
import struct
import sys

import numpy as np


def load_weights(model_dir):
    tensors = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            lo, hi = info["data_offsets"]
            raw = data[lo:hi]
            if info["dtype"] == "BF16":
                arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif info["dtype"] == "F16":
                arr = raw.view(np.float16).astype(np.float32)
            elif info["dtype"] == "F32":
                arr = raw.view(np.float32)
            else:
                raise SystemExit(f"{name}: dtype {info['dtype']}")
            tensors[name] = np.array(arr).reshape(info["shape"])
    return tensors


def rms_norm(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def main():
    model_dir, tokens_arg, out = sys.argv[1:4]
    tokens = [int(t) for t in tokens_arg.split(",")]
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    w = load_weights(model_dir)

    d = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    hd = cfg.get("head_dim") or d // n_heads
    eps = cfg.get("rms_norm_eps", 1e-6)
    theta = cfg.get("rope_theta", 10000.0)
    T = len(tokens)

    inv_freq = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.outer(np.arange(T, dtype=np.float64), inv_freq)
    emb = np.concatenate([ang, ang], axis=-1)
    cos, sin = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)  # [T, hd]

    x = w["model.embed_tokens.weight"][tokens].astype(np.float32)  # [T, d]
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms_norm(x, w[p + "input_layernorm.weight"], eps)
        q = h @ w[p + "self_attn.q_proj.weight"].T
        k = h @ w[p + "self_attn.k_proj.weight"].T
        v = h @ w[p + "self_attn.v_proj.weight"].T
        if p + "self_attn.q_proj.bias" in w:
            q = q + w[p + "self_attn.q_proj.bias"]
            k = k + w[p + "self_attn.k_proj.bias"]
            v = v + w[p + "self_attn.v_proj.bias"]
        q = q.reshape(T, n_heads, hd).transpose(1, 0, 2)  # [H, T, hd]
        k = k.reshape(T, n_kv, hd).transpose(1, 0, 2)
        v = v.reshape(T, n_kv, hd).transpose(1, 0, 2)
        if p + "self_attn.q_norm.weight" in w:
            q = rms_norm(q, w[p + "self_attn.q_norm.weight"], eps)
            k = rms_norm(k, w[p + "self_attn.k_norm.weight"], eps)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = n_heads // n_kv
        k = np.repeat(k, rep, axis=0)
        v = np.repeat(v, rep, axis=0)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        att = (probs @ v).transpose(1, 0, 2).reshape(T, n_heads * hd)
        x = x + att @ w[p + "self_attn.o_proj.weight"].T
        h = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps)
        gate = h @ w[p + "mlp.gate_proj.weight"].T
        up = h @ w[p + "mlp.up_proj.weight"].T
        x = x + ((gate / (1 + np.exp(-gate))) * up) @ w[p + "mlp.down_proj.weight"].T
    x = rms_norm(x, w["model.norm.weight"], eps)
    head = w.get("lm_head.weight", w["model.embed_tokens.weight"])
    logits = (x @ head.T).astype("<f4")
    logits.tofile(out)


if __name__ == "__main__":
    main()
