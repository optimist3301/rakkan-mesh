#!/usr/bin/env python3
"""Reference outputs of one real layer of DeepSeek-V4.1-Flash, for the Go
engine's check on real weights (internal/llm/dsv41/real_test.go).

One block of the unmodified model.py, with the published weights of one
layer (converted as convert.py converts them), runs a prompt and then a
few decode steps. Its input is the embedding of real text in all four
hyper-connection copies with the identity mix, as layer 0 receives it:
not the stream the layer sees in the full model, but real weights on real
activations, at the real widths, and long enough that what the tiny
fixture cannot reach is reached: more compressed entries than top-k
selects, a sliding window that wraps, attention over many blocks of 64.

Writes a safetensors file with:
  tokens [T]; input [T, 4, dim] bf16 (the same input every step)
  out [T, 4, dim] bf16, pre_mix [T, 4] f32: the block's outputs per position
  topk.prefill [P, k] i32, topk.decode{i} [1, k] i32 (index sources)
  candidates.prefill [P, W] bool, candidates.decode{i} (candidate source)
  cache.compress_kv [n, head dim] bf16, cache.index_k [n, index dim] bf16
  trace.attn [T, dim] bf16, trace.ffn [T, dim] bf16: the sublayers' outputs
  trace.experts [T, k] i64, trace.expert_weights [T, k] f32: the routing
  trace.attn_in, trace.ffn_in [T, dim] bf16: the sublayers' inputs (after norm)
  trace.freqs [S, rope dim/2, 2] f32: the layer's rotary cos and sin by position
  trace.qr, trace.q, trace.kv, trace.o, trace.low: attention's steps per
    position (query after q_norm, query after RoPE, window KV after
    quantization, sparse attention's output, wo_a's output)

usage: layer_check.py LAYER OUT.safetensors [--prompt N] [--decode N]
Downloads (fetch.py) the layer's shard (6.9 GiB) and the embeddings' (1.2 GiB).
"""
import argparse
import os

import torch

import fetch
import run

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
EMBED_SHARD = "model-00002-of-00048.safetensors"


def shard(layer):
    return f"model-{layer + 3:05d}-of-00048.safetensors"


def load_block(m, args, layer):
    from safetensors import safe_open

    m.default_dtype = torch.float8_e4m3fn  # what Transformer.__init__ sets for dtype "fp8"
    block = m.Block(layer, args, None)
    prefix = f"layers.{layer}."
    state = {}
    with safe_open(fetch.fetch(shard(layer)), "pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                state[k[len(prefix):]] = f.get_tensor(k)
    # convert.py: wo_a dequantized to bf16, packed FP4 experts typed as such.
    w, s = state["attn.wo_a.weight"], state.pop("attn.wo_a.scale")
    w = w.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32)).float() * s[:, None, :, None].float()
    state["attn.wo_a.weight"] = w.flatten(2, 3).flatten(0, 1).bfloat16()
    for k, v in state.items():
        if ".experts." in k and v.dtype == torch.int8:
            state[k] = v.view(torch.float4_e2m1fn_x2)
    block.load_state_dict(state, strict=True)
    return block


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer", type=int)
    ap.add_argument("out")
    ap.add_argument("--prompt", type=int, default=1100)
    ap.add_argument("--decode", type=int, default=6)
    a = ap.parse_args()
    P, D = a.prompt, a.decode

    m = run.setup()
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(os.cpu_count())
    args = run.load_args(m, os.path.join(run.UPSTREAM, "config.json"), max_batch_size=1, max_seq_len=P + D + 64)
    block = load_block(m, args, a.layer)

    with open(os.path.join(REPO, "docs", "ARCHITECTURE.md")) as f:
        text = f.read()
    tok = Tokenizer.from_file(fetch.fetch("tokenizer.json"))
    tokens = tok.encode(text, add_special_tokens=False).ids[: P + D]
    assert len(tokens) == P + D, len(tokens)
    with safe_open(fetch.fetch(EMBED_SHARD), "pt") as f:
        emb = f.get_slice("embed.weight")
        e = torch.stack([emb[t : t + 1][0] for t in tokens]).to(torch.bfloat16)  # [T, dim]
    x = e.unsqueeze(1).repeat(1, args.hc_mult, 1).unsqueeze(0)  # [1, T, hc, dim]
    mix = m.make_identity_pre_mix(x, args.hc_mult)

    out = {"tokens": torch.tensor(tokens), "input": x[0].clone()}
    sa = m.shared_attn
    attn = block.attn
    is_cand = a.layer == args.candidate_source_layer

    trace = {"attn": [], "ffn": [], "experts": [], "expert_weights": [], "attn_in": [], "ffn_in": []}
    block.attn.register_forward_pre_hook(lambda mod, inp: trace["attn_in"].append(inp[0][0].clone()))
    block.ffn.register_forward_pre_hook(lambda mod, inp: trace["ffn_in"].append(inp[0][0].clone()))
    block.attn.register_forward_hook(lambda mod, inp, o: trace["attn"].append(o[0].clone()))
    block.ffn.register_forward_hook(lambda mod, inp, o: trace["ffn"].append(o[0].clone()))

    def gate_hook(mod, inp, o):
        trace["expert_weights"].append(o[0].clone())
        trace["experts"].append(o[1].clone())

    block.ffn.gate.register_forward_hook(gate_hook)
    for k in ("qr", "q", "kv", "o", "low"):
        trace[k] = []
    attn.q_norm.register_forward_hook(lambda mod, inp, o: trace["qr"].append(o[0].clone()))
    attn.wo_b.register_forward_pre_hook(lambda mod, inp: trace["low"].append(inp[0][0].clone()))
    real_sparse = m.sparse_attn

    def sparse_hook(q, kv, sink, idx, scale):
        trace["q"].append(q[0].flatten(-2).clone())
        o = real_sparse(q, kv, sink, idx, scale)
        trace["o"].append(o[0].flatten(-2).clone())
        return o

    m.sparse_attn = sparse_hook
    real_window = attn._window_kv

    def window_hook(x, freqs_cis, start_pos):
        kv, idxs = real_window(x, freqs_cis, start_pos)
        trace["kv"].append((kv[0] if start_pos == 0 else kv[0, start_pos % args.window_size : start_pos % args.window_size + 1]).clone())
        return kv, idxs

    attn._window_kv = window_hook

    with torch.inference_mode():
        h, pre = block(x[:, :P].clone(), 0, mix[:, :P], None)
        outs, pres = [h[0]], [pre[0]]
        if attn.is_index_source:
            out["topk.prefill"] = sa.topk_idxs[0].clone()
        if is_cand:
            out["candidates.prefill"] = sa.candidates[0].clone()
        for i in range(D):
            pos = P + i
            h, pre = block(x[:, pos : pos + 1].clone(), pos, mix[:, pos : pos + 1], None)
            outs.append(h[0])
            pres.append(pre[0])
            if attn.is_index_source:
                out[f"topk.decode{i}"] = sa.topk_idxs[0].clone()
            if is_cand:
                out[f"candidates.decode{i}"] = sa.candidates[0].clone()
    out["trace.freqs"] = torch.view_as_real(attn.freqs_cis[: P + D]).contiguous()
    for k, v in trace.items():
        out["trace." + k] = torch.cat(v)
    out["out"] = torch.cat(outs)
    out["pre_mix"] = torch.cat(pres)
    if attn.is_kv_source:
        n = (P + D) // attn.compress_ratio
        out["cache.compress_kv"] = attn.compress_kv_cache[0, :n].clone()
        out["cache.index_k"] = attn.indexer.k_cache[0, :n].clone()
    save_file({k: v.contiguous() for k, v in out.items()}, a.out, metadata={"layer": str(a.layer), "prompt": str(P), "decode": str(D)})
    print(f"layer {a.layer}: {P} prompt + {D} decode positions -> {a.out}")


if __name__ == "__main__":
    main()
