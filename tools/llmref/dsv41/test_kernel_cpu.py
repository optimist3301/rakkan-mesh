#!/usr/bin/env python3
"""Checks kernel_cpu against direct definitions of what each kernel computes.

usage: test_kernel_cpu.py   (exits non-zero on failure)
"""
import itertools
import sys

import torch

import kernel_cpu as K

torch.manual_seed(0)
failures = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def test_round_fp4():
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # every value on a fine lattice, midpoints included
    v = torch.arange(-6 * 64, 6 * 64 + 1, dtype=torch.float32) / 64
    got = K.round_fp4(v)
    want = []
    for x in v.tolist():
        a = abs(x)
        best = min(range(8), key=lambda i: (abs(grid[i] - a), i % 2))  # nearest, ties to even code
        want.append(grid[best] if x >= 0 else -grid[best])
    want = torch.tensor(want)
    check("round_fp4 nearest, ties to even", torch.equal(got.abs(), want.abs()) and torch.equal(torch.signbit(got), torch.signbit(v)))


def test_act_quant():
    x = (torch.randn(7, 256) * torch.logspace(-3, 2, 256)).to(torch.bfloat16)
    q, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    sf = K.e8m0_to_f32(s)
    check("act_quant scales are powers of two", torch.equal(sf, torch.exp2(torch.log2(sf).round())))
    deq = (q.float().view(7, 8, 32) * sf.unsqueeze(-1)).flatten(1)
    y = x.clone()
    K.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, True)
    check("act_quant inplace equals quantize then dequantize", torch.equal(y.float(), deq.to(torch.bfloat16).float()))
    amax = x.float().view(7, 8, 32).abs().amax(-1)
    check("act_quant uses the smallest power of two that fits", bool(((amax / sf <= 448) & (amax / sf > 224)).all()))
    err = (deq - x.float()).abs() / x.float().abs().clamp_min(1e-30)
    big = x.float().abs() > sf.repeat_interleave(32, 1) * 2**-6  # normal FP8 range
    check("act_quant relative error within half an FP8 ulp", bool((err[big] <= 2**-4 + 1e-7).all()), f"max {err[big].max()}")


def test_fp4_act_quant():
    x = (torch.randn(5, 128) * 3).to(torch.bfloat16)
    for sdt, blk in ((torch.float8_e8m0fnu, 32), (torch.float8_e4m3fn, 16)):
        packed, s = K.fp4_act_quant(x, blk, False, sdt)
        y = x.clone()
        K.fp4_act_quant(y, blk, True, sdt)
        deq = K.decode_fp4_packed(packed).view(5, -1, blk) * s.float().unsqueeze(-1)
        check(f"fp4_act_quant {sdt} packed equals inplace", torch.equal(deq.flatten(1).to(torch.bfloat16), y))
        check(f"fp4_act_quant {sdt} stays on the grid", bool((y.float().view(5, -1, blk).abs() <= 6 * s.float().unsqueeze(-1)).all()))


def quant_weight_fp8(w):
    """Per 32x32 block, as the checkpoint stores weights."""
    w = w.float()
    N, Kd = w.shape
    blocks = w.view(N // 32, 32, Kd // 32, 32)
    amax = blocks.abs().amax((1, 3)).clamp_min(1e-4)
    s = K._log2_ceil_pow2(amax / 448)
    q = (blocks / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(N, Kd), K.f32_to_e8m0(s)


def test_gemms():
    torch.set_default_dtype(torch.bfloat16)
    try:
        M, N, Kd = 5, 96, 256
        x = torch.randn(M, Kd).to(torch.bfloat16)
        w = torch.randn(N, Kd, dtype=torch.float32) / 16
        qw, sw = quant_weight_fp8(w)
        qa, sa = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        got = K.fp8_gemm(qa, sa, qw, sw, torch.float8_e8m0fnu, 32).float()
        A = (qa.float().view(M, 8, 32) * K.e8m0_to_f32(sa).unsqueeze(-1)).flatten(1).double()
        W = (qw.float().view(N // 32, 32, 8, 32) * K.e8m0_to_f32(sw)[:, None, :, None]).view(N, Kd).double()
        want = (A @ W.T).float()
        check("fp8_gemm equals the product of dequantized operands", torch.allclose(got, want, rtol=2**-7, atol=1e-6))

        packed, s4 = K.fp4_act_quant(torch.randn(N, Kd).to(torch.bfloat16), 32, False)
        got = K.fp4_gemm(qa, sa, packed, s4, torch.float8_e8m0fnu, 32).float()
        W4 = (K.decode_fp4_packed(packed).view(N, 8, 32) * K.e8m0_to_f32(s4).unsqueeze(-1)).flatten(1).double()
        want = (A @ W4.T).float()
        check("fp4_gemm equals the product of dequantized operands", torch.allclose(got, want, rtol=2**-7, atol=1e-6))
    finally:
        torch.set_default_dtype(torch.float32)


def test_sparse_attn():
    b, m, h, d, n, topk = 2, 3, 4, 64, 150, 130
    q = torch.randn(b, m, h, d).to(torch.bfloat16)
    kv = torch.randn(b, n, d).to(torch.bfloat16)
    sink = torch.randn(h)
    idx = torch.stack([torch.randperm(n)[:topk] for _ in range(b * m)]).view(b, m, topk).int()
    idx[0, 0, 5:] = -1  # few valid
    idx[1, 2, :] = -1  # none valid
    got = K.sparse_attn(q, kv, sink, idx, d**-0.5).float()
    want = torch.zeros(b, m, h, d)
    for i, j in itertools.product(range(b), range(m)):
        sel = idx[i, j][idx[i, j] >= 0].long()
        if sel.numel() == 0:
            continue
        rows = kv[i, sel].double()
        s = q[i, j].double() @ rows.T * d**-0.5
        mx = s.amax(-1, keepdim=True)
        p = torch.exp(s - mx)
        denom = p.sum(-1, keepdim=True) + torch.exp(sink.double().unsqueeze(-1) - mx)
        want[i, j] = (p @ rows / denom).float()
    check("sparse_attn matches dense attention with a sink", torch.allclose(got, want, atol=2e-2, rtol=2e-2), f"max err {(got - want).abs().max()}")
    check("sparse_attn gives zeros without valid indices", bool((got[1, 2] == 0).all()))


def test_sinkhorn():
    mixes = torch.randn(3, 5, 24)
    pre, post, comb = K.hc_split_sinkhorn(mixes, torch.rand(3) + 0.5, torch.randn(24), 4, 20, 1e-6)
    check("sinkhorn comb columns sum to one", torch.allclose(comb.sum(-2), torch.ones(3, 5, 4), atol=1e-5))
    check("sinkhorn comb rows sum to one", torch.allclose(comb.sum(-1), torch.ones(3, 5, 4), atol=1e-3))
    check("pre in (0, 1], post in (0, 2)", bool(((pre > 0) & (pre <= 1 + 1e-6)).all() and ((post > 0) & (post < 2)).all()))


for t in (test_round_fp4, test_act_quant, test_fp4_act_quant, test_gemms, test_sparse_attn, test_sinkhorn):
    t()
sys.exit(1 if failures else 0)
