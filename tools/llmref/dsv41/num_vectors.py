#!/usr/bin/env python3
"""Test vectors for internal/num's DeepSeek-V4.1 arithmetic: the answers of
kernel_cpu (the reference's kernels) to inputs chosen to exercise rounding.

Every case gives inputs and expected outputs as bit patterns, base64 of
little-endian arrays: bf16 values as uint16, float32 as uint32, FP8/E8M0
bytes, packed FP4 bytes.

  round_e4m3   float32 -> E4M3 code (torch's cast): random values, every
               midpoint between neighbours, subnormals, the extremes
  act_quant    bf16 rows -> E4M3 codes, E8M0 scales per 32, and the
               in-place result (dequantized, bf16)
  fp4_quant    bf16 rows -> packed E2M1, scales (E8M0 per 32, or E4M3 per
               16), in-place result
  fp8_gemm     bf16 activation rows x FP8 weights (32x32 E8M0 scales) ->
               bf16 outputs
  fp4_gemm     bf16 activation rows x packed FP4 weights (E8M0 per 32 of
               a row) -> bf16 outputs
               Both GEMMs also give y32, the float32 sum before rounding to
               bf16: the order of adding blocks shows there, and only
               rarely after the rounding.

usage: num_vectors.py OUT.json.gz
"""
import base64
import gzip
import json
import sys

import torch

import kernel_cpu as K
from tiny import quant_fp4_rows, quant_fp8_blocks

torch.manual_seed(20261104)


def b64(t):
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    elif t.dtype == torch.float32:
        t = t.view(torch.int32)
    elif t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2, torch.int8):
        t = t.view(torch.uint8)
    return base64.b64encode(t.numpy().tobytes()).decode()


def e4m3_values():
    """Every finite E4M3 magnitude, ascending."""
    v = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    v = v[torch.isfinite(v) & (v >= 0)]
    return torch.unique(v)


def rows_with_ties(n_rows, width):
    """bf16 rows whose blocks of 32 reach 448 (so the scale is 1) and hold
    E4M3 midpoints, positive and negative, among random values."""
    vals = e4m3_values()
    mids = ((vals[1:] + vals[:-1]) / 2).to(torch.bfloat16)
    mids = mids[mids.float() == (vals[1:] + vals[:-1]) / 2]  # those bf16 holds exactly
    x = (torch.randn(n_rows, width) * 50).clamp(-447, 447)
    for r in range(n_rows):
        for b in range(0, width, 32):
            x[r, b] = 448.0 if (r + b) % 2 else -448.0
            pick = mids[torch.randint(len(mids), (24,))]
            sign = torch.where(torch.rand(24) < 0.5, -1.0, 1.0)
            x[r, b + 1 : b + 25] = pick.float() * sign
    return x.to(torch.bfloat16)


def act_rows(n_rows, width):
    """bf16 rows with blocks of widely different magnitudes, a zero block and
    a block below the 1e-4 floor."""
    x = torch.randn(n_rows, width)
    scale = torch.exp2(torch.randint(-30, 12, (n_rows, width // 32)).float())
    x = x * scale.repeat_interleave(32, 1)
    x[0, :32] = 0
    if width >= 64:
        x[0, 32:64] = torch.randn(32) * 1e-6
    return x.to(torch.bfloat16)


def case_round_e4m3():
    vals = e4m3_values()
    mids = (vals[1:] + vals[:-1]) / 2
    x = torch.cat([vals, mids, torch.rand(4000) * 448, torch.rand(2000) * 2**-6, torch.exp2(torch.rand(2000) * 26 - 17)])
    x = x[x <= 448]
    x = torch.cat([x, -x]).float()
    return {"in": b64(x), "out": b64(x.to(torch.float8_e4m3fn))}


def case_act_quant():
    out = []
    for x in (act_rows(3, 256), rows_with_ties(2, 128), act_rows(1, 5120)):
        q, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        y = x.clone()
        K.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, True)
        out.append({"rows": x.size(0), "cols": x.size(1), "x": b64(x), "q": b64(q), "scale": b64(s), "inplace": b64(y)})
    return out


def case_fp4_quant():
    out = []
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mids = (grid[1:] + grid[:-1]) / 2
    for scale_dtype, block in ((torch.float8_e8m0fnu, 32), (torch.float8_e4m3fn, 16)):
        x = act_rows(3, 512).float()
        x[1, :block] = 0  # zero group: the scale floors
        # a group of exact midpoints with scale 1: amax 6 gives s = 1 for both scale kinds
        x[2, :block] = mids[torch.randint(7, (block,))] * torch.where(torch.rand(block) < 0.5, -1.0, 1.0)
        x[2, 0] = 6.0
        x = x.to(torch.bfloat16)
        packed, s = K.fp4_act_quant(x, block, False, scale_dtype)
        y = x.clone()
        K.fp4_act_quant(y, block, True, scale_dtype)
        out.append(
            {
                "block": block,
                "scale_kind": "e8m0" if scale_dtype == torch.float8_e8m0fnu else "e4m3",
                "rows": x.size(0),
                "cols": x.size(1),
                "x": b64(x),
                "packed": b64(packed),
                "scale": b64(s),
                "inplace": b64(y),
            }
        )
    return out


def weights(n, k, std, spread=True):
    """float32 weights; with spread, their 32x32 blocks differ widely in
    scale. Without, blocks are alike, as in a trained model: then the order
    in which the blocks' terms are added decides the low bits."""
    w = torch.randn(n, k, dtype=torch.float32) * std
    if spread:
        w = w * torch.exp2(torch.randint(-12, 6, ((n + 31) // 32, k // 32)).float()).repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)
    return w


def plain_rows(n_rows, width):
    return torch.randn(n_rows, width).to(torch.bfloat16)


def case_fp8_gemm():
    out = []
    torch.set_default_dtype(torch.bfloat16)
    try:
        for M, N, Kd, spread in ((3, 96, 256, True), (2, 40, 128, True), (1, 32, 2048, True), (2, 256, 5120, False)):
            x = act_rows(M, Kd) if spread else plain_rows(M, Kd)
            npad = (N + 31) // 32 * 32
            w = weights(npad, Kd, Kd**-0.5, spread)
            qw, sw = quant_fp8_blocks(w)
            qw = qw[:N]  # rows beyond N do not exist; their scale block row is shared
            qa, sa = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
            y = K.fp8_gemm(qa, sa, qw.contiguous(), sw, torch.float8_e8m0fnu, 32)
            torch.set_default_dtype(torch.float32)  # the output dtype: the sum unrounded
            y32 = K.fp8_gemm(qa, sa, qw.contiguous(), sw, torch.float8_e8m0fnu, 32)
            torch.set_default_dtype(torch.bfloat16)
            out.append({"m": M, "n": N, "k": Kd, "x": b64(x), "w": b64(qw), "w_scale": b64(sw), "y": b64(y), "y32": b64(y32)})
    finally:
        torch.set_default_dtype(torch.float32)
    return out


def case_fp4_gemm():
    out = []
    torch.set_default_dtype(torch.bfloat16)
    try:
        for M, N, Kd, spread in ((3, 96, 256, True), (1, 64, 2304, True), (2, 33, 64, True), (2, 256, 5120, False)):
            x = act_rows(M, Kd) if spread else plain_rows(M, Kd)
            w = torch.randn(N, Kd, dtype=torch.float32) * Kd**-0.5
            if spread:
                w = w * torch.exp2(torch.randint(-8, 4, (N, Kd // 32)).float()).repeat_interleave(32, 1)
            pw, sw = quant_fp4_rows(w)
            pw = pw.view(torch.float4_e2m1fn_x2)
            qa, sa = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
            y = K.fp4_gemm(qa, sa, pw, sw, torch.float8_e8m0fnu, act_block_size=32)
            torch.set_default_dtype(torch.float32)
            y32 = K.fp4_gemm(qa, sa, pw, sw, torch.float8_e8m0fnu, act_block_size=32)
            torch.set_default_dtype(torch.bfloat16)
            out.append({"m": M, "n": N, "k": Kd, "x": b64(x), "w": b64(pw), "w_scale": b64(sw), "y": b64(y), "y32": b64(y32)})
    finally:
        torch.set_default_dtype(torch.float32)
    return out


def main():
    vectors = {
        "round_e4m3": case_round_e4m3(),
        "act_quant": case_act_quant(),
        "fp4_quant": case_fp4_quant(),
        "fp8_gemm": case_fp8_gemm(),
        "fp4_gemm": case_fp4_gemm(),
    }
    with gzip.open(sys.argv[1], "wt") as f:
        json.dump(vectors, f)
    print("written", sys.argv[1])


if __name__ == "__main__":
    main()
