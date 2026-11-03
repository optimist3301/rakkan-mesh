"""CPU stand-ins for the tilelang kernels of upstream/inference/kernel.py.

The reference model (upstream/inference/model.py) imports six functions from
`kernel`, written in tilelang for CUDA. These implement the same functions
with plain PyTorch on the CPU, so the reference runs anywhere; run.py installs
this module as `kernel` before importing the model.

What is reproduced exactly, and what is not:

- Quantization (act_quant, fp4_act_quant) follows the kernels operation by
  operation: block absmax in float32, the same lower clamps, power-of-two
  scales computed with the same bit manipulation, E4M3 scales for compressed
  KV, saturation, round-to-nearest-even into FP8 E4M3 and FP4 E2M1. Results
  should match the GPU bit for bit; check_kernels.py verifies that.
- GEMMs (fp8_gemm, fp4_gemm) take the per-32 block dot products exactly (in
  float64, where they are exact), round each to float32, apply the
  activation and weight scales, and accumulate the blocks in order in
  float32, as the kernels do. The GPU's tensor cores sum within a block in
  their own, not exactly specified, way: expect small differences there.
- sparse_attn runs the kernel's online softmax over blocks of 64 indices,
  including its rounding of probabilities to bfloat16 before they multiply
  the values and its attention sink; exp and the reductions are PyTorch's.
- hc_split_sinkhorn follows the kernel's sequence of operations in float32.

Nothing here is tuned for speed: it serves tests and reference dumps.
"""

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
BLOCK = 32

# E2M1 magnitudes by 3-bit code, and the midpoints between neighbours.
_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MID = (_FP4_GRID[1:] + _FP4_GRID[:-1]) / 2
# Rounding a tie at midpoint i goes up when code i + 1 is even.
_FP4_TIE_UP = torch.tensor([(i + 1) % 2 == 0 for i in range(7)])


def _f32(x):
    return torch.tensor(x, dtype=torch.float32)


def _log2_ceil_pow2(t):
    """2^ceil(log2(t)) for positive normal float32 t, by the kernels' bit
    manipulation: the exponent field, plus one if any mantissa bit is set."""
    bits = t.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).int()
    return ((e + 127) << 23).view(torch.float32)


def e8m0_to_f32(s):
    """Decode float8_e8m0fnu (or pass float32 through)."""
    if s.dtype == torch.float8_e8m0fnu:
        e = s.view(torch.uint8).int()
        # The biased exponent sits where float32 wants it; byte 0 is 2^-127,
        # a float32 subnormal, and byte 255 is NaN.
        f = (e << 23).view(torch.float32)
        f = torch.where(e == 0, torch.tensor(2.0**-127, dtype=torch.float32), f)
        return torch.where(e == 255, torch.tensor(float("nan")), f)
    return s.float()


def f32_to_e8m0(s):
    """Encode power-of-two float32 scales as float8_e8m0fnu."""
    bits = s.contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8).view(torch.float8_e8m0fnu)


def round_fp4(v):
    """Round float32 values in [-6, 6] to the nearest E2M1 value, ties to
    even code, keeping the sign (so -0 can result)."""
    a = v.abs()
    code = (a.unsqueeze(-1) > _FP4_MID).sum(-1)
    code = code + ((a.unsqueeze(-1) == _FP4_MID) & _FP4_TIE_UP).sum(-1)
    return torch.copysign(_FP4_GRID[code], v)


def fp4_codes(v):
    """4-bit E2M1 codes (sign in bit 3) of values already on the grid."""
    a = v.abs()
    code = (a.unsqueeze(-1) == _FP4_GRID).int().argmax(-1)
    return code | (torch.signbit(v).int() << 3)


def decode_fp4_packed(b):
    """[N, K/2] float4_e2m1fn_x2 (or uint8/int8) -> [N, K] float32; the low
    nibble holds the even element."""
    u = b.view(torch.uint8).int()
    lo, hi = u & 0xF, (u >> 4) & 0xF
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)
    mag = _FP4_GRID[codes & 7]
    return torch.where((codes & 8) != 0, -mag, mag)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 E4M3 quantization over the last dimension.

    inplace=True writes the dequantized values back into x (in x's dtype)."""
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1).clamp_min(_f32(1e-4))
    if scale_fmt is not None:
        s = _log2_ceil_pow2(amax * _f32(1 / FP8_MAX))
    else:
        s = amax * _f32(1 / FP8_MAX)
    q = (xf / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        y = (q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    q = q.flatten(-2)
    s = f32_to_e8m0(s) if scale_dtype == torch.float8_e8m0fnu else s
    return q, s


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise FP4 E2M1 quantization with E8M0 (power of two) or E4M3
    scales. inplace=True writes the dequantized values back into x."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        # Compressed KV: an all-zero group keeps a nonzero scale.
        amax = amax.clamp_min(_f32(6 * 2**-9))
        s = (amax / _f32(FP4_MAX)).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(_f32(6 * 2**-126))
        s = _log2_ceil_pow2(amax * _f32(1.0 / FP4_MAX))
    q = round_fp4((xf / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        y = (q * s.unsqueeze(-1)).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    codes = fp4_codes(q.flatten(-2))
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8).view(torch.float4_e2m1fn_x2)
    s = s.to(torch.float8_e4m3fn) if scale_dtype == torch.float8_e4m3fn else f32_to_e8m0(s)
    return packed, s


def _block_gemm(a, a_s, w, w_scale, act_block, out_dtype):
    """C = A @ W^T with A [M, K] float32 values of FP8 activations, a_s [M,
    K/act_block] float32, W [N, K] float32 weight values and w_scale(k) the
    float32 weight scales of K block k, broadcastable to [1, N]."""
    M, K = a.shape
    N = w.size(0)
    KB = K // BLOCK
    acc = torch.zeros(M, N, dtype=torch.float32)
    wb = w.double().view(N, KB, BLOCK)
    for m0 in range(0, M, 16):  # bound the float64 intermediate
        ab = a[m0 : m0 + 16].double().view(-1, KB, BLOCK)
        partial = torch.einsum("mkj,nkj->kmn", ab, wb).float()  # exact, then rounded once
        part = torch.zeros(ab.size(0), N, dtype=torch.float32)
        for k in range(KB):
            sa = a_s[m0 : m0 + 16, (k * BLOCK) // act_block].unsqueeze(1)
            part = part + partial[k] * sa * w_scale(k)
        acc[m0 : m0 + 16] = part
    return acc.to(out_dtype)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M,N] = A[M,K] @ B[N,K]^T with FP8 operands, per-row-and-K-block
    activation scales and per-(N block, K block) weight scales."""
    assert block_size == BLOCK, "the model uses 32x32 blocks"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    A = a.reshape(M, K).float()
    As = e8m0_to_f32(a_s).reshape(M, -1)
    Bs = e8m0_to_f32(b_s)
    row_block = torch.arange(N) // block_size
    c = _block_gemm(A, As, b.float(), lambda k: Bs[row_block, k].unsqueeze(0), block_size, torch.get_default_dtype())
    return c.view(*a.shape[:-1], N)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T; B packed two values per byte
    along K, with one E8M0 scale per 32 values of a row."""
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    A = a.reshape(M, K).float()
    As = e8m0_to_f32(a_s).reshape(M, -1)
    W = decode_fp4_packed(b)
    Bs = e8m0_to_f32(b_s)
    c = _block_gemm(A, As, W, lambda k: Bs[:, k].unsqueeze(0), act_block_size, torch.get_default_dtype())
    return c.view(*a.shape[:-1], N)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Attention of each query [b, m, h, d] over the kv rows its topk_idxs
    [b, m, topk] select (-1: none), with one shared key/value row per
    position, a per-head attention sink in the denominator, and the kernel's
    online softmax over blocks of 64 indices."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    scale = _f32(softmax_scale)
    qd = q.double()
    acc_o = torch.zeros(b, m, h, d, dtype=torch.float32)
    sum_exp = torch.zeros(b, m, h, dtype=torch.float32)
    run_max = torch.full((b, m, h), -1e30, dtype=torch.float32)
    bidx = torch.arange(b).view(b, 1, 1)
    for t in range(0, topk, 64):
        idx = topk_idxs[:, :, t : t + 64].long()
        valid = idx != -1
        rows = kv[bidx, idx.clamp_min(0)]  # [b, m, blk, d]
        rows = torch.where(valid.unsqueeze(-1), rows, torch.zeros((), dtype=kv.dtype))
        s = torch.einsum("bmhd,bmjd->bmhj", qd, rows.double()).float()
        s = torch.where(valid.unsqueeze(2), s, _f32(-float("inf")))
        s = s * scale
        prev = run_max
        run_max = torch.maximum(prev, s.amax(-1))
        rescale = torch.exp(prev - run_max)
        p = torch.exp(s - run_max.unsqueeze(-1))
        sum_exp = sum_exp * rescale + p.sum(-1)
        pv = torch.einsum("bmhj,bmjd->bmhd", p.to(torch.bfloat16).double(), rows.double()).float()
        acc_o = acc_o * rescale.unsqueeze(-1) + pv
    sum_exp = sum_exp + torch.exp(attn_sink.float().view(1, 1, h) - run_max)
    return (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Split the hyper-connection mixing logits into pre [.., hc], post [..,
    hc] and comb [.., hc, hc], comb made doubly stochastic by Sinkhorn."""
    hc = hc_mult
    eps = _f32(eps)
    m = mixes.float()
    sc, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * sc[1] + base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * sc[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
