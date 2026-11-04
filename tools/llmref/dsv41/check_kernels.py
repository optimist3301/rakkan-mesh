#!/usr/bin/env python3
"""On a CUDA machine: how closely kernel_cpu reproduces the real kernels.

1. Kernels: each tilelang kernel of upstream/inference/kernel.py against its
   kernel_cpu counterpart, on the real model's shapes and the same inputs.
   Quantization should agree bit for bit; GEMMs, attention and Sinkhorn are
   reported as the share of bit-identical outputs and the largest error.
2. Model: a fixture made by tiny.py, run on the GPU with the real kernels
   and compared with the golden dump it carries, made on the CPU.

usage: check_kernels.py [--fixture DIR] [--out REPORT.json] [--dry-run]
  --dry-run  no GPU: kernel_cpu stands in for the real kernels, on the CPU.
             Tests this script; every comparison must come out identical.
  --fixture  default: the committed tiny fixture. tilelang may reject its
             small widths; then make one with `tiny.py --medium DIR`.

On the GPU machine (Hopper, like the reference), gpu_server.sh sets up
PyTorch with CUDA and tilelang and runs this on both fixtures; by hand:
  python -m venv .venv && .venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.15.0
  .venv/bin/pip install -r requirements-cpu.txt tilelang
  .venv/bin/python tiny.py --medium /tmp/medium
  .venv/bin/python check_kernels.py --fixture /tmp/medium --out report.json
Send report.json back: it decides whether kernel_cpu needs changes.
"""
import argparse
import json
import os
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURE = os.path.join(HERE, "..", "..", "..", "internal", "llm", "testdata", "dsv41-tiny")

import kernel_cpu as C  # noqa: E402
import run  # noqa: E402


def bits(t):
    t = t.detach().cpu().contiguous()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2):
        return t.view(torch.uint8)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def compare(gpu, cpu):
    """Share of bit-identical elements and the largest absolute and relative
    differences, as plain numbers."""
    g, c = gpu.detach().cpu(), cpu.detach().cpu()
    same = (bits(g) == bits(c)).float().mean().item()
    r = {"identical": same, "shape": list(c.shape)}
    if g.is_floating_point() and g.dtype not in (torch.float8_e8m0fnu, torch.float4_e2m1fn_x2):
        gf, cf = g.float(), c.float()
        d = (gf - cf).abs()
        finite = torch.isfinite(cf)
        r["max_abs"] = d[finite].max().item() if finite.any() else 0.0
        r["max_rel"] = (d / cf.abs().clamp_min(1e-6))[finite].max().item() if finite.any() else 0.0
        r["nonfinite_mismatch"] = int((torch.isfinite(gf) != finite).sum())
    return r


def kernel_checks(K, device):
    """K: the real kernel module. Inputs are made on the CPU, seeded."""
    torch.manual_seed(1)
    out = {}
    cuda = lambda t: t.to(device)  # noqa: E731

    x = (torch.randn(64, 5120) * torch.logspace(-4, 3, 5120)).to(torch.bfloat16)
    qg, sg = K.act_quant(cuda(x), 32, "ue8m0", torch.float8_e8m0fnu)
    qc, sc = C.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    out["act_quant.values"], out["act_quant.scales"] = compare(qg, qc), compare(sg, sc)
    yg, yc = cuda(x.clone()), x.clone()
    K.act_quant(yg, 32, "ue8m0", torch.float8_e8m0fnu, True)
    C.act_quant(yc, 32, "ue8m0", torch.float8_e8m0fnu, True)
    out["act_quant.inplace"] = compare(yg, yc)

    k = (torch.randn(64, 512) * 3).to(torch.bfloat16)
    k[0, :16] = 0  # an all-zero group: the scale floors
    for name, sdt, blk in (("fp4_quant.e8m0_32", torch.float8_e8m0fnu, 32), ("fp4_quant.e4m3_16", torch.float8_e4m3fn, 16)):
        pg, sg = K.fp4_act_quant(cuda(k), blk, False, sdt)
        pc, sc = C.fp4_act_quant(k, blk, False, sdt)
        out[name + ".values"], out[name + ".scales"] = compare(pg, pc), compare(sg, sc)
        yg, yc = cuda(k.clone()), k.clone()
        K.fp4_act_quant(yg, blk, True, sdt)
        C.fp4_act_quant(yc, blk, True, sdt)
        out[name + ".inplace"] = compare(yg, yc)

    from tiny import quant_fp8_blocks, quant_fp4_rows

    torch.set_default_dtype(torch.bfloat16)
    try:
        for name, (M, N, Kd) in {"fp8_gemm.wq_a": (8, 1280, 5120), "fp8_gemm.wo_b": (8, 5120, 8192), "fp8_gemm.prefill": (100, 1280, 5120)}.items():
            a = torch.randn(M, Kd).to(torch.bfloat16)
            qa, sa = C.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
            qw, sw = quant_fp8_blocks(torch.randn(N, Kd, dtype=torch.float32) * Kd**-0.5)
            out[name] = compare(
                K.fp8_gemm(cuda(qa), cuda(sa), cuda(qw), cuda(sw), torch.float8_e8m0fnu, 32),
                C.fp8_gemm(qa, sa, qw, sw, torch.float8_e8m0fnu, 32),
            )
        for name, (M, N, Kd) in {"fp4_gemm.w1": (8, 2304, 5120), "fp4_gemm.w2": (8, 5120, 2304)}.items():
            a = torch.randn(M, Kd).to(torch.bfloat16)
            qa, sa = C.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
            pw, sw = quant_fp4_rows(torch.randn(N, Kd, dtype=torch.float32) * Kd**-0.5)
            pw = pw.view(torch.float4_e2m1fn_x2)
            out[name] = compare(
                K.fp4_gemm(cuda(qa), cuda(sa), cuda(pw), cuda(sw), torch.float8_e8m0fnu, act_block_size=32),
                C.fp4_gemm(qa, sa, pw, sw, torch.float8_e8m0fnu, act_block_size=32),
            )

        b, m, h, d, n, topk = 1, 4, 64, 512, 2048, 640
        q = torch.randn(b, m, h, d).to(torch.bfloat16)
        kv = torch.randn(b, n, d).to(torch.bfloat16)
        sink = torch.randn(h)
        idx = torch.stack([torch.randperm(n)[:topk] for _ in range(b * m)]).view(b, m, topk).int()
        idx[0, 1, 100:] = -1
        idx[0, 2, :] = -1
        out["sparse_attn"] = compare(
            K.sparse_attn(cuda(q), cuda(kv), cuda(sink), cuda(idx), d**-0.5),
            C.sparse_attn(q, kv, sink, idx, d**-0.5),
        )
    finally:
        torch.set_default_dtype(torch.float32)

    mixes = torch.randn(1, 64, 24)
    scale, base = torch.rand(3) + 0.5, torch.randn(24)
    g = K.hc_split_sinkhorn(cuda(mixes), cuda(scale), cuda(base), 4, 20, 1e-6)
    c = C.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    for name, gg, cc in zip(("pre", "post", "comb"), g, c):
        out["hc_split_sinkhorn." + name] = compare(gg, cc)
    return out


def model_check(fixture, device, dry_run):
    """Run the fixture's tokens on the GPU with the real kernels and compare
    every recorded tensor with the fixture's golden dump."""
    from safetensors.torch import load_file

    import tiny

    m = run.setup(cuda=not dry_run)
    model, args = tiny.load(m, fixture, device=device)
    torch.set_default_device(device)
    try:
        got = tiny.dump(model, m, args, tiny.fixture_tokens())
    finally:
        torch.set_default_device("cpu")
    want = load_file(os.path.join(fixture, "golden.safetensors"))
    per = {k: compare(got[k], want[k]) for k in sorted(want) if k in got}
    steps = sorted({k.split(".")[0] for k in want if k.endswith(".logits") and "spec" not in k})
    argmax = {s: bool(got[f"{s}.logits"].cpu().argmax(-1).eq(want[f"{s}.logits"].argmax(-1)).all()) for s in steps}
    idx = {k: v["identical"] for k, v in per.items() if k.endswith(("topk_idxs", "candidates"))}
    return {
        "missing": sorted(set(want) - set(got)),
        "argmax_agrees": argmax,
        "index_agreement": idx,
        "max_logit_abs_diff": max(per[f"{s}.logits"]["max_abs"] for s in steps),
        "tensors": per,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--out", default="check_kernels_report.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    device = "cpu" if a.dry_run else "cuda"
    if not a.dry_run and not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU (or --dry-run)")
    report = {"device": "dry run" if a.dry_run else torch.cuda.get_device_name(), "torch": torch.__version__}
    try:
        import tilelang

        report["tilelang"] = tilelang.__version__
    except Exception:  # noqa: BLE001
        pass

    run.setup(cuda=not a.dry_run)
    import kernel as K  # the real one, unless --dry-run made it kernel_cpu

    for name, fn in (("kernels", lambda: kernel_checks(K, device)), ("model", lambda: model_check(os.path.abspath(a.fixture), device, a.dry_run))):
        try:
            report[name] = fn()
        except Exception:  # noqa: BLE001 - report and go on
            report[name] = {"error": traceback.format_exc()}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"device {report['device']}, torch {report['torch']}, tilelang {report.get('tilelang')}")
    ks = report["kernels"]
    if "error" in ks:
        print("kernels: ERROR\n" + ks["error"])
    else:
        for name, r in ks.items():
            extra = f"  max_abs {r['max_abs']:.3g}  max_rel {r['max_rel']:.3g}" if "max_abs" in r else ""
            print(f"  {name:28s} identical {r['identical']:.4f}{extra}")
    ms = report["model"]
    if "error" in ms:
        print("model: ERROR\n" + ms["error"])
    else:
        print(f"model: argmax agrees on {sum(ms['argmax_agrees'].values())}/{len(ms['argmax_agrees'])} steps, max |dlogit| {ms['max_logit_abs_diff']:.3g}")
        worst = sorted(ms["index_agreement"].items(), key=lambda kv: kv[1])[:5]
        print("  least agreeing index tensors:", ", ".join(f"{k} {v:.3f}" for k, v in worst))
    print(f"full report: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
