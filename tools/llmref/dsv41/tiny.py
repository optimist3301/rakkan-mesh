#!/usr/bin/env python3
"""Build a tiny DeepSeek-V4.1 with random weights and dump the reference's
outputs for it: the test fixture of the Go engine.

The tiny model keeps every architectural feature of the real one, scaled
down: an encoder of SWA and ratio-2 CSA2 layers (one Full, two Reuse), a
decoder of ratio-1 layers with a candidate pool and a Reindex layer, two
Engram modules, hyper-connections, FP8/FP4 weights, MoE with a shared
expert, and a DSpark stage. Its vocabulary is the real tokenizer's first
512 ids, so the Engram token map is the real one's restriction.

Weights are written as the published checkpoint stores them (names,
dtypes, FP8 32x32 block scales, packed FP4 experts, FP8 wo_a), converted
with the unmodified upstream convert.py, and loaded as generate.py loads
them. The Go importer therefore reads exactly the real format.

Outputs, in OUT:
  model.safetensors, model.safetensors.index.json   checkpoint, HF format
  config.json                                       HF config (text_config)
  inference.json                                    upstream ModelArgs
  engram.json                                       Engram hash tables
  golden.safetensors                                reference outputs, see dump()

usage: tiny.py [--medium] OUT
  --medium  widths of the upstream self-test (dim 1024, matrices in multiples
            of 128) instead of the committed fixture's (dim 128): for
            check_kernels.py on a GPU, where tilelang kernels may not accept
            the tiny widths. Not committed; generate it where it is used.
"""
import json
import os
import sys
import tempfile

import torch

import fetch
import kernel_cpu as K
import run

SEED = 20261103

# Upstream ModelArgs of the tiny model. Every field the real config sets is
# set here; dimensions satisfy the kernels' block sizes (multiples of 32,
# and of 16 for the compressed KV's FP4 groups).
ARGS = dict(
    max_batch_size=1,
    max_seq_len=64,
    dtype="fp8",
    expert_dtype="fp4",
    vocab_size=512,
    dim=128,
    moe_inter_dim=64,
    n_layers=8,
    n_mtp_layers=1,
    n_heads=4,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=64,
    head_dim=64,
    rope_head_dim=16,
    norm_eps=1e-20,
    o_groups=2,
    o_lora_rank=32,
    window_size=8,
    # 0, 1: SWA; 2-4: encoder CSA2 m=2 (2 Full, 3-4 Reuse); 5-7: decoder
    # m=1 (5 Full and candidate source, 6 Reuse, 7 Reindex); 8: DSpark.
    compress_ratios=(0, 0, 2, 2, 2, 1, 1, 1, 0),
    kv_source_layers=(2, 5),
    index_source_layers=(2, 5, 7),
    compress_rope_theta=160000,
    original_seq_len=16,  # YaRN active within the dumped positions
    rope_theta=10000,
    rope_factor=4,
    beta_fast=32,
    beta_slow=1,
    index_n_heads=4,
    index_head_dim=32,
    index_topk=8,
    candidate_source_layer=5,
    candidate_topk_blocks=3,
    candidate_block_size=4,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=(1, 4),
    engram_max_ngram_size=4,
    engram_vocab_size=97,
    engram_n_heads=2,
    engram_head_dim=32,
    engram_pad_id=2,
    # vision: two blocks, 4-pixel patches merged 2x2, spans of at most 24
    vision_n_layers=2,
    vision_dim=64,
    vision_n_heads=4,
    vision_inter_dim=96,
    vision_patch_size=4,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=2,
    vision_max_n_token=24,
    vision_min_pixels=64 * 64,
    image_token_id=511,
    dspark_block_size=3,
    dspark_noise_token_id=500,
    dspark_target_layer_ids=(5, 6, 7),
    dspark_markov_rank=32,
    dspark_n_routed_experts=4,
    dspark_n_activated_experts=1,
    temperature=0.0,
)

# --medium: the upstream self-test's widths; sequence-related sizes stay
# small so that compression, top-k and the candidate pool still bite.
MEDIUM = dict(
    dim=1024,
    moe_inter_dim=256,
    n_heads=16,
    q_lora_rank=256,
    head_dim=128,
    rope_head_dim=32,
    o_groups=8,
    o_lora_rank=128,
    index_n_heads=16,
    index_head_dim=64,
    engram_head_dim=64,
)

PROMPT_LEN = 23  # odd: a ratio-2 group is left open after the prefill
DECODE_STEPS = 17


def quant_fp8_blocks(w, rows=32, cols=32):
    """FP8 E4M3 with one power-of-two E8M0 scale per rows x cols block."""
    N, Kd = w.shape
    b = w.view(N // rows, rows, Kd // cols, cols)
    amax = b.abs().amax((1, 3)).clamp_min(1e-4)
    s = K._log2_ceil_pow2(amax / 448)
    q = (b / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(N, Kd), K.f32_to_e8m0(s)


def quant_fp4_rows(w):
    """FP4 E2M1 packed two per byte along K, one E8M0 scale per 32 of a row."""
    packed, s = K.fp4_act_quant(w.float(), 32, False, torch.float8_e8m0fnu)
    return packed.view(torch.int8), s


class Gen:
    def __init__(self, seed=SEED):
        self.g = torch.Generator().manual_seed(seed)

    def normal(self, shape, std=1.0, mean=0.0):
        return torch.randn(shape, generator=self.g, dtype=torch.float32) * std + mean

    def uniform(self, shape, lo, hi):
        return torch.rand(shape, generator=self.g, dtype=torch.float32) * (hi - lo) + lo


def bf16(x):
    return x.to(torch.bfloat16)


def is_vl(name):
    """The image path's tensors: the vision tower, the aligner, the span
    embeddings and the gates' image routing biases."""
    return name.startswith(("vision.", "aligner.", "image_")) or name.endswith(".bias_vl")


def build_checkpoint(model):
    """Random weights for every tensor of the published checkpoint, keyed
    and typed as it stores them. The image path's come last, from their own
    generator, so the others are those of the text-only fixture."""
    gen = Gen()
    ckpt = {}
    params = dict(model.named_parameters(remove_duplicate=False))
    for name, p in params.items():
        if is_vl(name):
            continue
        if name.endswith(".scale") and name.replace(".scale", ".weight") in params:
            continue  # made with its weight
        if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
            continue  # tied to the backbone's
        shape = tuple(p.shape)
        leaf = name.rsplit(".", 1)[-1]
        if ".engram.embed.weight" in name:
            w = gen.normal(shape, 0.5)
            q, s = quant_fp8_blocks(w, 1, 32)
            ckpt[name], ckpt[name.replace("weight", "scale")] = q, s
        elif p.dtype == torch.float4_e2m1fn_x2:
            out, half = shape
            w = gen.normal((out, 2 * half), (2 * half) ** -0.5)
            ckpt[name], ckpt[name.replace("weight", "scale")] = quant_fp4_rows(w)
        elif p.dtype == torch.float8_e4m3fn or name.endswith("wo_a.weight"):
            w = gen.normal(shape, shape[1] ** -0.5)
            ckpt[name], ckpt[name.replace("weight", "scale")] = quant_fp8_blocks(w)
        elif "norm" in name or name.endswith(("q_weight", "k_weight")):
            ckpt[name] = bf16(gen.normal(shape, 0.1, 1.0))
        elif leaf == "attn_sink":
            ckpt[name] = gen.normal(shape, 1.0)
        elif name.endswith(("gate.bias", "gate.bias_vl")):
            ckpt[name] = gen.normal(shape, 0.1)
        elif leaf in ("hc_attn_fn", "hc_ffn_fn"):
            ckpt[name] = gen.normal(shape, shape[1] ** -0.5)
        elif leaf in ("hc_attn_scale", "hc_ffn_scale"):
            ckpt[name] = gen.uniform(shape, 0.5, 1.5)
        elif leaf in ("hc_attn_base", "hc_ffn_base"):
            ckpt[name] = gen.normal(shape, 1.0)
        elif name == "embed.weight" or name.endswith("markov_head.embed.weight"):
            ckpt[name] = bf16(gen.normal(shape, 1.0))
        elif p.dim() == 2:  # bf16 (or bf16-stored fp32) matrices
            ckpt[name] = bf16(gen.normal(shape, shape[1] ** -0.5))
        else:
            raise SystemExit(f"no initializer for {name} {p.dtype} {shape}")
    gen = Gen(SEED + 2)
    for name, p in params.items():
        if not is_vl(name):
            continue
        shape = tuple(p.shape)
        if name.endswith(".bias_vl"):
            ckpt[name] = gen.normal(shape, 0.1)
        elif "norm" in name:
            ckpt[name] = bf16(gen.normal(shape, 0.1, 1.0))
        elif name.startswith("image_"):
            ckpt[name] = bf16(gen.normal(shape, 1.0))
        elif name.endswith(".bias"):
            ckpt[name] = bf16(gen.normal(shape, 0.1))
        elif p.dim() == 2:
            ckpt[name] = bf16(gen.normal(shape, shape[1] ** -0.5))
        else:
            raise SystemExit(f"no initializer for {name} {p.dtype} {shape}")
    return ckpt


def hf_config(args, engram_num_embeddings, compressed_vocab):
    """config.json in the published schema, for the Go importer."""
    a = args
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": a.image_token_id,
        "vision_config": {
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": a.vision_n_layers,
            "hidden_size": a.vision_dim,
            "num_attention_heads": a.vision_n_heads,
            "intermediate_size": a.vision_inter_dim,
            "patch_size": a.vision_patch_size,
            "rope_theta": a.vision_rope_theta,
            "downsample_ratio": a.vision_downsample_ratio,
            "max_image_tokens": a.vision_max_n_token,
            "min_pixels": a.vision_min_pixels,
            "max_wh_ratio": a.vision_max_wh_ratio,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": a.vocab_size,
            "hidden_size": a.dim,
            "moe_intermediate_size": a.moe_inter_dim,
            "num_hidden_layers": a.n_layers,
            "num_attention_heads": a.n_heads,
            "num_key_value_heads": 1,
            "head_dim": a.head_dim,
            "qk_rope_head_dim": a.rope_head_dim,
            "q_lora_rank": a.q_lora_rank,
            "o_lora_rank": a.o_lora_rank,
            "o_groups": a.o_groups,
            "hidden_act": "silu",
            "swiglu_limit": a.swiglu_limit,
            "rms_norm_eps": a.norm_eps,
            "tie_word_embeddings": False,
            "max_position_embeddings": a.max_seq_len,
            "rope_theta": a.rope_theta,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": a.rope_factor,
                "beta_fast": a.beta_fast,
                "beta_slow": a.beta_slow,
                "original_max_position_embeddings": a.original_seq_len,
            },
            "n_routed_experts": a.n_routed_experts,
            "n_shared_experts": a.n_shared_experts,
            "num_experts_per_tok": a.n_activated_experts,
            "scoring_func": a.score_func,
            "topk_method": "noaux_tc",
            "norm_topk_prob": a.norm_topk_prob,
            "routed_scaling_factor": a.route_scale,
            "sliding_window": a.window_size,
            "compress_ratios": list(a.compress_ratios),
            "compress_rope_theta": a.compress_rope_theta,
            "kv_source_layer_ids": list(a.kv_source_layers),
            "index_source_layer_ids": list(a.index_source_layers),
            "index_n_heads": a.index_n_heads,
            "index_head_dim": a.index_head_dim,
            "index_topk": a.index_topk,
            "candidate_source_layer_id": a.candidate_source_layer,
            "candidate_topk_blocks": a.candidate_topk_blocks,
            "candidate_block_size": a.candidate_block_size,
            "hc_mult": a.hc_mult,
            "hc_sinkhorn_iters": a.hc_sinkhorn_iters,
            "hc_eps": a.hc_eps,
            "engram_layer_ids": list(a.engram_layer_ids),
            "engram_num_embeddings": list(engram_num_embeddings),
            "engram_max_ngram_size": a.engram_max_ngram_size,
            "engram_vocab_size": a.engram_vocab_size,
            "engram_n_heads": a.engram_n_heads,
            "engram_head_dim": a.engram_head_dim,
            "engram_pad_token_id": a.engram_pad_id,
            "engram_compressed_vocab_size": compressed_vocab,
            "num_nextn_predict_layers": a.n_mtp_layers,
            "dspark_block_size": a.dspark_block_size,
            "dspark_noise_token_id": a.dspark_noise_token_id,
            "dspark_target_layer_ids": list(a.dspark_target_layer_ids),
            "dspark_markov_rank": a.dspark_markov_rank,
            "dspark_n_routed_experts": a.dspark_n_routed_experts,
            "dspark_num_experts_per_tok": a.dspark_n_activated_experts,
        },
    }


def engram_tables(model, engram_mod, args, tokenizer):
    """What the engine needs to hash n-grams without the tokenizers library
    or numpy's RNG: the compressed token map and, per Engram layer, the
    bucket primes, row offsets and hash multipliers."""
    h = model.engram_hash
    return {
        "max_ngram_size": args.engram_max_ngram_size,
        "n_heads": args.engram_n_heads,
        "compressed_vocab_size": args.engram_compressed_vocab_size,
        "pad_id": int(h.pad_id),
        "token_map": h.token_map.tolist(),
        "layers": [
            {
                "layer": layer,
                "num_embeddings": model.engram_layout.num_embeddings[i],
                "primes": h.primes[i].tolist(),  # [ngram size - 2][head]
                "offsets": h.offsets[i].tolist(),  # per (ngram size, head), flattened
                "multipliers": h.multipliers[i].tolist(),  # per lookback
            }
            for i, layer in enumerate(model.engram_layout.layer_ids)
        ],
    }


class Recorder:
    """Collects intermediate values through forward hooks."""

    def __init__(self, model, m):
        self.out = {}
        self.step = None
        self.m = m
        for i, layer in enumerate(model.layers):
            layer.register_forward_hook(self.block_hook(f"layer{i}"))
            if layer.engram is not None:
                layer.engram.register_forward_hook(self.tensor_hook(f"layer{i}.engram"))
            if layer.attn.indexer is not None or layer.attn.is_kv_source:
                layer.attn.register_forward_hook(self.attn_hook(i, layer.attn))
        for i, layer in enumerate(model.mtp):
            layer.register_forward_hook(self.block_hook(f"mtp{i}"))

    def put(self, name, t):
        self.out[f"{self.step}.{name}"] = t.detach().clone().contiguous()

    def block_hook(self, name):
        def hook(mod, args, output):
            h, pre_mix = output
            self.put(f"{name}.h", h)
            self.put(f"{name}.pre_mix", pre_mix)

        return hook

    def tensor_hook(self, name):
        def hook(mod, args, output):
            self.put(name, output)

        return hook

    def attn_hook(self, i, attn):
        sa = self.m.shared_attn

        def hook(mod, args, output):
            self.put(f"layer{i}.attn.out", output)
            if attn.is_index_source and sa.topk_idxs is not None:
                self.put(f"layer{i}.topk_idxs", sa.topk_idxs)
            if i == self.m_args.candidate_source_layer and sa.candidates is not None:
                self.put(f"layer{i}.candidates", sa.candidates)

        return hook


def dump(model, m, args, tokens):
    """Teacher-forced run: prefill PROMPT_LEN tokens, then one decode step
    per further token. Recorded per step ("prefill", "decode0", ...):
      logits [b, vocab] f32; output ids; main_hidden (DSpark input);
      layer{i}.h / .pre_mix after every block; layer{i}.engram output;
      layer{i}.attn.out of source layers; layer{i}.topk_idxs of index
      sources; layer5.candidates; spec.* (DSpark draft ids, logits,
      confidence) after each decode step.
    Final caches: cache.layer{i}.window_kv / compress_kv / index_k."""
    rec = Recorder(model, m)
    rec.m_args = args
    out = rec.out
    out["tokens"] = torch.tensor(tokens, dtype=torch.int64)
    x = torch.tensor([tokens])
    rec.step = "prefill"
    ids, logits, main_hidden = model(x[:, :PROMPT_LEN], 0)
    out["prefill.logits"] = logits.clone()
    out["prefill.main_hidden"] = main_hidden.clone()
    model.forward_spec(ids, main_hidden, 0)
    for step, pos in enumerate(range(PROMPT_LEN, PROMPT_LEN + DECODE_STEPS)):
        rec.step = f"decode{step}"
        ids, logits, main_hidden = model(x[:, pos : pos + 1], pos)
        out[f"decode{step}.logits"] = logits.clone()
        out[f"decode{step}.main_hidden"] = main_hidden.clone()
        spec = model.forward_spec(ids, main_hidden, pos)
        out[f"decode{step}.spec.ids"], out[f"decode{step}.spec.logits"], out[f"decode{step}.spec.confidence"] = (
            t.clone() for t in spec
        )
    end = PROMPT_LEN + DECODE_STEPS
    for i, layer in enumerate(model.layers):
        a = layer.attn
        out[f"cache.layer{i}.window_kv"] = a.window_kv_cache[:1].clone()
        if a.is_kv_source:
            out[f"cache.layer{i}.compress_kv"] = a.compress_kv_cache[:1, : end // a.compress_ratio].clone()
            out[f"cache.layer{i}.index_k"] = a.indexer.k_cache[:1, : end // a.compress_ratio].clone()
    return {k: v.contiguous() for k, v in out.items()}


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--medium":
        ARGS.update(MEDIUM)
        argv = argv[1:]
    if len(argv) != 1:
        raise SystemExit(__doc__)
    out_dir = argv[0]
    os.makedirs(out_dir, exist_ok=True)
    m = run.setup()
    import engram as engram_mod
    from safetensors.torch import save_file

    torch.set_default_dtype(torch.bfloat16)
    tokenizer = run.SubsetTokenizer(fetch.fetch("tokenizer.json"), ARGS["vocab_size"])
    _, compressed = engram_mod.build_compressed_token_map(tokenizer)
    args = m.ModelArgs(**ARGS, engram_compressed_vocab_size=compressed)
    layout = engram_mod.EngramLayout.from_args(args)  # sizes the tables
    args.engram_num_embeddings = tuple(sum(p for per in layer for p in per) for layer in layout.primes)
    model = m.Transformer(args, tokenizer)

    ckpt = build_checkpoint(model)
    ckpt_path = os.path.join(out_dir, "model.safetensors")
    save_file(ckpt, ckpt_path, metadata={"format": "pt"})
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": os.path.getsize(ckpt_path)}, "weight_map": {k: "model.safetensors" for k in sorted(ckpt)}}, f, indent=1)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hf_config(args, args.engram_num_embeddings, compressed), f, indent=2)
    inference = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    with open(os.path.join(out_dir, "inference.json"), "w") as f:
        json.dump(inference, f, indent=2)
    with open(os.path.join(out_dir, "engram.json"), "w") as f:
        json.dump(engram_tables(model, engram_mod, args, tokenizer), f)

    # The dump comes from the files just written, loaded as a user would.
    model, args = load(m, out_dir)
    golden = dump(model, m, args, fixture_tokens())
    model, args = load(m, out_dir)  # caches anew
    golden.update(dump_vl(model, m, args))
    save_file(golden, os.path.join(out_dir, "golden.safetensors"), metadata={"seed": str(SEED)})
    print(f"{len(ckpt)} checkpoint tensors, {len(golden)} golden tensors in {out_dir}")


VL_DECODE_STEPS = 6


def fixture_image():
    """A seeded 30x17 PNG: scaled up to the minimum pixels, then down to
    the span limit, so both of the planner's branches run."""
    import io

    import numpy as np
    from PIL import Image

    g = torch.Generator().manual_seed(SEED + 3)
    a = torch.randint(0, 256, (17, 30, 3), generator=g, dtype=torch.uint8).numpy()
    y, x = np.mgrid[0:17, 0:30]
    a[..., 0] = (x * 8).astype(np.uint8)  # some structure besides the noise
    buf = io.BytesIO()
    Image.fromarray(a, "RGB").save(buf, "PNG")
    return buf.getvalue()


def dump_vl(model, m, args):
    """A prompt with an image, teacher-forced as dump(): text, the image's
    span, text; prefilled, then VL_DECODE_STEPS decode steps. Recorded under
    "vl.": png (the file), tokens and types (the expanded prompt and the
    steps), patches (load_image's), vit.block{i} and vit (the encoder's
    outputs), embeds (the aligner's), and per step as dump() does."""
    import image_processor as IP

    png = fixture_image()
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = IP.load_image({"data": png}, args)
    types = IP.image_token_types(n_llm_h, n_llm_w)
    g = torch.Generator().manual_seed(SEED + 4)
    words = torch.randint(3, args.vocab_size - 1, (5 + 4 + VL_DECODE_STEPS,), generator=g, device="cpu").tolist()
    tokens = words[:5] + [args.image_token_id] * types.numel() + words[5:]
    token_types = [IP.TEXT] * 5 + types.tolist() + [IP.TEXT] * (len(words) - 5)
    image = IP.ImageInput(5, patches, n_vit_h, n_vit_w, types)
    prompt_len = len(tokens) - VL_DECODE_STEPS

    out = {}
    for i, block in enumerate(model.vision.blocks):
        block.register_forward_hook(lambda mod, a, o, i=i: out.__setitem__(f"vl.vit.block{i}", o.detach().clone()))
    out["vl.vit"] = model.vision(patches, n_vit_h, n_vit_w).clone()
    out["vl.embeds"] = model.encode_image(patches, n_vit_h, n_vit_w).clone()
    rec = Recorder(model, m)
    rec.m_args = args
    rec.out = out
    x = torch.tensor([tokens])
    tt = torch.tensor([token_types])
    rec.step = "vl.prefill"
    _, logits, main_hidden = model(x[:, :prompt_len], 0, [[image]], tt[:, :prompt_len])
    out["vl.prefill.logits"] = logits.clone()
    out["vl.prefill.main_hidden"] = main_hidden.clone()
    for step, pos in enumerate(range(prompt_len, len(tokens))):
        rec.step = f"vl.decode{step}"
        _, logits, main_hidden = model(x[:, pos : pos + 1], pos)
        out[f"vl.decode{step}.logits"] = logits.clone()
        out[f"vl.decode{step}.main_hidden"] = main_hidden.clone()
    out["vl.png"] = torch.frombuffer(bytearray(png), dtype=torch.uint8)
    out["vl.tokens"] = torch.tensor(tokens, dtype=torch.int64)
    out["vl.types"] = torch.tensor(token_types, dtype=torch.int64)
    out["vl.patches"] = patches
    return {k: v.contiguous() for k, v in out.items()}


def fixture_tokens():
    g = torch.Generator().manual_seed(SEED + 1)
    # On the CPU, whatever the default device: the GPU check runs with CUDA's.
    tokens = torch.randint(3, ARGS["vocab_size"], (PROMPT_LEN + DECODE_STEPS,), generator=g, device="cpu").tolist()
    tokens[5:9] = tokens[12:16]  # a repeated 4-gram, so Engram n-grams recur
    return tokens


def load(m, fixture_dir, device="cpu"):
    """The tiny model from a fixture directory: upstream convert.py, then
    load_model, as generate.py loads the real one."""
    import convert
    from safetensors.torch import load_model

    torch.set_default_dtype(torch.bfloat16)
    args = run.load_args(m, os.path.join(fixture_dir, "inference.json"))
    tokenizer = run.SubsetTokenizer(fetch.fetch("tokenizer.json"), args.vocab_size)
    with torch.device(device):
        model = m.Transformer(args, tokenizer)
    with tempfile.TemporaryDirectory() as tmp:
        # convert.py takes every *.safetensors in its input: give it the checkpoint alone
        hf = os.path.join(tmp, "hf")
        os.mkdir(hf)
        for name in ("model.safetensors", "model.safetensors.index.json"):
            os.symlink(os.path.abspath(os.path.join(fixture_dir, name)), os.path.join(hf, name))
        convert.main(hf, os.path.join(tmp, "out"), 1, "fp4", None)
        load_model(model, os.path.join(tmp, "out", "model0-mp1.safetensors"), device=device)
    return model, args


if __name__ == "__main__":
    main()
