#!/usr/bin/env python3
"""Degraded-mount smoke: the two patched files that reference symbols from OTHER
patched files are mounted WITHOUT those files. Both must import (guarded fallback),
not raise ImportError and get swallowed by a loader into a slow emulation path."""
import importlib, sys

for mod in ("vllm.model_executor.layers.rotary_embedding.mrope",
            "vllm.model_executor.kernels.linear.mxfp4.r4dhip"):
    m = importlib.import_module(mod)
    sym = "cos_sin_halves" if "mrope" in mod else "fp8_quant_shared"
    assert not hasattr(m, sym), f"{mod} looks PATCHED ({sym} present) — test is not testing the fallback"
    print(f"stock (unpatched) as intended: {mod}")

qsa = importlib.import_module("vllm.models.qwen4_exp.amd.indexer_qsa")
assert callable(qsa.cos_sin_halves), "indexer_qsa fallback cos_sin_halves missing"
print("import OK  indexer_qsa  (fallback cos_sin_halves in use)")

moe = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.r4d_mxfp4_moe")
assert callable(moe.fp8_quant_shared), "r4d moe fallback fp8_quant_shared missing"
assert hasattr(moe, "R4dMxfp4MoEExperts"), "r4d moe class missing — loader would report 'r4d unavailable'"
print("import OK  r4d_mxfp4_moe  (fallback fp8_quant_shared in use, R4dMxfp4MoEExperts present)")

# the fallback must be numerically the same code path as stock mrope
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
with set_current_vllm_config(VllmConfig()):
    rope = MRotaryEmbedding(head_size=256, rotary_dim=64, max_position_embeddings=8192,
                            base=10000.0, is_neox_style=True, dtype=torch.bfloat16,
                            mrope_section=[16, 8, 8])
x = torch.randn(4, 2, 256, dtype=torch.bfloat16)
c, s = qsa.cos_sin_halves(rope, x)
ref_c, ref_s = rope._match_cos_sin_cache_dtype(x).chunk(2, dim=-1)
assert torch.equal(c, ref_c) and torch.equal(s, ref_s), "fallback cos/sin differ from chunk()"
print("fallback cos_sin_halves: bit-identical to chunk()")
print("\nDEGRADED-MOUNT SMOKE OK")
