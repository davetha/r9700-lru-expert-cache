"""Import + shape smoke test for the k4 patch set (run with the MOUNTS.txt binds)."""
import importlib, inspect, sys, os

mods = [
    "vllm.ir.op",
    "vllm.model_executor.layers.layernorm",
    "vllm.model_executor.layers.rotary_embedding.mrope",
    "vllm.model_executor.layers.fused_moe.modular_kernel",
    "vllm.model_executor.kernels.linear.mxfp4.r4dhip",
    "vllm.model_executor.layers.fused_moe.experts.r4d_mxfp4_moe",
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4",
    "vllm.model_executor.layers.utils",
    "vllm.models.qwen4_exp.amd.indexer_qsa",
    "vllm.models.qwen4_exp.amd.mtp",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating",
    "vllm.model_executor.layers.fused_gate_mul",
    "vllm.model_executor.models.qwen2_moe",
]
for m in mods:
    mod = importlib.import_module(m)
    print("import OK  %-70s %s" % (m, getattr(mod, "__file__", "?")))

# the patches are actually the ones loaded, not the image copies
from vllm.model_executor.layers.layernorm import GEMMA_NORM_FUSED, GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import cos_sin_halves
from vllm.model_executor.kernels.linear.mxfp4.r4dhip import fp8_quant_shared
from vllm.model_executor.layers.fused_moe.experts import r4d_mxfp4_moe
import vllm.models.qwen4_exp.amd.indexer_qsa as iq

assert hasattr(GemmaRMSNorm, "_weight_for")
from vllm.model_executor.layers.layernorm import _fused_norm_impl  # noqa: F401
assert "fp8_quant_shared" in inspect.getsource(r4d_mxfp4_moe.R4dMxfp4MoEExperts._apply_split)
assert "cos_sin_halves" in inspect.getsource(iq.apply_qsa_rope)
print("\nGEMMA_NORM_FUSED =", GEMMA_NORM_FUSED)
print("all patched symbols present")

# exercise cos_sin_halves + the fp8 memo on real tensors
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

with set_current_vllm_config(VllmConfig()):
    rope = MRotaryEmbedding(
        head_size=256, rotary_dim=64, max_position_embeddings=8192, base=10000.0,
        is_neox_style=True, dtype=torch.bfloat16, mrope_section=[16, 8, 8],
    ).cuda()
x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
c, s = cos_sin_halves(rope, x)
cache = rope._match_cos_sin_cache_dtype(x)
ref_c, ref_s = cache.chunk(2, dim=-1)
assert torch.equal(c, ref_c) and torch.equal(s, ref_s), "split cache differs from chunk()"
assert c.is_contiguous() and s.is_contiguous()
assert cos_sin_halves(rope, x)[0] is c, "split cache rebuilt on every call"
print("cos_sin_halves: values identical to chunk(), contiguous, memoized")

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mxfp4 import r4dhip

# EVERYTHING below runs under inference_mode, like the real model forward: that is where
# tensors have no version counter. The earlier version of this test ran outside it and so
# missed `_version` raising, which killed a whole arm at startup.
with torch.inference_mode():
    h = torch.randn(64, 2560, dtype=torch.bfloat16, device="cuda")
    assert h.is_inference(), "test tensor is not an inference tensor"

    # default OFF: plain pass-through, and no caching of any kind
    assert r4dhip._A8_SHARE is False, "VLLM_R4D_SHARE_A8 must default to 0"
    q1, s1 = fp8_quant_shared(h)
    q2, s2 = fp8_quant_shared(h)
    assert q1 is not q2, "sharing is on by default"
    qr, sr = ops.scaled_fp8_quant(h, use_per_token_if_dynamic=True)
    assert torch.equal(q1, qr) and torch.equal(s1, sr)
    print("fp8_quant_shared: default off, pass-through, correct on inference tensors")

    # the layernorm weight memo must not touch _version either
    with set_current_vllm_config(VllmConfig()):
        gn = GemmaRMSNorm(2560).cuda().to(torch.bfloat16)
    xw = torch.randn(8, 2560, dtype=torch.bfloat16, device="cuda")
    w1 = gn._weight_for(xw)
    assert w1.dtype == xw.dtype and gn._weight_for(xw) is w1, "weight memo miss"
    assert torch.equal(w1, (gn.weight.float() + 1.0).to(xw.dtype))
    print("GemmaRMSNorm._weight_for: no version counter, memoized, matches (1+w)")

    # enabled path: single-use identity handoff, correct values, no stale second serve
    r4dhip._A8_SHARE = True
    r4dhip._a8_slot = None
    try:
        a1, b1 = fp8_quant_shared(h)
        a2, b2 = fp8_quant_shared(h)          # same object -> the handoff hit
        assert a2 is a1 and b2 is b1, "handoff did not hit on the same tensor object"
        a3, _ = fp8_quant_shared(h)           # slot cleared -> recomputed, never stale
        assert a3 is not a1, "handoff served the same hit twice"
        assert torch.equal(a1, qr), "shared quant differs from the plain quant"
        h2 = torch.randn(64, 2560, dtype=torch.bfloat16, device="cuda")
        assert fp8_quant_shared(h2)[0] is not a3, "handoff hit on a different tensor"
    finally:
        r4dhip._A8_SHARE = False
        r4dhip._a8_slot = None
    print("fp8_quant_shared: when enabled, single-use identity handoff, values match")

# ---- draft-only W4 lm_head (#8): gate, guarded import, single pack ----------
import types

import vllm.models.qwen4_exp.amd.mtp as qmtp

assert qmtp.DRAFT_W4_LMHEAD is False, "VLLM_DRAFT_W4_LMHEAD must default to 0"


class _LP:                       # a stock LogitsProcessor's relevant fields
    logits_as_input = False
    scale = 1.0
    soft_cap = None
    head_dtype = None
    org_vocab_size = 248320

    def _gather_logits(self, logits):
        return torch.cat([logits, logits], dim=-1)   # stand-in for tp_size=2

    def get_top_tokens(self, lm_head, hidden_states):
        return torch.full((hidden_states.shape[0],), -1, dtype=torch.int64,
                          device=hidden_states.device)   # delegation sentinel


class _Shard:
    num_org_vocab_padding = 64
    org_vocab_start_index = 4096


class _Head:
    tp_size = 2

    def __init__(self):
        self.weight = torch.randn(1024, 256, dtype=torch.bfloat16, device="cuda")
        self.shard_indices = _Shard()


class _MTP:
    def __init__(self):
        self.logits_processor = _LP()
        self.lm_head = _Head()
        self._draft_w4_head = None

    _pack_w4_head_once = qmtp.Qwen4ExpMTP._pack_w4_head_once
    _w4_local_logits = qmtp.Qwen4ExpMTP._w4_local_logits
    _w4_draft_logits = qmtp.Qwen4ExpMTP._w4_draft_logits
    get_top_tokens = qmtp.Qwen4ExpMTP.get_top_tokens


m = _MTP()
x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
assert m._w4_draft_logits(x) is None, "gate off must fall through to bf16"
assert m._draft_w4_head is None, "gate off must not mark the head unavailable"

assert qmtp.DRAFT_HS_DUMP == "", "VLLM_DRAFT_HS_DUMP must default to off"

# The pack-time verdict is now a collective; stand in for the peer rank's vote.
# `+1` = the peer also said yes, `+0` = the peer refused.
_stock_all_reduce = qmtp.tensor_model_parallel_all_reduce
_peer_vote = [1]
qmtp.tensor_model_parallel_all_reduce = lambda t: t + _peer_vote[0]

qmtp.DRAFT_W4_LMHEAD = True
try:
    # K3's real module, mounted: pack + GEMM must produce the same shape and
    # dtype the bf16 head would, at the 4-bit grid's error and no worse.
    import vllm.model_executor.kernels.draft_w4_lmhead as w4

    assert w4.available(), "draft_w4_lmhead.available() is False in this container"
    mr = _MTP()
    ref = (x.float() @ mr.lm_head.weight.t().float())
    out = mr._w4_draft_logits(x)
    assert out is not None, "the real W4 path returned None"
    assert out.shape == (4, 2048) and out.dtype == x.dtype, (out.shape, out.dtype)
    got = out[:, :1024].float()               # _gather_logits duplicated the shard
    rel = (got - ref).norm() / ref.norm()
    assert rel < 0.25, f"W4 draft logits error {rel:.3f} is not the 4-bit grid"
    head_id = id(mr._draft_w4_head)
    mr._w4_draft_logits(x)
    assert id(mr._draft_w4_head) == head_id, "weight repacked on a later call"
    print(f"draft W4 lm_head: real kernel OK, rel err {rel:.3f}, packs once")

    # a non-plain logits processor must disable the path instead of dropping the op
    m3 = _MTP()
    m3.logits_processor.soft_cap = 30.0
    assert m3._w4_draft_logits(x) is None and m3._draft_w4_head is False

    # losing the mount must cost the speedup, not the server
    sys.modules["vllm.model_executor.kernels.draft_w4_lmhead"] = None
    m4 = _MTP()
    assert m4._w4_draft_logits(x) is None, "missing K3 module must fall back"
    assert m4._draft_w4_head is False, "fallback must be sticky"
    print("draft W4 lm_head: missing K3 module degrades to bf16, refuses soft_cap")

    sys.modules["vllm.model_executor.kernels.draft_w4_lmhead"] = w4   # unpoison

    # get_top_tokens: with the W4 head live it must reduce the SAME shard-local
    # logits it would have gathered, and never return a padded vocab slot.
    m5 = _MTP()
    m5.lm_head.tp_size = 1                    # the all-gather needs a real group
    local = m5._w4_local_logits(x)
    assert local is not None and local.shape == (4, 1024), local.shape
    top = m5.get_top_tokens(x)
    masked = local.clone()
    masked[..., -_Shard.num_org_vocab_padding:] = -float("inf")
    want = masked.argmax(dim=-1) + _Shard.org_vocab_start_index
    assert torch.equal(top, want), (top, want)
    assert top.dtype == torch.int64
    lo, hi = _Shard.org_vocab_start_index, _Shard.org_vocab_start_index + 1024
    assert int(top.min()) >= lo and int(top.max()) < hi - _Shard.num_org_vocab_padding

    # with the W4 head off it must fall through to the stock reduction
    qmtp.DRAFT_W4_LMHEAD = False
    m6 = _MTP()
    m6.lm_head.tp_size = 1
    assert torch.equal(m6.get_top_tokens(x), torch.full((4,), -1,
                       dtype=torch.int64, device=x.device))
    qmtp.DRAFT_W4_LMHEAD = True
    print("get_top_tokens: folds in the W4 head, masks padding, else delegates")

    # A SPLIT verdict must disable the W4 head on every rank. Under
    # get_top_tokens a one-rank failure would otherwise compare W4 values from
    # one shard against bf16 values from the other and bias the winner.
    _peer_vote[0] = 0
    m7 = _MTP()
    assert m7._w4_local_logits(x) is None, "a split TP verdict must disable W4"
    assert m7._draft_w4_head is False, "the split-verdict disable must be sticky"
    _peer_vote[0] = 1
    m8 = _MTP()
    assert m8._w4_local_logits(x) is not None, "a unanimous verdict must engage"
    print("draft W4 lm_head: TP verdict is unanimous or the head is off everywhere")
finally:
    qmtp.tensor_model_parallel_all_reduce = _stock_all_reduce
    qmtp.DRAFT_W4_LMHEAD = False
    sys.modules["vllm.model_executor.kernels.draft_w4_lmhead"] = w4  # noqa: F821

# ---- #2 GDN strided qkv and #7 fused shared-expert gate: gates + probes -----
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
import vllm.model_executor.models.qwen2_moe as q2moe
from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating as fsg

assert gdn.GDN_STRIDED_QKV is False, "VLLM_GDN_STRIDED_QKV must default to 0"
assert q2moe.FUSED_SHARED_GATE is False, "VLLM_FUSED_SHARED_GATE must default to 0"
assert fsg.SUPPORTS_STRIDED_QKV is True, "patched FLA kernel not loaded"
assert gdn._fla_supports_strided_qkv() is True, "GDN cannot see the strided FLA kernel"
assert q2moe._fused_gate_mul() is not None, "qwen2_moe cannot see the fused gate kernel"
# a contiguous [B,T,H,K] tensor must still report the stock token stride
assert fsg._token_stride(torch.empty(1, 4, 8, 128, device="cuda"), 8, 128) == 8 * 128
# ... and a layout the kernel cannot walk must ask for a copy
assert fsg._token_stride(torch.empty(1, 4, 128, 8, device="cuda").transpose(2, 3),
                         8, 128) is None
print("#2/#7: gates default off, both kernels visible, stride probe correct")


# ---- #11 fused silu_and_mul + per-token fp8 quant: gate, guard, exactness ---
from vllm.model_executor.layers import fused_silu_mul_quant as fsq
from vllm.model_executor.layers.fused_moe.experts import r4d_mxfp4_moe as r4dmoe
from vllm.platforms import current_platform as _plat

assert r4dmoe._FUSED_SILU_QUANT is False, "VLLM_FUSED_SILU_QUANT must default to 0"
assert r4dmoe._fused_silu_quant() is not None, "the MoE cannot see the fused kernel"

_fp8 = _plat.fp8_dtype()
# The whole point of the fusion is that it is a drop-in for the pair it replaces,
# so assert equality against the real ops, not a tolerance.
for _rows, _d in ((1, 320), (50, 320), (2048, 320)):
    _x = torch.randn(_rows, 2 * _d, device="cuda", dtype=torch.bfloat16)
    assert fsq.supported(_x)
    _ic2 = torch.empty(_rows, _d, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(_ic2, _x)
    _qr, _sr = ops.scaled_fp8_quant(_ic2, use_per_token_if_dynamic=True)
    _act, _q, _s = fsq.silu_mul_quant(_x, _fp8, write_act=True)
    assert torch.equal(_act, _ic2), f"activation differs at rows={_rows}"
    assert torch.equal(_s, _sr), f"scale differs at rows={_rows}"
    assert torch.equal(_q.view(torch.uint8), _qr.view(torch.uint8)), \
        f"fp8 payload differs at rows={_rows}"
# A row that is all zeros must land on vllm's scale floor, not divide by zero.
_z = torch.zeros(2, 640, device="cuda", dtype=torch.bfloat16)
_, _, _sz = fsq.silu_mul_quant(_z, _fp8)
assert torch.allclose(_sz, torch.full_like(_sz, 1.0 / (448.0 * 512.0))), "scale floor"
# A row too wide for one Triton block must be refused, not silently truncated.
assert not fsq.supported(torch.empty(2, 2 * 16384, device="cuda",
                                     dtype=torch.bfloat16)), "wide row must be refused"
print("#11 fused silu+quant: gate off by default, bit-identical to "
      "silu_and_mul + scaled_fp8_quant")

print("\nSMOKE OK")
