"""CPU-only import smoke test for the k2 UVA-offload patch to amd/model.py.
No GPU touched, no model constructed -- import only, same caliber as k4's smoke_imports.py
minus the .cuda() exercises (those need a GPU; this patch's own logic is import-time-safe)."""
import importlib
import inspect

# import the whole module chain amd/model.py pulls in, so a broken cross-patch mount
# (indexer_qsa.py etc, per k4's PATCHES.md "cross-patch import rule") shows up here too.
for m in [
    "vllm.models.qwen4_exp.amd.indexer_qsa",
    "vllm.models.qwen4_exp.amd.model",
]:
    mod = importlib.import_module(m)
    print("import OK  %-50s %s" % (m, getattr(mod, "__file__", "?")))

from vllm.models.qwen4_exp.amd import model as m

assert hasattr(m, "_k2_uva_offload_module_params"), "helper missing"
assert hasattr(m, "logger"), "module logger missing"

# Qwen4ExpModel/Qwen4ExpForConditionalGeneration.__init__ is rebound by the class-level
# @support_torch_compile decorator (compilation/decorators.py) to a wrapper that closes
# over the real body as `old_init` -- inspect.getsource() on the bound method returns the
# decorator's wrapper source, not ours. Check the mounted file text directly instead.
src_file = inspect.getsourcefile(m) or m.__file__
raw = open(src_file).read()
assert "VLLM_UVA_OFFLOAD_EMBED" in raw, "embed hook missing from mounted model.py"
assert "VLLM_UVA_OFFLOAD_VISUAL" in raw, "visual hook missing from mounted model.py"
assert raw.count("_k2_uva_offload_module_params(") == 3, (  # def + 2 call sites
    "expected exactly 2 call sites of the helper (embed_tokens, visual)"
)
print("both env-gated hooks present in source, helper called at exactly 2 sites")

# helper is inert with the env vars unset / on a module with no CUDA params to move --
# exercise the "nothing to offload" branch on a plain CPU nn.Module without importing torch.cuda.
import torch.nn as nn
class _Empty(nn.Module):
    pass
m._k2_uva_offload_module_params(_Empty(), "smoke-empty")  # no params -> no-op, must not raise
print("no-op path (empty module) OK")

class _CpuOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(__import__("torch").zeros(4))
m._k2_uva_offload_module_params(_CpuOnly(), "smoke-cpu-param")  # already-cpu param -> skip, must not raise
print("already-cpu-param skip path OK")

# Reproduce the PleOffloadWorker crash: that process constructs this model with every
# parameter as a meta tensor (torch.device("meta")). Before the fix, p.data.to(device="cpu")
# on a meta tensor raised NotImplementedError ("Cannot copy out of meta tensor; no data!").
# The fix guards on p.device.type != "cuda" so meta params are skipped, not copied.
import torch
class _MetaOnly(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.device("meta"):
            self.w = nn.Parameter(torch.zeros(4))
assert m.torch.device("meta") == torch.device("meta")
_meta_mod = _MetaOnly()
assert _meta_mod.w.device.type == "meta"
m._k2_uva_offload_module_params(_meta_mod, "smoke-meta-param")  # meta tensor -> must skip, not raise
assert _meta_mod.w.device.type == "meta", "meta param must be left untouched"
print("meta-device skip path OK (PleOffloadWorker repro)")

# Idempotency: a param already marked _vllm_is_uva_offloaded must not be touched twice.
# Note: `param.data` returns a fresh Tensor wrapper object on each Python-level access
# even when nothing changed, so compare storage via data_ptr(), not `is`.
class _AlreadyDone(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4))
_done_mod = _AlreadyDone()
_done_mod.w._vllm_is_uva_offloaded = True
_marker_ptr = _done_mod.w.data_ptr()
m._k2_uva_offload_module_params(_done_mod, "smoke-idempotent")
assert _done_mod.w.data_ptr() == _marker_ptr, "already-offloaded param must be left alone (idempotency)"
print("idempotent re-run skip path OK")

print("\nSMOKE OK (k2 UVA offload patch)")
