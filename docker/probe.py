"""ROCm-10 swap probe: does the fork's prebuilt stack load and run on the new runtime?
Fails loud at the first broken layer so the verdict is unambiguous."""
import ctypes, os, sys, subprocess
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda avail", torch.cuda.is_available())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
maps = subprocess.run(["bash", "-c", f"grep -E 'libamdhip64|libhsa-runtime' /proc/{os.getpid()}/maps | awk '{{print $6}}' | sort -u"], capture_output=True, text=True).stdout
print("HIP runtime mapped:\n" + maps)
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (x @ x).float().abs().mean().item())

lib = ctypes.CDLL("/app/fp8hip/libfp8hip_gemm.so"); print("fp8hip ctypes OK")
from vllm.model_executor.kernels import r4d_lib
print("r4d ctypes:", r4d_lib.lib())
r4d = r4d_lib.import_r4d(); print("r4d pybind11 module:", r4d, [a for a in dir(r4d) if not a.startswith("_")][:12] if r4d else "IMPORT FAILED")
from vllm import _custom_ops as ops
q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True); print("vllm _C op scaled_fp8_quant OK", q.dtype, s.shape)
import vllm._moe_C_stable_libtorch; print("_moe_C OK")
try:
    import aiter; print("aiter import OK", aiter.__version__ if hasattr(aiter, "__version__") else "")
except Exception as e:
    print("aiter import FAILED:", type(e).__name__, str(e)[:200])
import triton; print("triton", triton.__version__)
print("PROBE-OK")
