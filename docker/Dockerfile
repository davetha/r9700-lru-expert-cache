# ROCm 10.0 runtime swap under tcclaviger's Flash-Next vLLM fork image.
# Keeps python3.12 + torch 2.11 so the fork's prebuilt vllm _C/_moe_C, the closed
# pybind11 r4d.so / libfp8hip_gemm.so and rfi_hip keep their ABI; only the HIP
# runtime, ROCm libraries (rccl/hipblaslt/rocblas/...), torch build and triton change.
# The upstream tag tcclaviger/vllm:DevQwenNextFlash was removed from Docker Hub on 2026-09-05.
# This is a byte-identical mirror of that image (id 1465c4571b6b, built 2026-08-28). Override
# with --build-arg BASE=... ; tcclaviger/vllm:dev is the upstream successor (see README).
ARG BASE=ghcr.io/davetha/vllm-flashnext:DevQwenNextFlash
FROM ${BASE}
ENV PIP_EXTRA_INDEX_URL=https://stable.repo.amd.com/rocm/whl-next
RUN python3 -m pip install --no-cache-dir "torch[device-gfx1201]==2.11.0+rocm10.0.0" \
      "torchvision==0.26.0+rocm10.0.0" "torchaudio==2.11.0+rocm10.0.0" \
 && python3 -c "import torch; print(torch.__version__, torch.version.hip)"
# Point every ROCm lookup at the pip SDK; the 7.2.3 tree stays on disk but off every path.
ENV SDK=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core
RUN rm -f /opt/rocm && ln -s $SDK /opt/rocm && ls $SDK/lib/libamdhip64.so.7
ENV ROCM_PATH=$SDK ROCM_HOME=$SDK HIP_PATH=$SDK \
    LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/host-math/lib:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib:/usr/local/lib \
    PYTHONPATH=$SDK/share/amd_smi \
    HIP_FORCE_DEV_KERNARG=1
