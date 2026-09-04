// Cold-expert gather kernels for the hot/cold MoE residency study.
//
// Each "slab" is one expert's contiguous MXFP4 weight block (gate_up 819200 B,
// down 409600 B). `src` is a UVA device pointer onto PINNED HOST memory holding
// all E slabs; `idx` selects the ones this call needs; `dst` is a compact VRAM
// staging buffer of nsel slabs.
#include <hip/hip_runtime.h>

typedef unsigned int uint;
// HIP's uint4 is a class; the nontemporal builtins need a native vector type.
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// grid = (chunks, nsel), block = threads. Every thread issues 16-byte loads with a
// grid-stride inner loop, so `chunks * threads` sets how many PCIe reads are in flight.
template <bool NT>
__global__ __launch_bounds__(256) void gather_k(u32x4* __restrict__ dst,
                                                const u32x4* __restrict__ src,
                                                const int* __restrict__ idx,
                                                int vec_per_slab) {
  const size_t s = blockIdx.y;
  const u32x4* __restrict__ p = src + (size_t)idx[s] * (size_t)vec_per_slab;
  u32x4* __restrict__ q = dst + s * (size_t)vec_per_slab;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_per_slab; i += stride) {
    if (NT) {
      u32x4 v = __builtin_nontemporal_load(&p[i]);
      __builtin_nontemporal_store(v, &q[i]);
    } else {
      q[i] = p[i];
    }
  }
}

extern "C" {

// nt: 0 = plain loads/stores, 1 = nontemporal (bypass L2 on the store side).
int cold_gather(void* dst, const void* src, const int* idx, int nsel,
                long slab_bytes, int chunks, int threads, int nt, hipStream_t stream) {
  if (slab_bytes % 16) return -1;
  int vps = (int)(slab_bytes / 16);
  dim3 grid(chunks, nsel, 1);
  dim3 blk(threads, 1, 1);
  if (nt)
    hipLaunchKernelGGL((gather_k<true>), grid, blk, 0, stream,
                       (u32x4*)dst, (const u32x4*)src, idx, vps);
  else
    hipLaunchKernelGGL((gather_k<false>), grid, blk, 0, stream,
                       (u32x4*)dst, (const u32x4*)src, idx, vps);
  return (int)hipGetLastError();
}

}
