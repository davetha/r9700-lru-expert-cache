// Read-only control for the closed MoE grouped GEMM's hot path: touch exactly the bytes the
// GEMM touches (one contiguous slab per selected slot) with 16 B loads and nothing else, so
// the GEMM can be held to a number this access pattern can actually reach on this part.
#include <hip/hip_runtime.h>
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// grid = (chunks, nsel). Slot s reads idx[s]'s slab; the XOR sink keeps the loads live.
__global__ __launch_bounds__(256) void moe_read_k(const u32x4* __restrict__ src,
                                                  const int* __restrict__ idx,
                                                  int vec_per_slab, unsigned* __restrict__ sink) {
  const u32x4* __restrict__ p = src + (size_t)idx[blockIdx.y] * (size_t)vec_per_slab;
  const int stride = gridDim.x * blockDim.x;
  u32x4 acc = {0u, 0u, 0u, 0u};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_per_slab; i += stride)
    acc ^= p[i];
  const unsigned v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (v == 0x9e3779b9u) sink[0] = v;   // practically never; keeps the compiler honest
}

extern "C" int moe_read_probe(const void* src, const int* idx, int nsel, long slab_bytes,
                              int chunks, int threads, unsigned* sink, hipStream_t stream) {
  if (slab_bytes % 16) return -1;
  dim3 grid(chunks, nsel, 1), blk(threads, 1, 1);
  hipLaunchKernelGGL(moe_read_k, grid, blk, 0, stream, (const u32x4*)src, idx,
                     (int)(slab_bytes / 16), sink);
  return (int)hipGetLastError();
}
