// hcq_fp8skinny.hip -- hcq_gemm_fp8blk_nt_m16: small-M fp8 block-scaled skinny GEMM for gfx1201.
//
//   C[M,N] (bf16) = (A[M,K] * As[M,K/128]) @ (W[N,K] * Ws[N/128,K/128])^T,  M <= 16
//
// A is fp8-e4m3 quantised per token group of 128 along K (vllm's per_token_group_quant_fp8, row
// major scales), W is fp8-e4m3 with one fp32 scale per 128x128 block.  This is the operand set
// the closed fp8hip_gemm_w8a8_tiled kernel takes, byte for byte, INCLUDING the pre-shuffled
// weight: shuffle_weight_gfx1201 already lays the weight out in gfx1201 WMMA fragment order, so
// this kernel reads the SAME tensor the closed one does and no second copy exists in VRAM.
//
// WHY IT EXISTS.  At decode M the closed kernel launches exactly N/128 workgroups -- 20 of them
// for the 2560-wide gdn/attention out_proj on a 64-CU part.  Measured on the production trace
// (prof_eager_h15, rank 0):
//
//     shape (per rank)          grid   calls/step   us/call   weight MB   GB/s   % of 644.6
//     in_proj_qkvz  N=8192 K=2560   64      36        42.46      20.97     494      77%
//     qkv_proj      N=6656 K=2560   52      12        36.48      17.04     467      72%
//     out/o_proj    N=2560 K=3072   20      48        26.03       7.86     302      47%
//
// The 20-workgroup shape leaves 44 CUs idle and lands at less than half the part's bandwidth.
// These GEMMs are pure weight streaming at decode M (A is 15 KB, the scales 2 KB), so the fix is
// simply more workgroups: one 16-wide N tile per wave instead of one 128-wide tile per workgroup,
// times a k-split, which is the decomposition every skinny GEMM in libr4d uses.
//
// THE WEIGHT LAYOUT IS THE SHUFFLE, READ AS b128.  shuffle_weight_gfx1201 is
//     w.view(N/16, 16, K/32, 2, 2, 8).permute(0, 2, 4, 1, 3, 5)
// i.e. byte (n, k) with n = 16*nt + r, k = 32*k32 + 16*d3 + 8*d4 + c sits at
//     ((nt*(K/32) + k32)*2 + d4)*256 + r*16 + d3*8 + c.
// A wave32 fp8 WMMA fragment wants lane l to hold row (l&15) and the eight k of one 16-step
// starting at 8*(l>>4) -- so d4 is exactly (l>>4) and lane l's SIXTEEN contiguous bytes at
// r*16 are the d3=0 fragment (bytes 0..7) and the d3=1 fragment (bytes 8..15) of the same
// 32-K step.  One global_load_b128 per lane per 32 K feeds two WMMA, and the wave reads 512
// contiguous bytes.  Nothing is staged in LDS.
//
// THE BLOCK SCALES ARE APPLIED PER 128-K CHUNK, NOT PER ELEMENT.  Both scales are constant over
// a 128x128 tile, so
//     sum_k a_q[m,k] as[m,kb] w_q[n,k] ws[nb,kb]  =  sum_kb as[m,kb]*ws[nb,kb] * sum_{k in kb} a_q w_q
// and the inner sum is eight WMMA into one fp32 accumulator.  That is a regrouping of the same
// products, not an approximation; what differs from any other kernel is the ASSOCIATION, so this
// is not expected to be bit-identical to fp8hip (measured agreement is in the bench).
// A 16-row tile always lies inside one 128-row scale block, so ws is one scalar per (tile, kb).
// as varies with m, i.e. across the eight accumulator elements of a lane; it is staged into LDS
// TRANSPOSED at entry so a lane reads its eight values as two ds_read_b128 rather than eight
// strided global loads per chunk.
//
// SPLIT-K IS IN THE WORKGROUP, reduced through LDS in wave order, as everywhere else in this
// family: no HBM partial buffer, no counter, nothing to register with a caller, and the result
// does not depend on how the waves were scheduled -- so it is HIP-graph replayable.
//
// Rows past M are CLAMPED on the A load and their activation scale is zeroed, so the inner loop
// is branch-free and the epilogue drops them.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define WAVE32 32
#define HCQ_MAX_M 16

typedef int   v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

// NPW = 16-wide N tiles owned by one wave. WV waves cover N, SK waves split K; a block is
// WV*SK waves and owns WV*NPW tiles.
template <int NPW>
__global__ __launch_bounds__(1024) void hcq_gemm_fp8blk_nt_m16_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const float* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  extern __shared__ float sh[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nlane = lane & 15, half = lane >> 4;
  const int nkb = K >> 7;                  // 128-K scale blocks
  const int nk32 = K >> 5;                 // 32-K steps
  const int nkbs = nkb / SK;               // blocks per k-split
  const int kb0 = ksplit * nkbs;
  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;

  float* red = sh;                                    // ncols*SK*256 floats
  float* ast = sh + (size_t)ncols * SK * 256;         // 16*nkb floats, [kb][m]

  // As, transposed. Rows past M are zeroed rather than clamped: a clamped row would add a real
  // product into an accumulator element the store drops, which is harmless, but zero also keeps
  // any garbage in the padding out of the fp32 sum.
  for (int i = tid; i < 16 * nkb; i += nthreads) {
    const int m = i & 15, kb = i >> 4;
    ast[i] = (m < M) ? As[(size_t)m * nkb + kb] : 0.f;
  }
  __syncthreads();

  // Column tiles this wave owns, CLAMPED to the last real tile: grid.x rounds N/16 up to a
  // multiple of ncols, so the tail block addresses tiles that do not exist. Clamping the TILE
  // matters because the weight is indexed by tile -- an unclamped index is a memory fault, not a
  // wrong answer. The epilogue still drops those columns.
  const int ntmax = N / 16 - 1;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  int wtile[NPW], wsrow[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
    wsrow[j] = (wtile[j] * 16) >> 7;       // the 128-row scale block this tile sits in
  }

  const uint4* __restrict__ Wp = (const uint4*)W;
  const int arow = nlane < M ? nlane : M - 1;
  const unsigned char* __restrict__ Ar = A + (size_t)arow * K + half * 8;

  v8f tot[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) tot[j] = (v8f)(0.f);

  for (int kb = kb0; kb < kb0 + nkbs; kb++) {
    v8f acc[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) acc[j] = (v8f)(0.f);

#pragma unroll
    for (int q = 0; q < 4; q++) {                     // 4 * 32 K = one 128-K scale block
      const int k32 = kb * 4 + q;
      const unsigned int* pa = (const unsigned int*)(Ar + (size_t)k32 * 32);
      v2i a0, a1;
      a0[0] = (int)pa[0];  a0[1] = (int)pa[1];        // d3 = 0
      a1[0] = (int)pa[4];  a1[1] = (int)pa[5];        // d3 = 1, +16 bytes
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const uint4 wv = Wp[(((size_t)wtile[j] * nk32 + k32) * 2 + half) * 16 + nlane];
        v2i w0, w1;
        w0[0] = (int)wv.x;  w0[1] = (int)wv.y;
        w1[0] = (int)wv.z;  w1[1] = (int)wv.w;
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, acc[j]);
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, acc[j]);
      }
    }

    const float* asp = ast + kb * 16 + half * 8;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const float wsv = Ws[(size_t)wsrow[j] * nkb + kb];
#pragma unroll
      for (int e = 0; e < 8; e++) tot[j][e] += acc[j][e] * (wsv * asp[e]);
    }
  }

  // Reduce the SK partial tiles in LDS. The write is e*32+lane so the 32 lanes of a wave hit 32
  // distinct banks; the sum runs over every thread in the block in a fixed split order.
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
    for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = tot[j][e];
  }
  __syncthreads();
  for (int idx = tid; idx < ncols * 256; idx += nthreads) {
    const int col = idx >> 8, off = idx & 255;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
    const int l = off & 31;
    const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
    const int m = 8 * (l >> 4) + (off >> 5);
    if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
  }
}

// Negative return codes rather than exceptions: this is called through ctypes, and an exception
// crossing that boundary terminates the process instead of falling back.
extern "C" int hcq_gemm_fp8blk_nt_m16(void* a, void* as, void* w, void* ws, void* c,
                                      int M, int K, int N, int WV, int SK, int NPW,
                                      void* stream) {
  if (M < 1 || M > HCQ_MAX_M) return -1;
  if (N % 128) return -2;                       // ws rows and the tile clamp both assume it
  if (K % 128) return -3;
  if (SK < 1 || (K / 128) % SK) return -4;
  if (WV < 1 || WV * SK * WAVE32 > 1024) return -5;
  if (NPW != 1 && NPW != 2 && NPW != 4) return -6;
  const int ntiles = N / 16, ncols = WV * NPW;
  const size_t smem = ((size_t)ncols * SK * 256 + (size_t)16 * (K / 128)) * sizeof(float);
  if (smem > 64 * 1024) return -7;
  dim3 grid((ntiles + ncols - 1) / ncols), block(WV * SK * WAVE32);
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)as;
  auto Wp = (const unsigned char*)w;
  auto Zp = (const float*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(NPv) hipLaunchKernelGGL((hcq_gemm_fp8blk_nt_m16_kernel<NPv>), grid, block, smem, \
                                       st, Ap, Sp, Wp, Zp, Cp, M, K, N, WV, SK)
  switch (NPW) {
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 4: LAUNCH(4); break;
    default: return -6;
  }
#undef LAUNCH
  return 0;
}

extern "C" int hcq_gemm_fp8blk_nt_m16_max_m(void) { return HCQ_MAX_M; }

// ---------------------------------------------------------------------------------------------
// Streaming-read control. The GEMM above is pure weight streaming at decode M, so the only
// honest ceiling to compare it against is what a kernel that ONLY reads the same bytes, in the
// same b128 pattern, out of the same pool, achieves in the same graph. torch.sum on a uint8 view
// is not that control -- it measures a reduction, not the read.
__global__ __launch_bounds__(256) void hcq_stream_probe_kernel(
    const uint4* __restrict__ W, size_t n16, unsigned int* __restrict__ out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned int acc = 0;
  for (; i < n16; i += stride) {
    const uint4 v = W[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0xdeadbeefu) out[0] = acc;     // never taken; keeps every load live
}

extern "C" int hcq_stream_probe(void* w, long long bytes, int wgs, void* out, void* stream) {
  if (bytes % 16) return -1;
  if (wgs < 1) return -2;
  hipLaunchKernelGGL(hcq_stream_probe_kernel, dim3(wgs), dim3(256), 0, (hipStream_t)stream,
                     (const uint4*)w, (size_t)(bytes / 16), (unsigned int*)out);
  return 0;
}
