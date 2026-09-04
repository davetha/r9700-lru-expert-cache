// Device-side LRU expert cache for the r4d hot/cold MoE split (gfx1201 / ROCm 10).
//
// Two kernels, launched once per MoE layer per forward, ahead of moe_align_block_size:
//
//   lru_manage  one workgroup. Marks the experts routed this step, refreshes the LRU
//               stamp of the resident ones, picks victims for the missing ones (argmin
//               stamp over slots whose expert is NOT routed this step, so nothing needed
//               now is ever evicted), rewrites `table` (expert -> slot) and `map_cold`
//               (expert -> itself when not resident) in place, and emits the (expert,
//               slot) miss list the gather then fills.
//
//   lru_gather  copies the six per-expert buffers (w1, w2 and their ws_t/wref scale rows)
//               from the UVA host tensors into their new slots, 16 bytes per lane.
//
// Both are pure device work with no host sync, so they are HIP-graph capturable: every
// pointer is fixed at capture time and only the CONTENTS of table/map_cold/slots change.
//
// Determinism: both TP ranks see identical routing, so the cache must evolve identically.
// Nothing here uses atomics or any other source of run-to-run ordering; misses are
// enumerated in expert-id order via a two-level scan and victims are chosen by a total
// order on (stamp, slot index).
#include <hip/hip_runtime.h>
#include <limits.h>

typedef unsigned int u32;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

#define NT 256
#define SLOT_BITS 20
#define SLOT_MASK ((1 << SLOT_BITS) - 1)

__device__ __forceinline__ int blkSum(int v, int* sh) {
  const int t = threadIdx.x;
  sh[t] = v;
  __syncthreads();
  for (int o = NT >> 1; o; o >>= 1) {
    if (t < o) sh[t] += sh[t + o];
    __syncthreads();
  }
  const int r = sh[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ long long blkMin(long long v, long long* sh) {
  const int t = threadIdx.x;
  sh[t] = v;
  __syncthreads();
  for (int o = NT >> 1; o; o >>= 1) {
    if (t < o && sh[t + o] < sh[t]) sh[t] = sh[t + o];
    __syncthreads();
  }
  const long long r = sh[0];
  __syncthreads();
  return r;
}

// Exclusive scan of one value per thread over the 256-thread block; returns this thread's
// prefix and leaves the block total in sh[NT-1] + its own input (returned via *total).
__device__ __forceinline__ int blkExScan(int v, int* sh, int* total) {
  const int t = threadIdx.x;
  sh[t] = v;
  __syncthreads();
  for (int off = 1; off < NT; off <<= 1) {
    int add = (t >= off) ? sh[t - off] : 0;
    __syncthreads();
    sh[t] += add;
    __syncthreads();
  }
  const int incl = sh[t];
  *total = sh[NT - 1];
  __syncthreads();
  return incl - v;
}

#define MAXS 1024

extern "C" __global__ __launch_bounds__(NT) void lru_manage_k(
    const int* __restrict__ topk_ids,     // [mk] routed expert ids, <0 = padding
    int mk, int E, int S,
    int max_distinct,                     // > this many distinct -> read-through, no inserts
    int max_inserts,                      // hard cap on inserts per call (= miss buffer cap)
    int* __restrict__ table,              // [E]  expert -> slot, or -1
    int* __restrict__ map_cold,           // [E]  expert -> expert when NOT resident, else -1
    int* __restrict__ slot_expert,        // [S]  slot -> expert, or -1
    long long* __restrict__ slot_stamp,   // [S]  slot -> step it was last routed
    unsigned char* __restrict__ routed,   // [E]  scratch
    long long* __restrict__ step,         // [1]
    int* __restrict__ miss,               // [2*max_inserts] (expert, slot) pairs
    int* __restrict__ n_miss) {           // [1]
  __shared__ int sh[NT];
  __shared__ long long shl[NT];
  __shared__ int s_se[MAXS];              // slot -> expert, mirrored so the serial victim
  __shared__ long long s_st[MAXS];        // slot -> stamp,   loop stays in LDS
  __shared__ int s_nins;
  const int t = threadIdx.x;

  if (t == 0) {
    step[0] += 1;
    n_miss[0] = 0;
  }
  for (int i = t; i < E; i += NT) routed[i] = 0;
  __syncthreads();
  const long long st = step[0];

  for (int i = t; i < mk; i += NT) {
    const int e = topk_ids[i];
    if (e >= 0 && e < E) routed[e] = 1;
  }
  __syncthreads();

  // distinct routed experts this step
  int cnt = 0;
  for (int e = t; e < E; e += NT) cnt += routed[e];
  cnt = blkSum(cnt, sh);
  if (cnt > max_distinct) return;   // wide step (prefill chunk / big batch): read through

  // hits refresh their stamp; misses are enumerated in expert-id order so that the
  // (expert -> slot) assignment is identical on every rank
  const int C = (E + NT - 1) / NT;            // experts per thread, contiguous
  const int lo = t * C, hi = min(lo + C, E);
  int local = 0;
  for (int e = lo; e < hi; e++) {
    if (!routed[e]) continue;
    const int s = table[e];
    if (s >= 0) slot_stamp[s] = st;
    else local++;
  }
  __syncthreads();
  int total = 0;
  const int base = blkExScan(local, sh, &total);
  const int nins = min(total, max_inserts);
  if (nins == 0) return;
  {
    int j = base;
    for (int e = lo; e < hi; e++) {
      if (!routed[e] || table[e] >= 0) continue;
      if (j < nins) miss[2 * j] = e;
      j++;
    }
  }
  for (int i = t; i < S; i += NT) {
    s_se[i] = slot_expert[i];
    s_st[i] = slot_stamp[i];
  }
  if (t == 0) s_nins = nins;
  __syncthreads();

  for (int r = 0; r < nins; r++) {
    // victim = argmin (stamp, slot) over slots whose expert is NOT routed this step, so a
    // slot needed by this very step -- including one just filled -- is never taken
    long long best = LLONG_MAX;
    for (int i = t; i < S; i += NT) {
      const int ex = s_se[i];
      if (ex >= 0 && routed[ex]) continue;
      const long long k = (s_st[i] << SLOT_BITS) | (long long)i;
      if (k < best) best = k;
    }
    best = blkMin(best, shl);
    if (best == LLONG_MAX) {                        // nothing evictable left
      if (t == 0) s_nins = r;
      __syncthreads();
      break;
    }
    if (t == 0) {
      const int sl = (int)(best & SLOT_MASK);
      const int e = miss[2 * r];
      const int old = s_se[sl];
      if (old >= 0) {
        table[old] = -1;
        map_cold[old] = old;
      }
      s_se[sl] = e;
      s_st[sl] = st;
      table[e] = sl;
      map_cold[e] = -1;
      miss[2 * r + 1] = sl;
    }
    __syncthreads();
  }

  for (int i = t; i < S; i += NT) {
    slot_expert[i] = s_se[i];
    slot_stamp[i] = s_st[i];
  }
  if (t == 0) n_miss[0] = s_nins;
}

// grid = (chunks, lanes), block = 256. Block (c, j) copies misses j, j+lanes, ... ; slabs
// are split across the `chunks` blocks with a grid-stride loop of 16-byte accesses.
extern "C" __global__ __launch_bounds__(NT) void lru_gather_k(
    u32x4* __restrict__ d0, const u32x4* __restrict__ s0, int v0,
    u32x4* __restrict__ d1, const u32x4* __restrict__ s1, int v1,
    u32x4* __restrict__ d2, const u32x4* __restrict__ s2, int v2,
    u32x4* __restrict__ d3, const u32x4* __restrict__ s3, int v3,
    u32x4* __restrict__ d4, const u32x4* __restrict__ s4, int v4,
    u32x4* __restrict__ d5, const u32x4* __restrict__ s5, int v5,
    const int* __restrict__ miss, const int* __restrict__ n_miss) {
  const int nm = n_miss[0];
  const int stride = gridDim.x * NT;
  const int start = blockIdx.x * NT + threadIdx.x;
  for (int j = blockIdx.y; j < nm; j += gridDim.y) {
    const int e = miss[2 * j];
    const int sl = miss[2 * j + 1];
#define CP(D, S, V)                                                                    \
  {                                                                                    \
    const u32x4* __restrict__ p = (S) + (size_t)e * (size_t)(V);                       \
    u32x4* __restrict__ q = (D) + (size_t)sl * (size_t)(V);                            \
    for (int i = start; i < (V); i += stride) q[i] = p[i];                             \
  }
    CP(d0, s0, v0)
    CP(d1, s1, v1)
    CP(d2, s2, v2)
    CP(d3, s3, v3)
    CP(d4, s4, v4)
    CP(d5, s5, v5)
#undef CP
  }
}


// ---------------------------------------------------------------------------------------
// lru_fused_k: everything lru_manage_k does, PLUS both moe_align_block_size outputs -- the
// resident one (bucketed by `table`, i.e. by SLOT) and the fallback one (bucketed by
// `map_cold`, i.e. by expert id) -- computed from the table it has just updated. That takes
// the per-layer bookkeeping from 6 launches (manage, gather, 2 kernels per align x 2) to 2.
// Measured price of a launch node in a ~3250-node graph: 2.40 us of dispatch (k1/lru/FUSE.md).
//
// Drop-in for moe_align_block_size(topk_ids, block_size, E, <map>, ignore_invalid_experts=1):
// sentinel in sorted_ids is `numel` (= mk), unused expert_ids blocks are -1, a token whose
// mapped id is -1 is skipped, and the value written into expert_ids is the MAPPED id -- on
// the resident side that is the slot index, which is what _gemm_split needs to address the
// hot buffers.
//
// SINGLE EXIT. lru_manage_k's two early returns (read-through, nothing to insert) become
// branches here. An early return would leave sorted_ids/expert_ids/npad holding the PREVIOUS
// graph replay's values: silently wrong output, no crash, visible only on a wide step or a
// step that happened to have no misses.
//
// Ordering within one expert's block is not deterministic in vllm's own kernel (global
// atomicAdd cursors) and cannot change results -- sorted_ids holds the token index, which is
// also the output row, so rows are computed and written independently. We are deterministic
// anyway whenever mk <= NT (which covers decode: mk = 50 at B=1 with MTP-4, 200 at B=4),
// because the two TP ranks staying bit-identical has been worth having; above that we use
// LDS cursors like vllm does.
#define MAXE 1024

// Exclusive scan of TWO values per thread over the block in 3 barriers.
//
// The obvious implementation is two calls to blkExScan (the Hillis-Steele scan the manager
// uses), but that is 8 rounds x 2 __syncthreads() each, twice = 32 barriers, and measured
// at ~7 us/layer it ate half the launch saving the fusion exists to buy. Chunked two-level
// scan instead: 256 partials -> 8 chunks of 32 scanned serially by 8 threads -> 8 chunk
// totals scanned by thread 0. Both scans ride the same barriers.
#define NCHUNK 8
#define CHUNKW (NT / NCHUNK)

__device__ __forceinline__ void blkExScan2(int a, int b, int* oa, int* ob, int* ta, int* tb,
                                           int* sa, int* sb, int* ca, int* cb) {
  const int t = threadIdx.x;
  sa[t] = a;
  sb[t] = b;
  __syncthreads();
  if (t < NCHUNK) {
    int ra = 0, rb = 0;
    const int lo = t * CHUNKW;
    for (int i = lo; i < lo + CHUNKW; i++) {
      const int x = sa[i], y = sb[i];
      sa[i] = ra;
      sb[i] = rb;
      ra += x;
      rb += y;
    }
    ca[t] = ra;
    cb[t] = rb;
  }
  __syncthreads();
  if (t == 0) {
    int ra = 0, rb = 0;
    for (int i = 0; i < NCHUNK; i++) {
      const int x = ca[i], y = cb[i];
      ca[i] = ra;
      cb[i] = rb;
      ra += x;
      rb += y;
    }
    ca[NCHUNK] = ra;
    cb[NCHUNK] = rb;
  }
  __syncthreads();
  *oa = sa[t] + ca[t / CHUNKW];
  *ob = sb[t] + cb[t / CHUNKW];
  *ta = ca[NCHUNK];
  *tb = cb[NCHUNK];
}

// Both block-padded bucket scans at once: cum[i] = sum over j<i of ceil(cnt[j]/bs)*bs,
// with the grand total in cum[nb].
__device__ __forceinline__ void bucketScan2(int nbh, const int* cnt_h, int* cum_h,
                                            int nbc, const int* cnt_c, int* cum_c, int bs,
                                            int* sa, int* sb, int* ca, int* cb) {
  const int t = threadIdx.x;
  const int Ch = (nbh + NT - 1) / NT, Cc = (nbc + NT - 1) / NT;
  const int lh = t * Ch, hh = min(lh + Ch, nbh);
  const int lc = t * Cc, hc = min(lc + Cc, nbc);
  int a = 0, b = 0;
  for (int i = lh; i < hh; i++) a += ((cnt_h[i] + bs - 1) / bs) * bs;
  for (int i = lc; i < hc; i++) b += ((cnt_c[i] + bs - 1) / bs) * bs;
  int ea, eb, ta, tb;
  blkExScan2(a, b, &ea, &eb, &ta, &tb, sa, sb, ca, cb);
  int run = ea;
  for (int i = lh; i < hh; i++) {
    cum_h[i] = run;
    run += ((cnt_h[i] + bs - 1) / bs) * bs;
  }
  run = eb;
  for (int i = lc; i < hc; i++) {
    cum_c[i] = run;
    run += ((cnt_c[i] + bs - 1) / bs) * bs;
  }
  if (t == 0) {
    cum_h[nbh] = ta;
    cum_c[nbc] = tb;
  }
}

__device__ __forceinline__ void fillExpertIds(int nb, const int* __restrict__ cum,
                                              int* __restrict__ eids, int bs) {
  const int t = threadIdx.x;
  const int C = (nb + NT - 1) / NT;
  const int lo = t * C, hi = min(lo + C, nb);
  for (int b = lo; b < hi; b++)
    for (int i = cum[b]; i < cum[b + 1]; i += bs) eids[i / bs] = b;
}

extern "C" __global__ __launch_bounds__(NT) void lru_fused_k(
    const int* __restrict__ topk_ids, int mk, int E, int S,
    int max_distinct, int max_inserts,
    int* __restrict__ table, int* __restrict__ map_cold,
    int* __restrict__ slot_expert, long long* __restrict__ slot_stamp,
    unsigned char* __restrict__ routed, long long* __restrict__ step,
    int* __restrict__ miss, int* __restrict__ n_miss,
    int bs, int L, int NB,                      // block_size, sorted_ids len, expert_ids len
    int* __restrict__ sorted_hot, int* __restrict__ eids_hot, int* __restrict__ npad_hot,
    int* __restrict__ sorted_cold, int* __restrict__ eids_cold, int* __restrict__ npad_cold) {
  __shared__ int sh[NT];
  __shared__ long long shl[NT];
  __shared__ int s_se[MAXS];
  __shared__ long long s_st[MAXS];
  __shared__ int s_nins;
  __shared__ int s_tab[MAXE];                   // table mirrored in LDS for phases 2 and 5
  __shared__ int cnt_h[MAXS], cum_h[MAXS + 1];  // cnt_* double as the placement cursors
  __shared__ int cnt_c[MAXE], cum_c[MAXE + 1];
  __shared__ int sh2[NT], chk_a[NCHUNK + 1], chk_b[NCHUNK + 1];
  __shared__ int s_bkt[NT];                     // per-token bucket key, deterministic path
  const int t = threadIdx.x;

  // ---- phase 0: the manager, unchanged except that its exits are branches --------------
  if (t == 0) {
    step[0] += 1;
    n_miss[0] = 0;
  }
  for (int i = t; i < E; i += NT) routed[i] = 0;
  __syncthreads();
  const long long st = step[0];

  for (int i = t; i < mk; i += NT) {
    const int e = topk_ids[i];
    if (e >= 0 && e < E) routed[e] = 1;
  }
  __syncthreads();

  int cnt = 0;
  for (int e = t; e < E; e += NT) cnt += routed[e];
  cnt = blkSum(cnt, sh);
  bool ins = (cnt <= max_distinct);   // else: wide step (prefill / big batch), read through

  const int C = (E + NT - 1) / NT;
  const int lo = t * C, hi = min(lo + C, E);
  int nins = 0;
  if (ins) {
    int local = 0;
    for (int e = lo; e < hi; e++) {
      if (!routed[e]) continue;
      const int s = table[e];
      if (s >= 0) slot_stamp[s] = st;
      else local++;
    }
    __syncthreads();
    int total = 0;
    const int base = blkExScan(local, sh, &total);
    nins = min(total, max_inserts);
    ins = (nins > 0);                 // uniform: total comes from the block scan
    if (ins) {
      int j = base;
      for (int e = lo; e < hi; e++) {
        if (!routed[e] || table[e] >= 0) continue;
        if (j < nins) miss[2 * j] = e;
        j++;
      }
    }
  }
  if (ins) {
    for (int i = t; i < S; i += NT) {
      s_se[i] = slot_expert[i];
      s_st[i] = slot_stamp[i];
    }
    if (t == 0) s_nins = nins;
    __syncthreads();

    for (int r = 0; r < nins; r++) {
      long long best = LLONG_MAX;
      for (int i = t; i < S; i += NT) {
        const int ex = s_se[i];
        if (ex >= 0 && routed[ex]) continue;
        const long long k = (s_st[i] << SLOT_BITS) | (long long)i;
        if (k < best) best = k;
      }
      best = blkMin(best, shl);
      if (best == LLONG_MAX) {
        if (t == 0) s_nins = r;
        __syncthreads();
        break;
      }
      if (t == 0) {
        const int sl = (int)(best & SLOT_MASK);
        const int e = miss[2 * r];
        const int old = s_se[sl];
        if (old >= 0) {
          table[old] = -1;
          map_cold[old] = old;
        }
        s_se[sl] = e;
        s_st[sl] = st;
        table[e] = sl;
        map_cold[e] = -1;
        miss[2 * r + 1] = sl;
      }
      __syncthreads();
    }

    for (int i = t; i < S; i += NT) {
      slot_expert[i] = s_se[i];
      slot_stamp[i] = s_st[i];
    }
    if (t == 0) n_miss[0] = s_nins;
  }
  __syncthreads();

  // ---- phase 1: mirror the (possibly just-rewritten) table, clear counts, fill outputs --
  for (int i = t; i < E; i += NT) s_tab[i] = table[i];
  for (int i = t; i < S; i += NT) cnt_h[i] = 0;
  for (int i = t; i < E; i += NT) cnt_c[i] = 0;
  for (int i = t; i < L; i += NT) {
    sorted_hot[i] = mk;
    sorted_cold[i] = mk;
  }
  for (int i = t; i < NB; i += NT) {
    eids_hot[i] = -1;
    eids_cold[i] = -1;
  }
  __syncthreads();

  // ---- phase 2: one counting pass feeds both sides ------------------------------------
  // table and map_cold are complementary by construction (test_slots_prod.py check (c)),
  // so "resident -> hot bucket = slot, otherwise -> cold bucket = expert id" reproduces
  // both of vllm's calls at once.
  for (int i = t; i < mk; i += NT) {
    const int e = topk_ids[i];
    if (e < 0 || e >= E) continue;
    const int s = s_tab[e];
    if ((unsigned)s < (unsigned)S) atomicAdd(&cnt_h[s], 1);
    else atomicAdd(&cnt_c[e], 1);
  }
  __syncthreads();

  // ---- phase 3: block-padded exclusive scans -------------------------------------------
  bucketScan2(S, cnt_h, cum_h, E, cnt_c, cum_c, bs, sh, sh2, chk_a, chk_b);
  __syncthreads();
  if (t == 0) {
    npad_hot[0] = cum_h[S];
    npad_cold[0] = cum_c[E];
  }

  // ---- phase 4: expert_ids (hot side carries SLOT indices) ------------------------------
  fillExpertIds(S, cum_h, eids_hot, bs);
  fillExpertIds(E, cum_c, eids_cold, bs);

  // ---- phase 5: placement ---------------------------------------------------------------
  if (mk <= NT) {
    // deterministic: thread t owns token t and counts the matching tokens before it.
    // The keys go to LDS first -- the O(mk) scan below would otherwise be mk*mk global
    // loads (2500 at decode) for no reason.
    if (t < mk) {
      const int e = topk_ids[t];
      int key = -1;
      if (e >= 0 && e < E) {
        const int s = s_tab[e];
        key = ((unsigned)s < (unsigned)S) ? s : (MAXS + e);
      }
      s_bkt[t] = key;
    }
    __syncthreads();
    if (t < mk && s_bkt[t] >= 0) {
      const int key = s_bkt[t];
      int rank = 0;
      for (int j = 0; j < t; j++) rank += (s_bkt[j] == key);
      if (key < MAXS) sorted_hot[cum_h[key] + rank] = t;
      else sorted_cold[cum_c[key - MAXS] + rank] = t;
    }
  } else {
    for (int i = t; i < S; i += NT) cnt_h[i] = cum_h[i];   // cnt_* reused as cursors
    for (int i = t; i < E; i += NT) cnt_c[i] = cum_c[i];
    __syncthreads();
    for (int i = t; i < mk; i += NT) {
      const int e = topk_ids[i];
      if (e < 0 || e >= E) continue;
      const int s = s_tab[e];
      if ((unsigned)s < (unsigned)S) sorted_hot[atomicAdd(&cnt_h[s], 1)] = i;
      else sorted_cold[atomicAdd(&cnt_c[e], 1)] = i;
    }
  }
}

extern "C" {

int r4d_lru_manage(const int* topk_ids, int mk, int E, int S, int max_distinct,
                   int max_inserts, int* table, int* map_cold, int* slot_expert,
                   long long* slot_stamp, unsigned char* routed, long long* step,
                   int* miss, int* n_miss, hipStream_t stream) {
  if (E <= 0 || S <= 0 || S > MAXS || S > SLOT_MASK) return -1;
  hipLaunchKernelGGL(lru_manage_k, dim3(1), dim3(NT), 0, stream, topk_ids, mk, E, S,
                     max_distinct, max_inserts, table, map_cold, slot_expert, slot_stamp,
                     routed, step, miss, n_miss);
  return (int)hipGetLastError();
}

// sizes are BYTES per expert for each of the six buffers; all must be 16-byte multiples.
// Fused manager + both moe_align_block_size outputs. `L` and `NB` must be exactly the
// lengths vllm's python wrapper would have allocated:
//   L  = mk < E ? min(mk*bs, mk + E*(bs-1)) : mk + E*(bs-1)
//   NB = ceil(L / bs)
int r4d_lru_fused(const int* topk_ids, int mk, int E, int S, int max_distinct,
                  int max_inserts, int* table, int* map_cold, int* slot_expert,
                  long long* slot_stamp, unsigned char* routed, long long* step,
                  int* miss, int* n_miss, int bs, int L, int NB,
                  int* sorted_hot, int* eids_hot, int* npad_hot,
                  int* sorted_cold, int* eids_cold, int* npad_cold, hipStream_t stream) {
  if (E <= 0 || E > MAXE || S <= 0 || S > MAXS || S > SLOT_MASK) return -1;
  if (bs <= 0 || L <= 0 || NB <= 0 || mk < 0) return -1;
  hipLaunchKernelGGL(lru_fused_k, dim3(1), dim3(NT), 0, stream, topk_ids, mk, E, S,
                     max_distinct, max_inserts, table, map_cold, slot_expert, slot_stamp,
                     routed, step, miss, n_miss, bs, L, NB, sorted_hot, eids_hot, npad_hot,
                     sorted_cold, eids_cold, npad_cold);
  return (int)hipGetLastError();
}

int r4d_lru_gather(void* d0, const void* s0, long b0, void* d1, const void* s1, long b1,
                   void* d2, const void* s2, long b2, void* d3, const void* s3, long b3,
                   void* d4, const void* s4, long b4, void* d5, const void* s5, long b5,
                   const int* miss, const int* n_miss, int chunks, int lanes,
                   hipStream_t stream) {
  const long bs[6] = {b0, b1, b2, b3, b4, b5};
  for (int i = 0; i < 6; i++)
    if (bs[i] <= 0 || bs[i] % 16) return -1;
  if (chunks <= 0 || lanes <= 0) return -1;
  hipLaunchKernelGGL(lru_gather_k, dim3(chunks, lanes), dim3(NT), 0, stream,
                     (u32x4*)d0, (const u32x4*)s0, (int)(b0 / 16),
                     (u32x4*)d1, (const u32x4*)s1, (int)(b1 / 16),
                     (u32x4*)d2, (const u32x4*)s2, (int)(b2 / 16),
                     (u32x4*)d3, (const u32x4*)s3, (int)(b3 / 16),
                     (u32x4*)d4, (const u32x4*)s4, (int)(b4 / 16),
                     (u32x4*)d5, (const u32x4*)s5, (int)(b5 / 16), miss, n_miss);
  return (int)hipGetLastError();
}

}
