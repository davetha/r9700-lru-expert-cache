"""Is the ~28 GB/s host->device ceiling the PCIe link or one DMA queue?

Links report Gen5 x16 (32 GT/s x16, ~55-58 GB/s usable), so 28 GB/s is suspiciously close to
half. Split one large contiguous copy across k streams and see whether it scales.
"""
import ctypes, os, time
import torch

c_vp, c_sz, c_i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
hip = ctypes.CDLL("libamdhip64.so.7")
hip.hipMemcpyAsync.argtypes = [c_vp, c_vp, c_sz, c_i, c_vp]
hip.hipMemcpyAsync.restype = c_i
NB = int(os.environ.get("MB", "256")) * 2**20
DEVS = [int(v) for v in os.environ.get("DEVS", "0").split(",")]


def run(dev, nstream, kind, reps=5):
    with torch.cuda.device(dev):
        h = torch.empty(NB, dtype=torch.uint8, pin_memory=True)
        d = torch.empty(NB, dtype=torch.uint8, device=f"cuda:{dev}")
        sts = [torch.cuda.Stream(device=dev) for _ in range(nstream)]
        chunk = NB // nstream
        hp, dp = h.data_ptr(), d.data_ptr()
        src, dst = (hp, dp) if kind == 1 else (dp, hp)

        def go():
            for i, s in enumerate(sts):
                off = i * chunk
                n = chunk if i < nstream - 1 else NB - off
                hip.hipMemcpyAsync(c_vp((dst + off)), c_vp((src + off)), c_sz(n), kind,
                                   c_vp(s.cuda_stream))
        go(); torch.cuda.synchronize()
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter(); go(); torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        del h, d
        torch.cuda.empty_cache()
        return NB / best / 1e9


print(f"buffer {NB/2**20:.0f} MiB, devices {DEVS}")
for kind, nm in ((1, "H2D"), (2, "D2H")):
    for ns in (1, 2, 4, 8):
        gb = [run(d, ns, kind) for d in DEVS]
        print(f"  {nm} streams={ns}: " + "  ".join(f"cuda:{d} {g:6.1f} GB/s" for d, g in zip(DEVS, gb)))
