"""Confirm the shipped pick_cfg table on the public API at the production shape."""
import json, os, time, torch
import vllm.model_executor.layers.utils as U
import vllm.model_executor.kernels.draft_w4_lmhead as W

dev, N, K = "cuda:0", 124160, 2560
torch.manual_seed(0)
w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
t0 = time.perf_counter(); packed = W.pack_w4a16(w); torch.cuda.synchronize()
print(f"pack {time.perf_counter()-t0:.2f}s  {packed.nbytes/2**20:.1f} MB", flush=True)


def timeit(fn, calls=5, reps=7):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / calls)
    ts.sort(); del g
    return ts[len(ts) // 2]


out = {}
for n in (1, 2, 4, 5, 20):
    x = torch.randn(n, K, device=dev).bfloat16()
    cfg = W.pick_cfg(min(n, W.MAX_M + 1), packed.n_pad, packed.k)
    t_w4 = timeit(lambda: W.gemm_w4a16(x, packed))
    t_bf = timeit(lambda: U.rocm_unquantized_gemm_impl(x, w, None))
    out[n] = {"cfg": cfg, "w4_us": t_w4, "bf16_us": t_bf, "x": t_bf / t_w4}
    print(f"n={n:3d} cfg={cfg} w4 {t_w4:7.1f}us  bf16 {t_bf:7.1f}us  {t_bf/t_w4:.2f}x",
          flush=True)
json.dump(out, open("/w/tests/k3/w4_confirm.json", "w"), indent=1, default=str)
print("DONE", flush=True)
