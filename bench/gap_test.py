import torch, time, os
dev = torch.device('cuda:0')
x = torch.ones(1024, device=dev); y = torch.zeros(1024, device=dev)
def body(n):
    for _ in range(n): y.add_(x)     # 1 tiny kernel each
def timeit(fn, iters=20):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters
N = 1000
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    body(10); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): body(N)
torch.cuda.synchronize()
eager = timeit(lambda: body(N), 5)
graph = timeit(lambda: g.replay(), 20)
print(f'env HIP_FORCE_DEV_KERNARG={os.environ.get("HIP_FORCE_DEV_KERNARG")} GPU_MAX_HW_QUEUES={os.environ.get("GPU_MAX_HW_QUEUES")} extra={os.environ.get("EXTRA_NOTE")}')
print(f'eager {eager*1e6/N:.2f} us/kernel   graph replay {graph*1e6/N:.2f} us/kernel  (N={N} tiny add_ kernels)')
# kernel time itself
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    g.replay(); torch.cuda.synchronize()
ks = [e for e in p.events() if e.device_type.name == 'CUDA']
d = sorted(e.time_range.elapsed_us() for e in ks)
print(f'profiler: {len(ks)} kernels, median kernel {d[len(d)//2]:.2f} us; sum {sum(d):.0f} us')
