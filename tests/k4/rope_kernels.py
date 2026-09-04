"""Kernel count for the mrope cos/sin fetch: chunk()+contiguous vs pre-split cache."""
import torch
from torch.profiler import ProfilerActivity, profile

dev = "cuda"
RD, MAXPOS = 64, 262144
cache = torch.randn(MAXPOS, RD, dtype=torch.bfloat16, device=dev)
half = RD // 2
cos_c, sin_c = cache[..., :half].contiguous(), cache[..., half:].contiguous()


def count(fn):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return sum(1 for e in p.events()
               if e.device_type == torch.autograd.DeviceType.CUDA and e.device_index >= 0)


for n in (5, 32, 512):
    pos = torch.randint(0, MAXPOS, (3, n), device=dev)

    def old():
        cs = cache[pos]
        c, s = cs.chunk(2, dim=-1)
        return c.contiguous(), s.contiguous()

    def new():
        return cos_c[pos], sin_c[pos]

    a, b = old(), new()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    print("tokens=%4d  chunk+contiguous %d kernels -> split cache %d kernels"
          % (n, count(old), count(new)))
print("extra VRAM for the split halves: %.1f MB" % (cache.numel() * 2 / 2**20))
