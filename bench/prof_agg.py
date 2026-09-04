#!/usr/bin/env python3
"""Aggregate a kineto/torch-profiler JSON trace: GPU kernel time by name, per rank file.
Usage: prof_agg.py <trace.json[.gz]> [topN]"""
import gzip, json, sys, re, collections
fn = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
op = gzip.open if fn.endswith('.gz') else open
with op(fn, 'rt') as f:
    d = json.load(f)
ev = d['traceEvents'] if isinstance(d, dict) else d
kern = collections.Counter(); cnt = collections.Counter()
cats = collections.Counter()
t_min, t_max = 1e30, 0
for e in ev:
    if e.get('ph') != 'X':
        continue
    cat = e.get('cat', '')
    cats[cat] += e.get('dur', 0)
    if cat in ('kernel', 'gpu_memcpy', 'gpu_memset', 'Kernel', 'Memcpy', 'Memset'):
        n = e['name']
        n = re.sub(r'<.*', '', n)[:110]
        kern[n] += e['dur']; cnt[n] += 1
        t_min = min(t_min, e['ts']); t_max = max(t_max, e['ts'] + e['dur'])
tot = sum(kern.values())
print(f'file={fn}\ncategories(us): ' + ', '.join(f'{k}={v/1e3:.1f}ms' for k, v in cats.most_common(8)))
print(f'GPU-kernel span {(t_max-t_min)/1e3:.1f} ms, kernel busy {tot/1e3:.1f} ms ({100*tot/max(t_max-t_min,1):.1f}% of span)')
print(f'{"us":>10} {"%":>6} {"calls":>7} {"us/call":>8}  name')
for n, us in kern.most_common(top):
    print(f'{us:10.0f} {100*us/tot:6.2f} {cnt[n]:7d} {us/cnt[n]:8.1f}  {n}')
