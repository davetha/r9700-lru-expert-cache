#!/usr/bin/env python3
"""Drive one profiled decode on :8057: warm, /start_profile, N-token completion, /stop_profile.
Prints wall/tokens so the kernel-sum can be checked against real step time."""
import json, sys, time, urllib.request
B = 'http://127.0.0.1:8057'
PROMPT = sys.argv[1] if len(sys.argv) > 1 else 'Write a detailed essay about the history of the Roman aqueducts and their engineering.'
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 200
def post(path, body):
    r = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.loads(f.read() or b'{}')
def gen(n):
    t = time.time()
    o = post('/v1/chat/completions', {'model': 'q38fn-mxfp4', 'messages': [{'role': 'user', 'content': PROMPT}],
                                      'max_tokens': n, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
    dt = time.time() - t
    u = o['usage']
    return dt, u['completion_tokens'], u['prompt_tokens']
dt, ct, pt = gen(64); print(f'warm: {ct} tok in {dt:.2f}s  ({ct/dt:.1f} tok/s) prompt={pt}', flush=True)
post('/start_profile', {})
t0 = time.time()
dt, ct, pt = gen(MAXTOK)
post('/stop_profile', {})
print(f'PROFILED: {ct} tok in {dt:.2f}s = {ct/dt:.1f} tok/s (incl prefill of {pt}); profile window {time.time()-t0:.2f}s', flush=True)
