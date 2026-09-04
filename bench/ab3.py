"""Best-of-3 decode bench reporting tok/s AND ms/step (wall / spec-decode drafts), so kernel
A/Bs are not confounded by text-dependent MTP acceptance. Saves texts + metrics to argv[1]."""
import json, sys, time, urllib.request
BASE = "http://127.0.0.1:8057/v1"; OUT = sys.argv[1]
MID = json.load(urllib.request.urlopen(BASE + "/models", timeout=60))["data"][0]["id"]
def metrics():
    m = {}
    for line in urllib.request.urlopen("http://127.0.0.1:8057/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("vllm:spec_decode_num_") and not line.startswith("#"):
            k, v = line.rsplit(" ", 1); m[k.split("{")[0]] = m.get(k.split("{")[0], 0) + float(v)
    return m
CASES = [("prose", "Write a detailed technical explanation of how a B-tree index works in a database.", 256),
         ("JSON", "Output a JSON array of 200 objects with fields id,name,category,price,in_stock. Only JSON.", 800),
         ("code", "Write a complete Python module implementing an LRU cache with TTL expiry, type hints and docstrings.", 600)]
res = {}
for label, p, n in CASES:
    best_tps, best_ms, texts, accs, tps_list = 0, 1e9, [], [], []
    for _ in range(3):
        m0 = metrics()
        req = urllib.request.Request(BASE + "/completions", data=json.dumps({"model": MID, "prompt": p, "max_tokens": n, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); w = time.time() - t0
        m1 = metrics(); acc = {k: m1[k] - m0.get(k, 0) for k in m1}
        steps = acc.get("vllm:spec_decode_num_drafts_total", 0)
        drafts, accepted = acc.get("vllm:spec_decode_num_draft_tokens_total", 0), acc.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        u = d["usage"]; tps = u["completion_tokens"] / w; tps_list.append(round(tps, 1))
        best_tps = max(best_tps, tps); texts.append(d["choices"][0]["text"])
        if steps: best_ms = min(best_ms, 1000 * w / steps); accs.append(round(accepted / drafts, 3) if drafts else None)
    res[label] = {"tps": round(best_tps, 1), "tps_runs": tps_list, "ms_per_step": round(best_ms, 2), "tok_per_step": round(u["completion_tokens"] / steps, 2) if steps else None,
                  "accept_rate": accs[-1] if accs else None, "texts": texts, "same_text_across_runs": len(set(texts)) == 1, "completion_tokens": u["completion_tokens"]}
    print(f"  {label:6} {best_tps:6.1f} tok/s  {best_ms:6.2f} ms/step  tok/step={res[label]['tok_per_step']}  accept={res[label]['accept_rate']}  stable={res[label]['same_text_across_runs']}  runs={tps_list}")
json.dump(res, open(OUT, "w"))

# prefill probe: ~16K-token prompt, 1 output token -> prompt tokens / wall (prefix cache defeated by a nonce)
import random
words = ("the quick brown fox jumps over the lazy dog while engineers tune kernels for decode throughput ").split()
nonce = str(random.random())
long_prompt = nonce + " " + " ".join(random.choice(words) for _ in range(12500))
req = urllib.request.Request(BASE + "/completions", data=json.dumps({"model": MID, "prompt": long_prompt, "max_tokens": 1, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); w = time.time() - t0
pt = d["usage"]["prompt_tokens"]
res["prefill"] = {"prompt_tokens": pt, "wall_s": round(w, 2), "tok_per_s": round(pt / w, 1)}
print(f"  prefill {pt} tokens in {w:.2f}s = {pt/w:.0f} tok/s")
json.dump(res, open(OUT, "w"))
