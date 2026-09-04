"""Mixed code/prose/json traffic for the routing capture; logs the spec-decode step counter
before/after each request so the trace can be split per prompt."""
import json, time, urllib.request
B = "http://127.0.0.1:8057/v1"
MID = json.load(urllib.request.urlopen(B + "/models", timeout=60))["data"][0]["id"]
def steps():
    for line in urllib.request.urlopen("http://127.0.0.1:8057/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            return float(line.rsplit(" ", 1)[1])
    return -1
REQS = [
 ("code1", "Write a complete Python module implementing an LRU cache with TTL expiry, type hints and docstrings.", 500),
 ("prose1", "Write a detailed essay about the history of the Roman aqueducts and their engineering.", 500),
 ("json1", "Output a JSON array of 150 objects with fields id,name,category,price,in_stock. Only JSON.", 500),
 ("code2", "Write a C++ class for a thread-safe bounded queue using std::mutex and condition variables, with comments and a small usage example.", 500),
 ("prose2", "Explain, for a curious teenager, how vaccines train the immune system. Be detailed and friendly.", 500),
 ("json2", "Produce a JSON schema for a REST API that manages library books, members and loans, with descriptions for every field. JSON only.", 500),
]
log = []
for name, p, n in REQS:
    s0 = steps(); t0 = time.time()
    req = urllib.request.Request(B + "/completions", data=json.dumps({"model": MID, "prompt": p, "max_tokens": n, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900)); w = time.time() - t0
    s1 = steps(); u = d["usage"]
    log.append({"name": name, "steps": [s0, s1], "completion_tokens": u["completion_tokens"], "tps": round(u["completion_tokens"] / w, 1)})
    print(f"{name:7} steps {s0:.0f}-{s1:.0f}  tok={u['completion_tokens']}  {u['completion_tokens']/w:.1f} tok/s", flush=True)
json.dump(log, open("$REPO_ROOT/artifacts/traffic_boundaries.json", "w"), indent=1)
