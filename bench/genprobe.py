"""Greedy decode with logprobs on :8057 -> argv[1]; `cmp a b` compares top-1 logprob per position
until the first token divergence, giving the numeric noise level of the decode path."""
import json, sys, urllib.request
B = "http://127.0.0.1:8057/v1"
P = ["Write a detailed technical explanation of how a B-tree index works in a database.",
     "Write a complete Python module implementing an LRU cache with TTL expiry, type hints and docstrings."]
if sys.argv[1] == "cmp":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    for i, (ra, rb) in enumerate(zip(a, b)):
        ta, tb = ra["tokens"], rb["tokens"]; n = 0
        while n < min(len(ta), len(tb)) and ta[n] == tb[n]: n += 1
        d = [abs(x - y) for x, y in zip(ra["lp"][:n], rb["lp"][:n])]
        print(f"prompt{i}: identical prefix {n}/{min(len(ta),len(tb))} tokens; over prefix max|dlp|={max(d) if d else 0:.4f} mean={sum(d)/max(len(d),1):.5f}")
    sys.exit()
MID = json.load(urllib.request.urlopen(B + "/models", timeout=60))["data"][0]["id"]
out = []
for p in P:
    req = urllib.request.Request(B + "/completions", data=json.dumps({"model": MID, "prompt": p, "max_tokens": 300, "temperature": 0, "logprobs": 1}).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
    out.append({"tokens": d["logprobs"]["tokens"], "lp": d["logprobs"]["token_logprobs"]})
    print(len(d["logprobs"]["tokens"]), "tokens")
json.dump(out, open(sys.argv[1], "w"))
