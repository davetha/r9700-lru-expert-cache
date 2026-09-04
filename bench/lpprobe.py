"""Teacher-forced numerics probe: prompt_logprobs of fixed texts on :8057 -> argv[1] (json).
Compare two servers' files with: python3 lpprobe.py cmp a.json b.json"""
import json, sys, urllib.request, math
B = "http://127.0.0.1:8057/v1"
if sys.argv[1] == "cmp":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    for k in a:
        la, lb = a[k], b[k]; n = min(len(la), len(lb))
        d = [abs(x - y) for x, y in zip(la[:n], lb[:n]) if x is not None and y is not None]
        print(f"{k:8} tokens={n} max|dlogprob|={max(d):.4f} mean={sum(d)/len(d):.5f} identical={sum(1 for x in d if x==0)}/{len(d)}")
    sys.exit()
MID = json.load(urllib.request.urlopen(B + "/models", timeout=60))["data"][0]["id"]
texts = {
 "prose": open("$REPO_ROOT/artifacts/ab3_baseline.json") and json.load(open("$REPO_ROOT/artifacts/ab3_baseline.json"))["prose"]["texts"][0],
 "code": json.load(open("$REPO_ROOT/artifacts/ab3_baseline.json"))["code"]["texts"][0],
 "json": json.load(open("$REPO_ROOT/artifacts/ab3_baseline.json"))["JSON"]["texts"][0],
}
out = {}
for k, t in texts.items():
    req = urllib.request.Request(B + "/completions", data=json.dumps({"model": MID, "prompt": "Text:\n" + t, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    pl = d["choices"][0]["prompt_logprobs"]
    lp = [None if e is None else list(e.values())[0]["logprob"] for e in pl]
    out[k] = lp
    print(k, "tokens", len(lp), "sum logprob", round(sum(x for x in lp if x is not None), 2))
json.dump(out, open(sys.argv[1], "w"))
