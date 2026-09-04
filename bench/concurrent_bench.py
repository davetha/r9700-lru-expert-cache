"""Concurrency A/B for the LRU expert cache: fire B simultaneous, DIFFERENT-prompt streams and
report aggregate + per-stream tok/s and the engine's ms/step.

Why not just tok/s: with MTP the text drives the accept rate, and kernel changes change the
text, so tok/s is not a clean kernel metric (see ab3.py). ms/step is. Getting ms/step right
under concurrency needs care: `vllm:spec_decode_num_drafts_total` counts ONE draft per RUNNING
REQUEST per engine step, so with B requests in flight it advances by ~B per step. A background
sampler therefore reads drafts together with `vllm:num_requests_running` and converts:

    engine_steps(window) = delta_drafts / running

and only windows where `running` is stable at exactly B (both endpoints) are used for the
headline "steady" number, so the ramp-up/ramp-down tail cannot bias it.

    python3 concurrent_bench.py out.json            # B = 1,2,4
    B=1,2,4,8 N=500 REPS=2 python3 concurrent_bench.py out.json

Env: BASE (default http://127.0.0.1:8057), B, N (max_tokens/stream), REPS, SAMPLE (s),
     WARMUP=1 (one throwaway short request first).
Run it under the GPU lock, on an otherwise idle server, once per arm (LRU=1 and LRU=0).
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8057")
API = BASE + "/v1"
BS = [int(v) for v in os.environ.get("B", "1,2,4").split(",")]
NTOK = int(os.environ.get("N", "500"))
REPS = int(os.environ.get("REPS", "1"))
SAMPLE = float(os.environ.get("SAMPLE", "0.25"))
WARMUP = os.environ.get("WARMUP", "1") == "1"
OUT = sys.argv[1] if len(sys.argv) > 1 else "concurrent.json"

# 8 distinct prompts, mixed code/prose/json, cycled so every concurrent stream is different
# (identical prompts would share a prefix-cache block and route identically -- the exact
# opposite of the union-of-cold-experts effect we are trying to measure).
PROMPTS = [
    ("code1", "Write a complete Python module implementing an LRU cache with TTL expiry, type hints and docstrings."),
    ("prose1", "Write a detailed essay about the history of the Roman aqueducts and their engineering."),
    ("json1", "Output a JSON array of 150 objects with fields id,name,category,price,in_stock. Only JSON."),
    ("code2", "Write a C++ class for a thread-safe bounded queue using std::mutex and condition variables, with comments and a small usage example."),
    ("prose2", "Explain, for a curious teenager, how vaccines train the immune system. Be detailed and friendly."),
    ("json2", "Produce a JSON schema for a REST API that manages library books, members and loans, with descriptions for every field. JSON only."),
    ("code3", "Write a Rust implementation of a bounded MPSC channel using a ring buffer and atomics, with unit tests."),
    ("prose3", "Write a detailed technical explanation of how a B-tree index works in a database, including splits and merges."),
]

MID = json.load(urllib.request.urlopen(API + "/models", timeout=120))["data"][0]["id"]


def scrape():
    """-> {metric: value}; counters summed over label sets, gauges summed over engines."""
    m = {}
    body = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    for line in body.splitlines():
        if not line.startswith("vllm:") or line.startswith("#"):
            continue
        k, _, v = line.rpartition(" ")
        k = k.split("{")[0]
        if k.startswith("vllm:spec_decode_num_") or k in ("vllm:num_requests_running",
                                                          "vllm:num_requests_waiting"):
            try:
                m[k] = m.get(k, 0.0) + float(v)
            except ValueError:
                pass
    return m


class Sampler(threading.Thread):
    """Samples (t, drafts, draft_tokens, accepted, running) until stopped."""
    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                m = scrape()
                self.rows.append((time.time(),
                                  m.get("vllm:spec_decode_num_drafts_total", 0.0),
                                  m.get("vllm:spec_decode_num_draft_tokens_total", 0.0),
                                  m.get("vllm:spec_decode_num_accepted_tokens_total", 0.0),
                                  m.get("vllm:num_requests_running", -1.0)))
            except Exception as e:                       # fail loud, keep sampling
                print(f"    !! metrics scrape failed: {e}", file=sys.stderr)
            self.stop.wait(SAMPLE)

    def engine_steps(self, B):
        """-> (steps_all, secs_all, steps_steady, secs_steady, n_windows_steady).

        A window contributes delta_drafts/running engine steps. `steady` keeps only windows
        whose running count is exactly B at BOTH endpoints.
        """
        sa = ta = ss = ts = 0.0
        nw = 0
        for (t0, d0, _, _, r0), (t1, d1, _, _, r1) in zip(self.rows, self.rows[1:]):
            dd, dt = d1 - d0, t1 - t0
            if dd <= 0 or dt <= 0:
                continue
            r = max(1.0, (r0 + r1) / 2.0) if r0 >= 0 else float(B)
            sa += dd / r
            ta += dt
            if r0 == B and r1 == B:
                ss += dd / B
                ts += dt
                nw += 1
        return sa, ta, ss, ts, nw


def stream(label, prompt, res, barrier):
    """One completion stream. Streaming only so we get TTFT; token COUNT comes from the
    server's usage block, never from counting SSE chunks (vLLM coalesces ~2.7 tok/chunk
    under MTP, so chunk counting understates the rate)."""
    body = json.dumps({"model": MID, "prompt": prompt, "max_tokens": NTOK, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(API + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.time()
    ttft, usage, text = None, None, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            raw = raw.strip()
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:]
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("choices") and d["choices"][0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                text.append(d["choices"][0]["text"])
            if d.get("usage"):
                usage = d["usage"]
    w = time.time() - t0
    if usage is None:
        raise RuntimeError(f"{label}: no usage block -- server did not honour include_usage")
    res[label] = {"tokens": usage["completion_tokens"], "wall_s": round(w, 3),
                  "ttft_s": round(ttft, 3) if ttft else None,
                  "tok_s": round(usage["completion_tokens"] / w, 2),
                  "t_start": t0, "t_end": t0 + w, "text_sha": hash("".join(text)) & 0xffffffff}


def run_one(B, rep):
    res, barrier = {}, threading.Barrier(B)
    picks = [PROMPTS[(rep * B + i) % len(PROMPTS)] for i in range(B)]
    m0 = scrape()
    smp = Sampler()
    smp.start()
    time.sleep(2 * SAMPLE)                               # a couple of idle samples first
    ths = [threading.Thread(target=stream, args=(f"{lbl}#{i}", p, res, barrier))
           for i, (lbl, p) in enumerate(picks)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    time.sleep(2 * SAMPLE)
    smp.stop.set()
    smp.join()
    m1 = scrape()

    if len(res) != B:
        raise RuntimeError(f"B={B}: only {len(res)}/{B} streams returned")
    toks = sum(v["tokens"] for v in res.values())
    span = max(v["t_end"] for v in res.values()) - min(v["t_start"] for v in res.values())
    sa, ta, ss, ts, nw = smp.engine_steps(B)
    drafts = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    dtok = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    out = {
        "B": B, "rep": rep, "n_tokens_requested": NTOK,
        "streams": {k: {kk: vv for kk, vv in v.items() if kk not in ("t_start", "t_end")}
                    for k, v in sorted(res.items())},
        "agg_tok_s": round(toks / span, 2), "agg_tokens": toks, "span_s": round(span, 3),
        "ms_per_step_all": round(1000 * ta / sa, 3) if sa else None,
        "ms_per_step_steady": round(1000 * ts / ss, 3) if ss else None,
        "steady_windows": nw, "steady_s": round(ts, 2),
        "engine_steps_all": round(sa, 1),
        "tok_per_step": round(toks / sa, 3) if sa else None,
        "accept_rate": round(acc / dtok, 4) if dtok else None,
        "request_steps_total": drafts,
    }
    per = " ".join(f"{k}={v['tok_s']:.1f}" for k, v in sorted(res.items()))
    print(f"  B={B} rep{rep}  agg {out['agg_tok_s']:7.1f} tok/s   ms/step steady "
          f"{out['ms_per_step_steady']} (all {out['ms_per_step_all']}, {nw} steady windows / "
          f"{out['steady_s']}s)   tok/step {out['tok_per_step']}   accept {out['accept_rate']}",
          flush=True)
    print(f"       per-stream tok/s: {per}", flush=True)
    print(f"       ttft: " + " ".join(f"{k}={v['ttft_s']}" for k, v in sorted(res.items())),
          flush=True)
    return out


def main():
    print(f"model {MID}  base {BASE}  N={NTOK} tok/stream  B={BS}  reps={REPS}", flush=True)
    if WARMUP:
        r = {}
        stream("warmup", "Say hello.", r, threading.Barrier(1))
        print(f"  warmup: {r['warmup']['tokens']} tok in {r['warmup']['wall_s']}s", flush=True)
    runs = []
    for B in BS:
        for rep in range(REPS):
            runs.append(run_one(B, rep))
            json.dump(runs, open(OUT, "w"), indent=1)
    print("\n  %-4s %12s %14s %14s %10s %9s" %
          ("B", "agg tok/s", "ms/step steady", "ms/step all", "tok/step", "accept"))
    for r in runs:
        print("  %-4d %12.1f %14s %14s %10s %9s" %
              (r["B"], r["agg_tok_s"], r["ms_per_step_steady"], r["ms_per_step_all"],
               r["tok_per_step"], r["accept_rate"]))
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
