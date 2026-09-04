import numpy as np, json
d = np.load("/w/artifacts/routes_rank0.npz")
ids, steps = d["ids"], d["steps"]
S, L, W = ids.shape
K = int(d["topk"])
print("shape", ids.shape, "topk", K, "maxrows", int(d["maxrows"]))
# rows per step, from layer 0
valid = (ids >= 0).sum(axis=2)          # [S, L] entries
rows0 = valid[:, 0] // K
print("rows/step layer0: hist", np.bincount(rows0))
# per-layer consistency of row count within a step
print("layers with differing entry counts within a step (frac):",
      float((valid[:, :48] != valid[:, :1]).any(axis=1).mean()))
print("layer48 entries: hist", np.bincount(valid[:, 48] // K))
# distinct experts per layer per step
dis = np.zeros((S, L), dtype=np.int32)
for s in range(S):
    for l in range(L):
        r = ids[s, l]
        dis[s, l] = len(np.unique(r[r >= 0]))
print("distinct/layer/step: mean %.2f  layers0-47 mean %.2f  layer48 mean %.2f"
      % (dis.mean(), dis[:, :48].mean(), dis[:, 48].mean()))
np.save("/w/artifacts/_dis.npy", dis)
np.save("/w/artifacts/_rows.npy", rows0)
# rank0 vs rank1 identical?
e = np.load("/w/artifacts/routes_rank1.npz")
print("rank0 == rank1 ids:", bool((e["ids"] == ids).all()), "steps equal:", bool((e["steps"]==steps).all()))
# row-count transitions (candidate prompt boundaries)
ch = np.flatnonzero(np.diff(rows0.astype(int)) != 0) + 1
print("row-count change points:", len(ch), ch[:60].tolist())
