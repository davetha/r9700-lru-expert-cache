"""Sanity: is my router index == the profile's layer index? If the axes were permuted the
static baseline would be junk. Overlap of measured top-257 vs profile top-257, diagonal vs
best off-diagonal."""
import json, numpy as np
d = np.load("/w/artifacts/routes_rank0.npz"); ids = d["ids"][:, :48]; K = int(d["topk"])
prof = json.load(open("/hp/hot_profile.json"))["layers"]
L, E, N = 48, 512, 257
cnt = np.zeros((L, E), np.int64)
for li in range(L):
    r = ids[:, li].ravel(); r = r[r >= 0]
    cnt[li] = np.bincount(r, minlength=E)
meas = [set(np.argsort(-cnt[li], kind="stable")[:N].tolist()) for li in range(L)]
pset = [set(prof[str(li)]["ranked"][:N]) for li in range(L)]
diag = np.array([len(meas[i] & pset[i]) / N for i in range(L)])
M = np.array([[len(meas[i] & pset[j]) / N for j in range(L)] for i in range(L)])
off = M - np.diag(np.diag(M)) - np.eye(L)
print("diagonal (my router i vs profile layer i) overlap: mean %.3f min %.3f max %.3f"
      % (diag.mean(), diag.min(), diag.max()))
print("best OFF-diagonal overlap:                        mean %.3f max %.3f"
      % (off.max(axis=1).mean(), off.max()))
print("argmax of each row == i for %d/%d layers" % (int((M.argmax(axis=1) == np.arange(L)).sum()), L))
# coverage the profile hot set gets on the measured counts, routing-weighted, per layer
gb = 15; per = 1_228_800; budget = int(gb*1e9)//per
lay, rk, co = [], [], []
for li in range(L):
    c = np.asarray(prof[str(li)]["counts"], float)
    lay.append(np.full(E, li)); rk.append(np.asarray(prof[str(li)]["ranked"])); co.append(c/c.sum())
lay, rk, co = map(np.concatenate, (lay, rk, co))
keep = np.argsort(-co, kind="stable")[:budget]
hot = {li: [] for li in range(L)}
for li, e in zip(lay[keep].tolist(), rk[keep].tolist()): hot[li].append(e)
cov = np.array([cnt[li][hot[li]].sum()/cnt[li].sum() for li in range(L)])
print("\nprofile-estimated coverage @15GB: %.3f (in-sample, from _hot_sets)" % (co[keep].sum()/L))
print("MEASURED routing-weighted coverage of that same set on this traffic: mean %.3f "
      "min %.3f max %.3f" % (cov.mean(), cov.min(), cov.max()))
print("=> measured cold fraction %.3f vs predicted %.3f (%.2fx)"
      % (1-cov.mean(), 1-co[keep].sum()/L, (1-cov.mean())/(1-co[keep].sum()/L)))
