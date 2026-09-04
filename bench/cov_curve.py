import json, numpy as np
prof=json.load(open("/h/hot_profile.json"))["layers"]
per_expert = 15.5e9/12613
L=len(prof); cs=[]
for li,v in prof.items():
    c=np.asarray(v["counts"],float); cs.append(c/c.sum())
allc=np.sort(np.concatenate(cs))[::-1]
cum=np.cumsum(allc)/L
print("per-expert bytes %.0f  total experts/rank %d"%(per_expert, len(allc)))
print(" GB   experts  coverage  cold_frac  cold_rel_to_15GB")
base=None
for gb in [10,12,13,14,15,15.5,16,17,18,19,20,22,24,28,32]:
    n=int(gb*1e9//per_expert); n=min(n,len(allc)); cov=cum[n-1]
    if gb==15: base=1-cov
    print(" %4.1f  %6d   %.3f     %.3f     %s"%(gb,n,cov,1-cov, ("%.2fx"%((1-cov)/base)) if base else ""))
