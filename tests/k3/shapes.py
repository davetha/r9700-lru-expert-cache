import json,struct,glob,re,os
D="/models/q38fn-heretic2-mxfp4-fp8"
pats=[r"\.mlp\.gate$", r"shared_expert_gate", r"indexer", r"linear_attn\.in_proj_a", r"linear_attn\.in_proj_b", r"linear_attn\.in_proj_ba", r"hyper_connection", r"input_mix", r"block_inject", r"fc_embedding", r"fc_hidden", r"^mtp\..*self_attn\..*proj", r"^mtp\..*shared_expert\.", r"^lm_head"]
rx=[re.compile(p) for p in pats]
allw={}
for f in sorted(glob.glob(D+"/model-*.safetensors")):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]
        h=json.loads(fh.read(n))
    for k,v in h.items():
        if k=="__metadata__": continue
        allw[k]=(v["dtype"],v["shape"])
print("total tensors",len(allw))
hits={}
for k,(dt,sh) in allw.items():
    kk=k[:-7] if k.endswith(".weight") else k
    if len(sh)!=2: continue
    if any(r.search(kk) for r in rx):
        hits[k]=(dt,sh)
# restrict layer index 0..2 + mtp + lm_head
def keep(k):
    m=re.search(r"layers\.(\d+)\.",k)
    if m: return int(m.group(1))<=2
    return True
sel={k:v for k,v in hits.items() if keep(k)}
for k in sorted(sel): print(k, sel[k])
print("=== distinct (dtype,shape) ===")
d={}
for k,v in sel.items(): d.setdefault((v[0],tuple(v[1])),[]).append(k)
for kk in sorted(d, key=lambda z:-z[1][0]*z[1][1]): print(kk, len(d[kk]), d[kk][0])
