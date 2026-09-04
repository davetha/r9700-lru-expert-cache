import json,struct,glob,re
D="/models/q38fn-heretic2-mxfp4-fp8"
pats=[r"\.mlp\.gate$", r"shared_expert_gate", r"indexer", r"linear_attn\.in_proj_a", r"linear_attn\.in_proj_b", r"linear_attn\.in_proj_ba", r"hyper_connection", r"input_mix", r"block_inject", r"fc_embedding", r"fc_hidden", r"^mtp\..*self_attn\..*proj", r"^mtp\..*shared_expert\.", r"^lm_head", r"\.ple\."]
rx=[re.compile(p) for p in pats]
allw={}
for f in sorted(glob.glob(D+"/model-*.safetensors")):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]
        h=json.loads(fh.read(n))
    for k,v in h.items():
        if k!="__metadata__": allw[k]=(v["dtype"],v["shape"])
d={}
for k,(dt,sh) in allw.items():
    kk=k[:-7] if k.endswith(".weight") else k
    if len(sh)!=2 or dt!="BF16": continue
    if any(r.search(kk) for r in rx):
        d.setdefault(tuple(sh),[]).append(k)
print("shape  count  example  (generic-name-count)")
for kk in sorted(d, key=lambda z:-z[0]*z[1]):
    names=d[kk]
    gen=sorted({re.sub(r"\.(\d+)\.",".N.",re.sub(r"layers\.\d+","layers.N",n)) for n in names})
    print(kk, len(names), gen[:6])
