import gzip,json,glob,sys,statistics as st,collections
fn=sys.argv[1] if len(sys.argv)>1 else glob.glob("prof/dp0_pp0_tp0*.gz")[0]
ev=json.load(gzip.open(fn,"rt"))["traceEvents"]
k=[e for e in ev if e.get("cat")=="kernel"]
moe=sorted(e["dur"] for e in k if "r4d_gemm_moe" in e["name"])
n=len(moe); q=lambda p: moe[int(p*(n-1))]
if n: print("moe calls",n,"min",moe[0],"p10",q(.1),"p25",q(.25),"p50",q(.5),"p75",q(.75),"p90",q(.9),"max",moe[-1],"mean",round(st.mean(moe),1))
g=[e for e in k if "r4d_gemm_moe" in e["name"]][:2]
print([(e["args"].get("grid"),e["args"].get("block")) for e in g])
ann=[e for e in ev if e.get("cat")=="gpu_user_annotation" and "generation" in e.get("name","")]
ann.sort(key=lambda e:e["ts"])
print("gen annotations",len(ann),"dur ms p50",st.median(e["dur"] for e in ann)/1e3)
a=ann[len(ann)//2]; lo,hi=a["ts"],a["ts"]+a["dur"]
fam=collections.Counter(); cnt=collections.Counter()
for e in k:
    if lo<=e["ts"]<hi:
        nm=e["name"].split("<")[0][:44]; fam[nm]+=e["dur"]; cnt[nm]+=1
busy=sum(fam.values())
print("one step: annotation %.1f ms, kernel busy %.1f ms" % (a["dur"]/1e3, busy/1e3))
for nm,d in fam.most_common(16): print("  %6.2f ms %4dx %s" % (d/1e3, cnt[nm], nm))
ar=[e["dur"] for e in k if "r4d_ar_" in e["name"]]
if ar and ann: print("AR calls/step",round(len(ar)/len(ann),1),"mean us",round(st.mean(ar),1))
