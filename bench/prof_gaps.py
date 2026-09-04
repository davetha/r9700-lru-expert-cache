import gzip,json,glob,sys,collections
fn=sys.argv[1] if len(sys.argv)>1 else glob.glob("prof/dp0_pp0_tp0*.gz")[0]
ev=json.load(gzip.open(fn,"rt"))["traceEvents"]
k=sorted((e for e in ev if e.get("cat") in("kernel","gpu_memcpy","gpu_memset")), key=lambda e:e["ts"])
ann=sorted((e for e in ev if e.get("cat")=="gpu_user_annotation" and "generation" in e.get("name","")), key=lambda e:e["ts"])
a=ann[len(ann)//2]; lo,hi=a["ts"],a["ts"]+a["dur"]
ks=[e for e in k if lo<=e["ts"]<hi]
cpu=[e for e in ev if e.get("cat") in ("cpu_op","user_annotation","cuda_runtime","python_function") and e.get("ph")=="X" and e["ts"]<hi and e["ts"]+e.get("dur",0)>lo]
gaps=[]; prev_end=lo; prev_name="(step start)"
for e in ks:
    g=e["ts"]-prev_end
    if g>120: gaps.append((g,prev_end,e["ts"],prev_name,e["name"].split("<")[0][:50]))
    prev_end=max(prev_end,e["ts"]+e["dur"]); prev_name=e["name"].split("<")[0][:50]
gaps.sort(reverse=True)
print("step %.1f ms, %d gaps>120us totalling %.1f ms" % (a["dur"]/1e3, len(gaps), sum(g[0] for g in gaps)/1e3))
for g,s,t,pn,nn in gaps[:12]:
    print("\nGAP %.0f us after [%s] before [%s]" % (g,pn,nn))
    inside=[c for c in cpu if c["ts"]<t and c["ts"]+c["dur"]>s]
    # prefer smaller (leaf-ish) events that mostly lie inside the gap
    inside.sort(key=lambda c: -min(c["ts"]+c["dur"],t)+max(c["ts"],s))
    seen=set()
    for c in inside[:14]:
        nm=c["name"][:90]
        if nm in seen: continue
        seen.add(nm)
        print("   %-14s %7.0f us  %s" % (c.get("cat"), min(c["ts"]+c["dur"],t)-max(c["ts"],s), nm))
# also: cuda_runtime sync calls in the step
rt=collections.Counter(); rtd=collections.Counter()
for c in cpu:
    if c.get("cat")=="cuda_runtime": rt[c["name"]]+=1; rtd[c["name"]]+=c["dur"]
print("\nruntime calls in step:", ", ".join("%s x%d %.0fus"%(n,rt[n],rtd[n]) for n,_ in rt.most_common(12)))
