import locality_sim as L
prof = L.load_profile()
tr = L.synth_traces(prof, nseg=1)[0]
li = 0
nx = L.belady_nexts(tr, li)
print("steps", len(tr), "distinct/step", len(tr[0][li]))
for cap in (50, 100, 254, 339, 500):
    b = L.Belady(cap, nexts=nx)
    m = n = 0
    sizes = []
    for t, s in enumerate(tr):
        mm, _ = b.step(s[li])
        if t > 50:
            m += mm
            n += len(s[li])
        sizes.append(len(b.res))
    print("cap=%d miss=%.2f%% final_res=%d max_res=%d" % (cap, 100*m/n, sizes[-1], max(sizes)))
