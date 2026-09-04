#!/usr/bin/env python3
"""expert_counts.npz -> hot_profile.json for VLLM_R4D_HOT_PROFILE.
Decode routings summed over domains (decode is the PCIe-bound phase). The npz layer axis
is in string-sorted order (0,1,10,11,...): row i is real layer sorted(range(48), key=str)[i]."""
import json, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
z = np.load(src)
dec = sum(z[k] for k in z.files if k.endswith('__decode'))
pre = sum(z[k] for k in z.files if k.endswith('__prefill'))
L, E = dec.shape
real = sorted(range(L), key=str)
layers = {}
for i in range(L):
    c = dec[i]
    order = np.argsort(-c, kind='stable')
    layers[str(real[i])] = {'ranked': order.tolist(), 'counts': c[order].tolist()}
tot_dec = int(dec.sum() // (L * 10)); tot_pre = int(pre.sum() // (L * 10))
json.dump({'source': src, 'decode_tokens': tot_dec, 'prefill_tokens': tot_pre,
           'layers': layers}, open(dst, 'w'))
print(f'wrote {dst}: {L} layers x {E} experts, from {tot_dec} decode tokens')
