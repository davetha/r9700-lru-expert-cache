#!/usr/bin/env python3
"""Read safetensors headers only (no tensor data) from a sharded checkpoint dir.
Pure stdlib -- no safetensors/torch dependency needed."""
import json, struct, sys, glob, os

DTYPE_BYTES = {
    'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
    'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1,
    'F8_E4M3': 1, 'F8_E5M2': 1, 'BOOL': 1,
}

def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop('__metadata__', None)
    return hdr

def main(d):
    files = sorted(glob.glob(os.path.join(d, '*.safetensors')))
    all_t = {}
    for fp in files:
        hdr = read_header(fp)
        for k, v in hdr.items():
            all_t[k] = v
    out = {}
    for k, v in all_t.items():
        dt = v['dtype']
        shape = v['shape']
        nbytes_elem = DTYPE_BYTES[dt]
        numel = 1
        for s in shape:
            numel *= s
        out[k] = {'dtype': dt, 'shape': shape, 'bytes': numel * nbytes_elem}
    json.dump(out, sys.stdout)

if __name__ == '__main__':
    main(sys.argv[1])
