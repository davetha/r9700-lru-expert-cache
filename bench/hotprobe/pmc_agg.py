#!/usr/bin/env python3
"""Aggregate rocprofv3 counter_collection CSVs: mean counter value per (kernel, counter)."""
import csv, glob, sys, collections, statistics
acc = collections.defaultdict(list)
for fn in glob.glob(sys.argv[1] + "/**/*counter_collection.csv", recursive=True):
    with open(fn) as f:
        for row in csv.DictReader(f):
            k = row.get("Kernel_Name", "")[:60]
            if "moe_mxfp4a8" in k or "moe_read_k" in k:
                acc[(k, row["Counter_Name"])].append(float(row["Counter_Value"]))
for (k, c), v in sorted(acc.items()):
    print(f"{k[:40]:40s} {c:24s} n={len(v):4d} mean={statistics.mean(v):14.1f} med={statistics.median(v):14.1f}")
