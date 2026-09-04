#!/usr/bin/env bash
# rocprofv3 hardware counters on the closed MoE GEMM and the read control, eager mode.
# One rocprofv3 run per counter group (groups that cannot be co-collected fail loudly).
set -uo pipefail
cd /w/k1/hotprobe
ln -sf $SDK/lib/libhsa-amd-aqlprofile64.so.1 /usr/lib/libhsa-amd-aqlprofile64.so; export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-}
export MODE=eager NREP=${NREP:-3}
groups=(
  "GRBM_GUI_ACTIVE FETCH_SIZE"
  "GRBM_GUI_ACTIVE SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_WAIT_ANY"
  "GRBM_GUI_ACTIVE TA_TA_BUSY SQ_INST_CYCLES_VALU SQ_INSTS_VALU"
  "GL2C_HIT GL2C_MISS GL2C_EA_WRREQ_STALL GL2C_EA_RDREQ"
  "MeanOccupancyPerCU"
)
i=0
for gsp in "${groups[@]}"; do
  i=$((i+1)); rm -rf pmc_$i
  echo "=== group $i: $gsp"
  rocprofv3 --pmc $gsp --output-format csv -d pmc_$i -o run -- python3 moe_hot_probe.py > pmc_$i.log 2>&1
  echo "rc=$?"; grep -E "error|Error|ERROR" pmc_$i.log | head -3
  ls pmc_$i/*counter_collection.csv 2>/dev/null | head -2
done
