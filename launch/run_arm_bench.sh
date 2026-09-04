#!/usr/bin/env bash
# run_arm_bench.sh <name> <hot_gb> [docker -e args...]
# Relaunch with extra env (MTP_N and friends pass through), wait for readiness, then run
# bench/ab3.py best-of-3 (tok/s + ms/step) and save ab3_<name>.json.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$HERE/.." && pwd)}
PORT=${PORT:-8057}
NAME=${NAME:-q38fn-mxfp4}
LOCK=${LOCK:-$REPO_ROOT/gpu.lock}
name=$1; hot=$2; shift 2
mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

EXTRA_DOCKER_ARGS="$*" flock -w 600 "$LOCK" "$HERE/launch_q38fn_prof.sh" "$hot" || exit 1
for i in $(seq 1 60); do
  sleep 10
  curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ] || {
    echo "ARM $name: server DIED"; docker logs "$NAME" > "logs/$name.startup.log" 2>&1
    grep -E "Error" "logs/$name.startup.log" | tail -3; exit 1; }
done
for i in $(seq 1 90); do
  docker logs "$NAME" > "logs/$name.startup.log" 2>&1
  grep -q "GPU KV cache size" "logs/$name.startup.log" && break; sleep 10
done
sleep 5
docker logs "$NAME" > "logs/$name.startup.log" 2>&1
if grep -q "r4d unavailable" "logs/$name.startup.log"; then
  echo "ARM $name: R4D FALLBACK ENGAGED - results invalid"
  grep -m1 "r4d unavailable" "logs/$name.startup.log" | cut -c1-300; exit 1; fi
grep -q "hot experts: budget" "logs/$name.startup.log" || {
  echo "ARM $name: NO HOT EXPERTS BUILT - results invalid"; exit 1; }
echo "=== ARM $name (MTP_N=${MTP_N:-4}; $*)"
flock -w 900 "$LOCK" python3 bench/ab3.py "ab3_$name.json"
