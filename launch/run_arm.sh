#!/usr/bin/env bash
# run_arm.sh <name> <hot_gb> [docker -e args...]
# Relaunch the server with extra env, wait for it to be genuinely ready, profile one
# 200-token decode, stash the trace under prof_<name>/ and print the step breakdown.
#
# Env: REPO_ROOT, PORT, NAME, and everything launch_q38fn_prof.sh takes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$HERE/.." && pwd)}
PORT=${PORT:-8057}
NAME=${NAME:-q38fn-mxfp4}
LOCK=${LOCK:-$REPO_ROOT/gpu.lock}
PROF_DIR=${PROF_DIR:-$REPO_ROOT/prof}
name=$1; hot=$2; shift 2
mkdir -p "$REPO_ROOT/logs" "$PROF_DIR"
cd "$REPO_ROOT"

EXTRA_DOCKER_ARGS="$*" flock -w 600 "$LOCK" "$HERE/launch_q38fn_prof.sh" "$hot" || exit 1
for i in $(seq 1 60); do
  sleep 10
  curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ] || {
    echo "ARM $name: server DIED"; docker logs "$NAME" > "logs/$name.startup.log" 2>&1
    grep -E "Error" "logs/$name.startup.log" | tail -3; exit 1; }
done
# /v1/models can answer before the engine finishes profiling; wait for the KV-cache sizing line
for i in $(seq 1 90); do
  docker logs "$NAME" > "logs/$name.startup.log" 2>&1
  grep -q "GPU KV cache size" "logs/$name.startup.log" && break; sleep 10
done
sleep 5
docker logs "$NAME" > "logs/$name.startup.log" 2>&1   # the next arm replaces the container
# fail loud if the r4d MoE path silently fell back (an ImportError in a mounted patch did this once)
if grep -q "r4d unavailable" "logs/$name.startup.log"; then
  echo "ARM $name: R4D FALLBACK ENGAGED - results invalid"
  grep -m1 "r4d unavailable" "logs/$name.startup.log" | cut -c1-300; exit 1; fi
grep -q "hot experts: budget" "logs/$name.startup.log" || {
  echo "ARM $name: NO HOT EXPERTS BUILT - results invalid"; exit 1; }

rm -rf "${PROF_DIR:?}"/*
flock -w 900 "$LOCK" python3 bench/prof_run.py \
  "Write a detailed essay about the history of the Roman aqueducts and their engineering." 200 || exit 1
for i in $(seq 1 30); do
  sleep 5
  ls "$PROF_DIR"/dp0_pp0_tp1* >/dev/null 2>&1 && \
    ! docker logs "$NAME" 2>&1 | tail -3 | grep -q "Stopping profiler" && break
done
sleep 20; mkdir -p "prof_$name"; mv "$PROF_DIR"/* "prof_$name"/
echo "=== ARM $name ($*)"
python3 bench/prof_step.py "prof_$name"/dp0_pp0_tp0*.gz | sed -n 3,12p
