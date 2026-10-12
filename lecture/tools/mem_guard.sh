#!/usr/bin/env bash
# Run a heavy command under a memory guard. This machine (DGX Spark, CPU and GPU share ~121GB) runs
# the desktop in one container: an OOM kills everything (desktop, terminals, Claude, Ollama), as
# on 2026-10-10 22:47 and 23:54. The guard stops the command first, before memory runs out.
#   MIN_FREE_GB=16 tools/mem_guard.sh <command> [args...]
#   tools/mem_guard.sh --free      (print the free memory the guard sees, in GB)
# Exit code 99 = stopped by the guard (do not just rerun it). Otherwise the command's own exit code.
#
# "Free" is the smallest of: /proc/meminfo MemAvailable, and for every cgroup above us that has a
# memory limit, limit - usage + inactive file cache. The 23:54 OOM happened with MemAvailable at
# 115GB: the real limit was kubepods.slice memory.max (~117.7GiB, shared with other pods).
MIN=${MIN_FREE_GB:-16}
CG=/sys/fs/cgroup
LIMITED=()
rel=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
while [ -n "$rel" ] && [ "$rel" != "/" ]; do
  [ -r "$CG$rel/memory.max" ] && [ "$(cat "$CG$rel/memory.max")" != max ] && LIMITED+=("$CG$rel")
  rel=$(dirname "$rel")
done
free_gb() {
  local best d max cur inact
  best=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  for d in "${LIMITED[@]}"; do
    max=$(cat "$d/memory.max"); cur=$(cat "$d/memory.current")
    inact=$(awk '$1 == "inactive_file" {print $2}' "$d/memory.stat")
    local g=$(( (max - cur + inact) / 1073741824 ))
    [ "$g" -lt "$best" ] && best=$g
  done
  echo "$best"
}
[ "${1:-}" = --free ] && { free_gb; exit 0; }

"$@" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  if [ "$(free_gb)" -lt "$MIN" ]; then
    echo "mem_guard: free memory $(free_gb)GB < ${MIN}GB, stopping: $*" >&2
    pkill -TERM -P $pid; kill -TERM $pid 2>/dev/null
    # A model being loaded lives in Ollama's runner process, not in the command itself.
    for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop "$m" >/dev/null 2>&1; done
    pkill -TERM -f "ollama runner|llama-server"
    wait $pid 2>/dev/null
    exit 99
  fi
  sleep 0.5
done
wait $pid
