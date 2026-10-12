#!/usr/bin/env bash
# Benchmark the remaining big models one at a time: download -> measure -> next. Nothing runs in
# parallel. Two OOM safeguards (an OOM killed the whole desktop container on 2026-10-10 22:47):
#   1. before loading: skip a model whose estimated memory would leave less than MIN_FREE_GB free
#   2. while running: tools/mem_guard.sh stops the run as soon as free memory drops under MIN_FREE_GB
# Models that were stopped or skipped are written to results/skipped.txt and never rerun.
#   ./run_remaining.sh            progress -> results/queue.log
cd "$(dirname "$0")"
MODELS=(qwen3-vl:30b-a3b-instruct-q8_0 qwen2.5vl:72b llama3.2-vision:90b llama4:scout)
export MIN_FREE_GB=${MIN_FREE_GB:-16}
GUARD="$(cd ../tools && pwd)/mem_guard.sh"
LOG=results/queue.log
URL=http://localhost:11434

say() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
skip() { say "SKIP $1 ($2)"; echo "$1: $2, not rerun" >>results/skipped.txt; }
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
# model files + 15% + 4GB for the 8192-token KV cache and compute buffers (32b used 18.4 + 2.0 + ~1)
need_gb() {
  python3 -I - "$1" <<'EOF'
import json, os, sys
name, _, tag = sys.argv[1].partition(":")
m = json.load(open(os.path.expanduser(f"~/.ollama/models/manifests/registry.ollama.ai/library/{name}/{tag or 'latest'}")))
print(int(sum(l["size"] for l in m["layers"]) / 1e9 * 1.15 + 4 + 0.5))
EOF
}

if ! curl -fs "$URL/api/tags" >/dev/null; then
  OLLAMA_MAX_LOADED_MODELS=1 setsid nohup ollama serve >>../ollama.log 2>&1 &
  for _ in $(seq 30); do curl -fs "$URL/api/tags" >/dev/null && break; sleep 1; done
  say "ollama server started (OLLAMA_MAX_LOADED_MODELS=1)"
fi

for m in "${MODELS[@]}"; do
  f=${m//[:\/]/_}
  if [ -s "results/$f.jsonl" ]; then say "DONE ALREADY $m"; continue; fi
  if grep -q "^$m:" results/skipped.txt 2>/dev/null; then say "SKIPPED BEFORE $m"; continue; fi
  start=$(date +%s)
  if ! ollama pull "$m" >/dev/null 2>&1; then skip "$m" "download failed"; continue; fi
  say "PULLED $m $(( $(date +%s) - start ))s"

  need=$(need_gb "$m"); free=$(free_gb)
  if [ $(( free - need )) -lt "$MIN_FREE_GB" ]; then
    skip "$m" "estimated ${need}GB would leave under ${MIN_FREE_GB}GB free (free ${free}GB)"; continue
  fi
  say "MEASURING $m (estimated ${need}GB, free ${free}GB)"
  "$GUARD" python3 -I vlm_bench.py "$m" >"results/log_$f.txt" 2>&1
  rc=$?
  if [ $rc -eq 99 ]; then
    mkdir -p results/aborted_oom; mv "results/$f.jsonl" "results/log_$f.txt" results/aborted_oom/ 2>/dev/null
    skip "$m" "stopped by the memory guard (free memory under ${MIN_FREE_GB}GB)"
  elif [ $rc -eq 0 ]; then say "BENCHED $m"
  else say "BENCH FAILED $m (see results/log_$f.txt)"; fi
done
python3 -I vlm_bench.py --report >/dev/null && say "REPORT DONE"
