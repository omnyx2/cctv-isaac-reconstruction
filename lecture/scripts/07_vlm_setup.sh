#!/usr/bin/env bash
# Step 7: get the VLM (vision-language model) ready. It looks at a picture and answers in words.
#   ./07_vlm_setup.sh                 -> qwen3-vl:30b-a3b-instruct (4-bit, ~19 GB download, ~20 GB memory)
#   ./07_vlm_setup.sh qwen2.5vl:7b    -> smaller and lighter (~6 GB)
#   ./07_vlm_setup.sh all             -> every model in VLM_MODELS (env.sh, ~55 GB), tests the default one
# Uses the Ollama already installed for this user (no system install); starts its server if it is off.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_isaac; need_project
MODEL="${1:-$VLM_MODEL}"
PULL=("$MODEL")
if [ "$MODEL" = all ]; then MODEL="$VLM_MODEL"; PULL=("${VLM_MODELS[@]}"); fi
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
LOG="$WORK/lecture/ollama.log"

say "Ollama"
command -v ollama >/dev/null || { fail "ollama is not installed: see https://ollama.com/download"; exit 1; }
if curl -fs "$OLLAMA_URL/api/tags" >/dev/null; then
  ok "server already running at $OLLAMA_URL"
else
  # One model in memory at a time: two big models together could run this machine out of memory.
  # Context 8k: otherwise qwen3-vl reserves memory for 262k tokens (45 GB instead of 20 GB).
  OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=8192 nohup ollama serve >"$LOG" 2>&1 &
  for _ in $(seq 30); do curl -fs "$OLLAMA_URL/api/tags" >/dev/null && break; sleep 1; done
  curl -fs "$OLLAMA_URL/api/tags" >/dev/null || { fail "server did not start, see $LOG"; exit 1; }
  ok "server started (log: $LOG)"
fi

for m in "${PULL[@]}"; do
  say "Model $m"
  if ollama list | awk 'NR>1{print $1}' | grep -qx "$m"; then ok "already downloaded"
  else ollama pull "$m" && ok "downloaded"; fi
done
ollama list

say "Test: ask the VLM about 3 frames of MVI_20011 (first answer is slow: the model loads)"
cd "$PROJECT"
OUT="$PROJECT/data/results/vlm/lecture_test.jsonl"
"$ISAAC_SIM/python.sh" tools/vlm_analyze.py --config config/detrac_MVI_20011.yaml \
  --model "$MODEL" --interval 10 --out "$OUT" 2>&1 | grep -vE "^\[|Warning"
ok "answers saved to $OUT"
