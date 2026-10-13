#!/usr/bin/env bash
# Step 7: get the VLM (vision-language model) ready. It looks at a picture and answers in words.
#   ./07_vlm_setup.sh                 -> qwen3-vl:30b-a3b-instruct (4-bit, ~19 GB download, ~20 GB memory)
#   ./07_vlm_setup.sh qwen2.5vl:7b    -> smaller and lighter (~6 GB)
#   ./07_vlm_setup.sh all             -> every model in VLM_MODELS (env.sh, ~55 GB), tests the default one
# Uses the Ollama already installed for this user (no system install); starts its server if it is off.
# On our DGX Spark the model files are copied from the shared folder (PUBLIC_LAB/ollama_models, filled once
# by the teacher with 03_spark_public.sh fill) instead of downloaded: same command, no internet needed.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_isaac; need_project
MODEL="${1:-$VLM_MODEL}"
PULL=("$MODEL")
if [ "$MODEL" = all ]; then MODEL="$VLM_MODEL"; PULL=("${VLM_MODELS[@]}"); fi

# DGX Spark: model files from the shared folder (env.sh vlm_from_shared); the rest are downloaded below.
vlm_from_shared "${PULL[@]}"

say "Ollama"
ollama_up || exit 1

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
