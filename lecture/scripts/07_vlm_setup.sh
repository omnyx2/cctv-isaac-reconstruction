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

# DGX Spark: copy the model files (manifest + blobs) from the shared folder into this user's Ollama folder.
# The shared drive is slow (~15 MB/s, measured 2026-10-13; reading in parallel does not help): the default
# model (19 GB) takes ~20 minutes. The copy shows its progress and, stopped, continues on the next run.
SHARED="$PUBLIC_LAB/ollama_models"
LOCAL_OLLAMA="${OLLAMA_MODELS:-$HOME/.ollama/models}"
if [ -d "$SHARED/manifests" ]; then
  for m in "${PULL[@]}"; do
    name="${m%%:*}"; tag="${m#*:}"; [ "$tag" = "$m" ] && tag=latest
    man="manifests/registry.ollama.ai/library/$name/$tag"
    [ -f "$SHARED/$man" ] || continue   # not in the shared folder: ollama pull downloads it below
    mkdir -p "$LOCAL_OLLAMA/blobs" "$(dirname "$LOCAL_OLLAMA/$man")"
    todo=()
    for b in $(python3 -I -c 'import json, sys
d = json.load(open(sys.argv[1]))
for l in [d["config"]] + d["layers"]: print(l["digest"].replace(":", "-"))' "$SHARED/$man"); do
      [ "$(stat -c %s "$LOCAL_OLLAMA/blobs/$b" 2>/dev/null)" = "$(stat -c %s "$SHARED/blobs/$b")" ] || todo+=("$SHARED/blobs/$b")
    done
    if [ ${#todo[@]} -gt 0 ]; then
      gb=$(du -cb "${todo[@]}" | tail -1 | awk '{printf "%.1f", $1 / 1e9}')
      say "Model $m: $gb GB from the shared folder (about $(awk -v g="$gb" 'BEGIN{printf "%d", g * 1000 / 15 / 60 + 1}') min)"
      # --partial-dir: an interrupted file continues next time; Ollama never sees a half-copied blob.
      rsync -t --partial-dir=.rsync-partial --info=progress2 "${todo[@]}" "$LOCAL_OLLAMA/blobs/"
    fi
    cp "$SHARED/$man" "$LOCAL_OLLAMA/$man"   # last: the model shows up in Ollama only when complete
    ok "$m on this computer"
  done
fi

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
