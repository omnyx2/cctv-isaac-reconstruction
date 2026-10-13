# Shared settings, read first by every lab script (source env.sh).
# To change a value, export it before running a script.  e.g.  export ISAAC_SIM=~/isaacsim

# Work folder: the folder that contains lecture/ (e.g. ~/AI_CCTV_PROJ)
WORK="${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_URL="${REPO_URL:-https://github.com/omnyx2/cctv-isaac-reconstruction.git}"
PROJECT="${PROJECT:-$WORK/cctv-isaac-reconstruction}"

# Find the Isaac Sim 5.1 folder (the one with isaac-sim.sh and python.sh).
if [ -z "${ISAAC_SIM:-}" ]; then
  for d in "$HOME/Desktop/Issac-sim-5.1" "$HOME/Desktop/Isaac-sim-5.1" "$HOME/isaacsim" \
           "$HOME/IsaacSim/_build/linux-aarch64/release" "$HOME/IsaacSim/_build/linux-x86_64/release"; do
    if [ -x "$d/python.sh" ]; then ISAAC_SIM="$d"; break; fi
  done
fi
export ISAAC_SIM

# Isaac Sim itself is never changed: extra packages go only into the project's .pydeps folder.
PYDEPS="$PROJECT/.pydeps"
export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
export CCTV_DATASETS="${CCTV_DATASETS:-$PROJECT/data/datasets}"

# VLM for steps 7-8: qwen3-vl 30B-A3B, 4-bit (q4_K_M), ~20 GB in memory. Best of the models we tested
# on this DGX Spark (lecture/vlm_bench/results/REPORT.md). Smaller: VLM_MODEL=qwen2.5vl:7b (~6 GB).
# Do not use 72B or bigger models here: they ran out of memory and killed the whole desktop.
VLM_MODEL="${VLM_MODEL:-qwen3-vl:30b-a3b-instruct}"
# More models to compare in the VLM console (./07_vlm_setup.sh all), one per family, all safe here:
# Qwen (default, ~19 GB), Qwen small (~6 GB), Google Gemma (~7 GB), Meta Llama (~8 GB), Mistral (~15 GB).
VLM_MODELS=(qwen3-vl:30b-a3b-instruct qwen2.5vl:7b gemma4:e2b llama3.2-vision:11b mistral-small3.2:24b)
# DGX Spark only: shared folder (all pods see it) with the videos and VLM models, see 03_spark_public.sh.
PUBLIC_LAB="${PUBLIC_LAB:-/mnt/public/CCTV_recon_dL}"

# DGX Spark (aarch64): Isaac Sim needs libgomp preloaded.
if [ "$(uname -m)" = "aarch64" ] && [ -f /lib/aarch64-linux-gnu/libgomp.so.1 ]; then
  case ":${LD_PRELOAD:-}:" in *libgomp.so.1*) ;; *) export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1" ;; esac
fi

say()  { printf '\n\033[1;36m==== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[!!]\033[0m %s\n' "$*"; }
need_isaac() {
  [ -n "${ISAAC_SIM:-}" ] && [ -x "$ISAAC_SIM/python.sh" ] && return 0
  fail "Isaac Sim not found. Run:  export ISAAC_SIM=<folder with isaac-sim.sh>"; exit 1
}
# Ollama server for the VLM (steps 7-8): this user's own, always with ONE model in memory at a time
# (two big VLMs together can run this machine out of memory, and an OOM kills the whole desktop).
# Context 8k: otherwise qwen3-vl reserves memory for 262k tokens (45 GB instead of 20 GB).
# A server of ours started without these settings is restarted with them.
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
ollama_servers() {  # pids of this user's "ollama serve" processes (not the model runners it starts)
  local pid
  for pid in $(pgrep -u "$(id -u)" -x ollama); do
    [ "$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null)" = "ollama serve " ] && echo "$pid"
  done
  return 0
}
ollama_up() {
  command -v ollama >/dev/null || { fail "ollama is not installed: see https://ollama.com/download"; return 1; }
  local pid
  for pid in $(ollama_servers); do
    local envs; envs="$(tr '\0' '\n' </proc/"$pid"/environ 2>/dev/null)"
    grep -qx OLLAMA_MAX_LOADED_MODELS=1 <<<"$envs" && continue
    grep -q '^OLLAMA_HOST=.*:1143[5-9]' <<<"$envs" && continue   # another port (03 fill's download server)
    echo "  restarting Ollama (pid $pid) with one model at a time"
    kill "$pid"; for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  done
  if ! curl -fs "$OLLAMA_URL/api/tags" >/dev/null; then
    OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=8192 \
      nohup ollama serve >"$WORK/lecture/ollama.log" 2>&1 &
    for _ in $(seq 30); do curl -fs "$OLLAMA_URL/api/tags" >/dev/null && break; sleep 1; done
    curl -fs "$OLLAMA_URL/api/tags" >/dev/null || { fail "Ollama did not start, see $WORK/lecture/ollama.log"; return 1; }
    ok "Ollama started, one model at a time (log: $WORK/lecture/ollama.log)"
  elif [ -z "$(ollama_servers)" ]; then
    fail "Ollama at $OLLAMA_URL is not ours (system service?): make sure it runs with OLLAMA_MAX_LOADED_MODELS=1"
  else
    ok "Ollama running at $OLLAMA_URL, one model at a time"
  fi
}
# DGX Spark: copy VLM models (manifest + blobs) from the shared folder into this user's Ollama folder.
# The shared drive is slow (~15 MB/s, measured 2026-10-13; reading in parallel does not help): the default
# model (19 GB) takes ~20 minutes. The copy shows its progress and, stopped, continues on the next run.
# Models that are not in the shared folder are skipped (07 downloads them).
vlm_from_shared() {  # model...
  local SHARED="$PUBLIC_LAB/ollama_models" LOCAL_OLLAMA="${OLLAMA_MODELS:-$HOME/.ollama/models}"
  local m name tag man b todo gb
  [ -d "$SHARED/manifests" ] || return 0
  for m in "$@"; do
    name="${m%%:*}"; tag="${m#*:}"; [ "$tag" = "$m" ] && tag=latest
    man="manifests/registry.ollama.ai/library/$name/$tag"
    [ -f "$SHARED/$man" ] || continue   # not in the shared folder: 07 downloads it with ollama pull
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
}
need_project() {
  [ -f "$PROJECT/main.py" ] && return 0
  fail "Project not found: $PROJECT  ->  run ./01_get_code.sh first"; exit 1
}
