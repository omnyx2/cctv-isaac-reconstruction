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
PUBLIC_LAB="${PUBLIC_LAB:-/mnt/public/cctv_lab}"

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
need_project() {
  [ -f "$PROJECT/main.py" ] && return 0
  fail "Project not found: $PROJECT  ->  run ./01_get_code.sh first"; exit 1
}
