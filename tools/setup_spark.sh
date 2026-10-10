#!/usr/bin/env bash
# One-command setup on DGX Spark (Linux aarch64) from the folder made by
# tools/pack_for_spark.py (or sent by tools/deploy_to_spark.py). Safe to run again:
# every step is skipped when it is already done.
#
#   bash ~/cctv_transfer/setup_spark.sh [--ollama] [--no-isaac-build]
#
#   1  Isaac Sim 5.1 from source, if $ISAAC_SIM/isaac-sim.sh is missing
#      (sudo apt: gcc-11, g++-11, git-lfs; ./build.sh asks to accept the EULA)
#   2  the project from the git bundle into $PROJECT (clone, or pull when it exists)
#   3  model weights, data/ caches and the datasets into place
#   4  packages into Isaac Sim's Python (ultralytics with --no-deps: it must not pull torch)
#   5  environment check; if YOLO cannot use the GPU with Isaac Sim's torch, a venv with
#      PyTorch cu130 is made and CCTV_PERCEPTION_PYTHON set (GB10 = sm_121)
#   6  --ollama: Ollama and qwen2.5vl:7b for the VLM
#   7  environment variables in ~/.bashrc
#
# Defaults (override in the environment):
#   PROJECT=~/CCTV_Project  CCTV_DATASETS=~/datasets  ISAAC_SIM_SRC=~/IsaacSim
#   ISAAC_SIM=$ISAAC_SIM_SRC/_build/linux-aarch64/release  PERCEPTION_VENV=~/cctv-perception
#   ISAAC_SIM_REF=        (git branch or tag of IsaacSim; empty = the repository default)
set -euo pipefail

PACK="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PROJECT:-$HOME/CCTV_Project}"
export CCTV_DATASETS="${CCTV_DATASETS:-$HOME/datasets}"
ISAAC_SIM_SRC="${ISAAC_SIM_SRC:-$HOME/IsaacSim}"
ISAAC_SIM="${ISAAC_SIM:-$ISAAC_SIM_SRC/_build/linux-aarch64/release}"
PERCEPTION_VENV="${PERCEPTION_VENV:-$HOME/cctv-perception}"
ISAAC_SIM_REF="${ISAAC_SIM_REF:-}"
WITH_OLLAMA=0
BUILD_ISAAC=1
for arg in "$@"; do
  case "$arg" in
    --ollama) WITH_OLLAMA=1 ;;
    --no-isaac-build) BUILD_ISAAC=0 ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done

step() { echo; echo "==== $*"; }
bashrc() {  # bashrc NAME VALUE: export in ~/.bashrc once (replaced if present)
  touch ~/.bashrc
  sed -i "/^export $1=/d" ~/.bashrc
  echo "export $1=\"$2\"" >> ~/.bashrc
}

step "0  machine"
echo "$(uname -m), $(lsb_release -ds 2>/dev/null || cat /etc/os-release | head -1)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || echo "WARNING: nvidia-smi failed"
[ "$(uname -m)" = "aarch64" ] || echo "NOTE: not aarch64 - fine on an x86 Linux too, the Isaac Sim build path then differs (set ISAAC_SIM)"

step "1  Isaac Sim ($ISAAC_SIM)"
if [ -x "$ISAAC_SIM/isaac-sim.sh" ]; then
  echo "found"
elif [ "$BUILD_ISAAC" = 0 ]; then
  echo "not found and --no-isaac-build given: set ISAAC_SIM to your Isaac Sim, then run again"; exit 1
else
  sudo apt-get update
  sudo apt-get install -y gcc-11 g++-11 git git-lfs build-essential
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
  if [ ! -d "$ISAAC_SIM_SRC/.git" ]; then
    git clone --depth=1 --recursive ${ISAAC_SIM_REF:+--branch "$ISAAC_SIM_REF"} \
      https://github.com/isaac-sim/IsaacSim "$ISAAC_SIM_SRC"
  fi
  (cd "$ISAAC_SIM_SRC" && git lfs install && git lfs pull && ./build.sh)
  [ -x "$ISAAC_SIM/isaac-sim.sh" ] || { echo "build finished but $ISAAC_SIM/isaac-sim.sh is missing"; exit 1; }
fi

step "2  project ($PROJECT)"
if [ -d "$PROJECT/.git" ]; then
  git -C "$PROJECT" pull --ff-only "$PACK/CCTV_Project.bundle" master
else
  git clone "$PACK/CCTV_Project.bundle" "$PROJECT"
fi
chmod +x "$PROJECT"/run_isaac.sh "$PROJECT"/run_presentation.sh "$PROJECT"/tools/*.sh

step "3  model weights, data/, datasets ($CCTV_DATASETS)"
[ -d "$PACK/project_files" ] && cp -rn "$PACK/project_files/." "$PROJECT/"
mkdir -p "$CCTV_DATASETS"
[ -d "$PACK/datasets" ] && cp -rn "$PACK/datasets/." "$CCTV_DATASETS/"
ls "$PROJECT"/*.pt

step "4  packages into Isaac Sim's Python"
"$ISAAC_SIM/python.sh" -m pip install -r "$PROJECT/requirements-isaac.txt"
"$ISAAC_SIM/python.sh" -m pip install --no-deps ultralytics==8.4.174 ultralytics-thop==2.2.2

step "5  environment check (Isaac Sim's Python)"
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1"
set +e
(cd "$PROJECT" && "$ISAAC_SIM/python.sh" tools/check_env.py)
status=$?
set -e
if [ "$status" = 3 ]; then
  echo
  echo "YOLO cannot use the GPU with Isaac Sim's torch: making $PERCEPTION_VENV (PyTorch cu130)"
  python3 -m venv "$PERCEPTION_VENV" || { sudo apt-get install -y python3-venv && python3 -m venv "$PERCEPTION_VENV"; }
  "$PERCEPTION_VENV/bin/pip" install --upgrade pip
  "$PERCEPTION_VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  "$PERCEPTION_VENV/bin/pip" install -r "$PROJECT/requirements-perception.txt"
  export CCTV_PERCEPTION_PYTHON="$PERCEPTION_VENV/bin/python"
  bashrc CCTV_PERCEPTION_PYTHON "$CCTV_PERCEPTION_PYTHON"
  (cd "$PROJECT" && "$CCTV_PERCEPTION_PYTHON" tools/check_env.py) || echo "WARNING: the venv check failed too - see above"
fi

if [ "$WITH_OLLAMA" = 1 ]; then
  step "6  Ollama + qwen2.5vl:7b"
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5vl:7b
fi

step "7  ~/.bashrc"
bashrc CCTV_DATASETS "$CCTV_DATASETS"
bashrc ISAAC_SIM "$ISAAC_SIM"
echo "CCTV_DATASETS=$CCTV_DATASETS  ISAAC_SIM=$ISAAC_SIM  CCTV_PERCEPTION_PYTHON=${CCTV_PERCEPTION_PYTHON:-}"

step "done"
echo "Run:   cd $PROJECT && ./run_isaac.sh        (presentation: ./run_presentation.sh)"
echo "Check: cd $PROJECT && \$ISAAC_SIM/python.sh tools/world_hash.py --compare tools/world_hash_baseline.json"
