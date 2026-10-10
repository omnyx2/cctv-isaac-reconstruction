#!/usr/bin/env bash
# Set the project up on DGX Spark from the folder made by tools/pack_for_spark.py.
#   bash ~/cctv_transfer/setup_spark.sh
# Environment (defaults):
#   PROJECT=~/CCTV_Project   where the project goes
#   CCTV_DATASETS=~/datasets where the datasets go
#   ISAAC_SIM=~/isaacsim     the Isaac Sim build with isaac-sim.sh / python.sh
#                            (source build: <IsaacSim>/_build/linux-aarch64/release)
# Isaac Sim itself is not installed by this script: see docs/DGX_SPARK.md section 3.
set -euo pipefail
PACK="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PROJECT:-$HOME/CCTV_Project}"
export CCTV_DATASETS="${CCTV_DATASETS:-$HOME/datasets}"
ISAAC_SIM="${ISAAC_SIM:-$HOME/isaacsim}"

echo "== 1/5 code: $PROJECT"
if [ -d "$PROJECT/.git" ]; then
  git -C "$PROJECT" pull "$PACK/CCTV_Project.bundle" master
else
  git clone "$PACK/CCTV_Project.bundle" "$PROJECT"
fi
chmod +x "$PROJECT"/run_isaac.sh "$PROJECT"/run_presentation.sh

echo "== 2/5 model weights and data/ (caches, calibrations, DETRAC sequences)"
cp -rn "$PACK/project_files/." "$PROJECT/"

echo "== 3/5 datasets: $CCTV_DATASETS"
mkdir -p "$CCTV_DATASETS"
if [ -d "$PACK/datasets" ]; then cp -rn "$PACK/datasets/." "$CCTV_DATASETS/"; fi

echo "== 4/5 environment variables in ~/.bashrc"
grep -q "CCTV_DATASETS" ~/.bashrc 2>/dev/null || {
  echo "export CCTV_DATASETS=\"$CCTV_DATASETS\"" >> ~/.bashrc
  echo "export ISAAC_SIM=\"$ISAAC_SIM\"" >> ~/.bashrc
}

echo "== 5/5 packages into Isaac Sim's Python, then the environment check"
if [ -x "$ISAAC_SIM/python.sh" ]; then
  "$ISAAC_SIM/python.sh" -m pip install -r "$PROJECT/requirements-isaac.txt"
  (cd "$PROJECT" && "$ISAAC_SIM/python.sh" tools/check_env.py) || true
  echo
  echo "If 'GPU kernels for this GPU' or 'YOLO inference' failed above, make the YOLO venv"
  echo "(docs/DGX_SPARK.md section 3, step 3) and set CCTV_PERCEPTION_PYTHON."
else
  echo "Isaac Sim not found at $ISAAC_SIM (python.sh). Build it first (docs/DGX_SPARK.md), then:"
  echo "  \$ISAAC_SIM/python.sh -m pip install -r $PROJECT/requirements-isaac.txt"
  echo "  cd $PROJECT && \$ISAAC_SIM/python.sh tools/check_env.py"
fi
echo
echo "Run:  cd $PROJECT && ./run_isaac.sh"
