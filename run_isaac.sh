#!/usr/bin/env bash
# Isaac Sim with the CCTV extensions (Linux, e.g. DGX Spark). Same as run_isaac.bat:
# windows docked around a clean viewport, the default scenario previewed.
#   ./run_isaac.sh [extra Kit arguments]
# ISAAC_SIM: the Isaac Sim directory with isaac-sim.sh (a source build on DGX Spark:
#   <IsaacSim>/_build/linux-aarch64/release); default ~/isaacsim
# CCTV_DATASETS: where the datasets are (default ~/datasets), see cctv_utils/paths.py
# CCTV_PERCEPTION_PYTHON: a Python of its own for YOLO, see docs/DGX_SPARK.md
set -euo pipefail
cd "$(dirname "$0")"
ISAAC_SIM="${ISAAC_SIM:-$HOME/isaacsim}"
exec "$ISAAC_SIM/isaac-sim.sh" --ext-folder exts --enable cctv.dual_tracking --enable cctv.timeline \
  --/exts/cctv.dual_tracking/layout=true \
  --/exts/cctv.dual_tracking/default_config=config/tum_r09_s03_north50.yaml \
  "$@"
