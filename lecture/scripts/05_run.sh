#!/usr/bin/env bash
# Step 5: run it! CCTV video -> YOLO tracking -> 3D replay in Isaac Sim
#   ./05_run.sh            -> MVI_20011 (bus stop seen from a footbridge: cars and people)
#   ./05_run.sh 40191      -> MVI_40191 (expressway, both directions)
#   ./05_run.sh 20011 --save-dir ~/snaps --save-every 100   (save side-by-side pictures)
# In the window:  Q quit   Space pause   R restart
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_isaac; need_project
SEQ="${1:-20011}"; shift || true
CONFIG="config/detrac_MVI_${SEQ#MVI_}.yaml"
cd "$PROJECT"
[ -f "$CONFIG" ] || { fail "No config file: $CONFIG"; exit 1; }
say "Run: $CONFIG  (Isaac Sim takes 20-30 s to start)"
exec "$ISAAC_SIM/python.sh" main.py --config "$CONFIG" "$@"
