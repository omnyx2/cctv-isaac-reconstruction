#!/usr/bin/env bash
# Step 6: run inside the Isaac Sim window as an extension.
#   ./06_run_extension.sh          -> CCTV windows docked around the viewport, MVI_20011 ready
#   ./06_run_extension.sh 40191    -> the expressway instead
#   ./06_run_extension.sh tum_r09_s01_south50   -> any config/<name>.yaml (more scenes: ./09_more_scenes.sh)
#   In the "CCTV Dual Tracking" window press Start; "CCTV Timeline" scrubs and slows the video.
#   VLM_MODEL=qwen2.5vl:7b ./06_run_extension.sh   -> another model in the VLM console (default: env.sh)
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_isaac; need_project
SEQ="${1:-20011}"; shift || true
CONFIG="config/detrac_MVI_${SEQ#MVI_}.yaml"
[ -f "$PROJECT/config/${SEQ%.yaml}.yaml" ] && CONFIG="config/${SEQ%.yaml}.yaml"
[ -f "$PROJECT/$CONFIG" ] || { fail "No $PROJECT/$CONFIG"; exit 1; }
# The "CCTV Setup" window settings, given as start options: our datasets folder, and
# perception_python empty = YOLO runs with Isaac Sim's own Python and torch (works on GB10).
# setup_ok: steps 2-4 already did what the "CCTV Setup" window offers, so it does not pop up over the viewport.
PERSIST=/persistent/exts/cctv.dual_tracking
# Isaac Sim normally ignores PYTHONPATH; this flag lets it see our .pydeps (ultralytics, lap).
PYENV_FLAG=/plugins/carb.scripting-python.plugin/pythonFlags/Py_IgnoreEnvironmentFlag=false
# Quit without "Would you like to save this stage?": the replay scene is rebuilt every run, and the
# dialog blocks unattended runs (quit_on_finish) forever.
NOSAVE_FLAG=/app/file/ignoreUnsavedOnExit=true
# The VLM console talks to Ollama: start it now, one model at a time (Isaac Sim runs fine without it).
ollama_up || echo "  (the VLM console will say the VLM is unavailable)"
cd "$PROJECT"
exec "$ISAAC_SIM/isaac-sim.sh" --ext-folder exts --enable cctv.dual_tracking --enable cctv.timeline \
  --$PYENV_FLAG --$NOSAVE_FLAG \
  --/exts/cctv.dual_tracking/layout=true \
  --/exts/cctv.dual_tracking/vlm_model="$VLM_MODEL" \
  --$PERSIST/perception_python="" \
  --$PERSIST/setup_ok=true \
  --$PERSIST/datasets_root="$CCTV_DATASETS" \
  --/exts/cctv.dual_tracking/default_config="$CONFIG" "$@"
