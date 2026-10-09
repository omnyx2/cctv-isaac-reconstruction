#!/usr/bin/env bash
# Presentation mode (Linux), as run_presentation.bat: 1920x1080 window, real video |
# reconstruction, timeline | VLM below, no other panels, no viewport overlays, no main menu.
#   ./run_presentation.sh [extra Kit arguments, e.g. snapshot_dir / record_dir settings]
set -euo pipefail
cd "$(dirname "$0")"
exec ./run_isaac.sh \
  --/app/window/width=1920 --/app/window/height=1080 \
  --/exts/cctv.dual_tracking/presentation_layout=true \
  --/app/viewport/defaults/hud/visible=false --/app/viewport/defaults/guide/axis/visible=false \
  --/exts/omni.kit.viewport.menubar.camera/visible=false --/exts/omni.kit.viewport.menubar.render/visible=false \
  --/exts/omni.kit.viewport.menubar.settings/visible=false --/exts/omni.kit.viewport.menubar.display/visible=false \
  --/exts/omni.kit.viewport.menubar.lighting/visible=false --/exts/omni.kit.viewport.menubar.waypoint/visible=false \
  "$@"
