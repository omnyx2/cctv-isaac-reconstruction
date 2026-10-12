#!/usr/bin/env bash
# Step 2: install the missing Python packages.
# Isaac Sim is NOT touched. Packages go only into the project's .pydeps folder, and env.sh
# points Python at it with PYTHONPATH. To undo: delete the .pydeps folder.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_isaac; need_project
PY="$ISAAC_SIM/python.sh"
PIP=("$PY" -m pip install --disable-pip-version-check --target "$PYDEPS" --upgrade)

say "Install packages -> $PYDEPS"
"${PIP[@]}" --no-deps polars==2.0.0 polars-runtime-32==2.0.0
# --no-deps: these must not pull in a second torch / numpy that would clash with Isaac Sim's.
"${PIP[@]}" --no-deps ultralytics==8.4.174 ultralytics-thop==2.2.2 lap==0.5.13
# opencv-python: Isaac Sim's OpenCV is "headless" (cannot open windows), so add one that can.
"${PIP[@]}" --no-deps opencv-python==4.11.0.86
rm -rf "$PYDEPS/bin"   # command-line tools are not needed

say "Check"
"$PY" -c "
import cv2, lap, ultralytics, polars, numpy, torch
print('cv2        ', cv2.__version__, '(can open windows:', 'QT' in cv2.getBuildInformation(), ')')
print('ultralytics', ultralytics.__version__)
print('numpy      ', numpy.__version__, '(Isaac Sim own)')
print('torch      ', torch.__version__, 'GPU:', torch.cuda.is_available())
" 2>&1 | grep -vE "^\[|Warning|settings"
ok "Packages ready"
