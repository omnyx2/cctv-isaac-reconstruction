#!/usr/bin/env bash
# For the teacher: steps 0-4 in one go. Run it before class; in class only 05_run.sh is needed.
# On our DGX Spark (shared folder found) it takes everything from the shared folder instead:
# all 80 UA-DETRAC videos (~8 GB, ~1 min), the TUM accident clips (~1.1 GB) and the default VLM
# (19 GB, ~20 min; Step 7 then only tests it). Elsewhere: the lab's two videos from the internet.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
./00_check_machine.sh
./01_get_code.sh
./02_install_packages.sh
if [ -d "$PUBLIC_LAB/UA-DETRAC/tar" ]; then
  ./03_download_data.sh --detrac-all
  ./09_more_scenes.sh
  vlm_from_shared "$VLM_MODEL"
else
  ./03_download_data.sh
fi
./04_check_env.sh || true
echo; echo "All set! Run:  $(pwd)/05_run.sh"
