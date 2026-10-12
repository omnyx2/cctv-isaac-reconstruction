#!/usr/bin/env bash
# For the teacher: steps 0-4 in one go. Run it before class; in class only 05_run.sh is needed.
set -euo pipefail
cd "$(dirname "$0")"
./00_check_machine.sh
./01_get_code.sh
./02_install_packages.sh
./03_download_data.sh
./04_check_env.sh || true
echo; echo "All set! Run:  $(pwd)/05_run.sh"
