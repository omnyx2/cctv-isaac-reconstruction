#!/usr/bin/env bash
# Step 4: run the project's own health check (tools/check_env.py).
# These [!!] are fine today: TUMTraf-Accid3nD, data/cache, Ollama (not used in this lab).
set -uo pipefail
source "$(dirname "$0")/env.sh"
need_isaac; need_project
cd "$PROJECT"
"$ISAAC_SIM/python.sh" tools/check_env.py 2>&1 | grep -E "^\[(ok|!!)\]|^      ->|^python|checks ok|PERCEPTION"
