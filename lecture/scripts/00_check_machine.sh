#!/usr/bin/env bash
# Step 0: look at the computer. Installs and changes nothing.
set -uo pipefail
source "$(dirname "$0")/env.sh"

say "Computer"
echo "CPU type : $(uname -m)        (aarch64 = an ARM computer like DGX Spark)"
echo "OS       : $(. /etc/os-release && echo "$PRETTY_NAME")"

say "Graphics card (GPU)"
if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then ok "GPU found"; else fail "nvidia-smi does not work"; fi

say "Screen (DISPLAY)"
if [ -n "${DISPLAY:-}" ]; then ok "DISPLAY=$DISPLAY  windows can open"; else fail "No DISPLAY. Use a terminal on the desktop that has a monitor."; fi

say "Isaac Sim"
if [ -n "${ISAAC_SIM:-}" ]; then ok "$ISAAC_SIM  (version $(cat "$ISAAC_SIM/VERSION" 2>/dev/null))"; else fail "Not found. Tell us where:  export ISAAC_SIM=..."; fi

say "Commands we need"
for c in git curl python3; do
  if command -v $c >/dev/null; then ok "$c"; else fail "$c missing"; fi
done

say "Free disk space ($WORK)"
df -h "$WORK" | tail -1
