#!/usr/bin/env bash
# More hands-on (DGX Spark): more camera scenes for Step 6, copied from the shared folder to this pod.
# TUM Traffic Accid3nD (CC BY-NC-SA 4.0, accident-dataset.github.io): German motorway cameras, 1920x1200,
# 25 photos per second, with real accidents.
#   ./09_more_scenes.sh          -> the real accident clips (config/tum_*.yaml, up to 35 s each)
#   ./09_more_scenes.sh all      -> ... and the full camera recordings (config/auto_*.yaml, scene_*.yaml)
#   ./09_more_scenes.sh list     -> which scenes can run now
#   ./09_more_scenes.sh fill     -> teacher, once (after ./03_spark_public.sh fill): cut the accident clips
#                                   out of the full recordings, into the shared folder
# Then: ./06_run_extension.sh tum_r09_s01_south50   (or pick it in the "Scenario" list of the CCTV window)
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_project
MODE="${1:-}"
PUB="$PUBLIC_LAB/datasets/TUMTraf-Accid3nD"
LOCAL="$CCTV_DATASETS/TUMTraf-Accid3nD"
umask 002

# The folder a config plays: its video.path, relative to the datasets folder (${DATASETS}/...).
video_dir() { sed -n "s|^video: *{path: *'\${DATASETS}/\([^']*\)'.*|\1|p" "$1"; }

if [ "$MODE" = fill ]; then
  [ -d "$PUB/data" ] || { fail "No $PUB/data: run ./03_spark_public.sh fill first"; exit 1; }
  say "Accident clips: $PUB/data -> $PUB/accident_scenes"
  # The export tool also rewrites config/tum_*.yaml; run it on a copy so the project stays untouched.
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  cp -r "$PROJECT/tools" "$PROJECT/analysis" "$PROJECT/cctv_utils" "$PROJECT/config" "$TMP/"
  ( cd "$TMP" && "$ISAAC_SIM/python.sh" tools/export_accident_scenes.py --root "$PUB/data" --out "$PUB/accident_scenes.part" )
  rm -rf "$PUB/accident_scenes" && mv "$PUB/accident_scenes.part" "$PUB/accident_scenes"
  chmod -R g+rwX,o+rX "$PUB/accident_scenes" 2>/dev/null || true
  ok "$(ls -d "$PUB/accident_scenes"/*/ | wc -l) clips in $PUB/accident_scenes"
  exit 0
fi

if [ "$MODE" != list ]; then
  [ -d "$PUB" ] || { fail "No $PUB: teacher runs ./03_spark_public.sh fill and ./09_more_scenes.sh fill"; exit 1; }
  CONFIGS=("$PROJECT"/config/tum_r*.yaml)
  [ "$MODE" = all ] && CONFIGS+=("$PROJECT"/config/auto_r09_*.yaml "$PROJECT"/config/scene_r*.yaml)
  for c in "${CONFIGS[@]}"; do
    d="$(video_dir "$c")"; [ -n "$d" ] || continue
    if [ ! -d "$PUBLIC_LAB/datasets/$d" ]; then fail "$(basename "$c" .yaml): not in the shared folder"; continue; fi
    echo "  $(basename "$c" .yaml): $(du -sh "$PUBLIC_LAB/datasets/$d" | cut -f1)"
    mkdir -p "$CCTV_DATASETS/$d"
    rsync -a "$PUBLIC_LAB/datasets/$d/" "$CCTV_DATASETS/$d/"
  done
  # Labels and the scene description (what happened, when) next to each accident clip.
  [ -d "$PUB/accident_scenes" ] && rsync -a --exclude images "$PUB/accident_scenes/" "$LOCAL/accident_scenes/"
fi

say "Scenes ready to play (./06_run_extension.sh <name>)"
for c in "$PROJECT"/config/*.yaml; do
  d="$(video_dir "$c")"
  if [ -n "$d" ] && [ -d "$CCTV_DATASETS/$d" ]; then
    printf '  %-36s %s photos\n' "$(basename "$c" .yaml)" "$(ls "$CCTV_DATASETS/$d" | wc -l)"
  fi
done
printf '  %-36s %s\n' detrac_MVI_20011 "(Step 3)" detrac_MVI_40191 "(Step 3)"
