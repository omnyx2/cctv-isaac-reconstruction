#!/usr/bin/env bash
# More hands-on (DGX Spark): more camera scenes for Step 6, from the shared folder to this computer.
# TUM Traffic Accid3nD (CC BY-NC-SA 4.0, accident-dataset.github.io): German motorway cameras, 1920x1200,
# 25 photos per second, with real accidents.
#   ./09_more_scenes.sh          -> the real accident clips (config/tum_*.yaml, up to 35 s each, ~1.1 GB)
#   ./09_more_scenes.sh all      -> ... and the full camera recordings (config/auto_r09_*.yaml, scene_*.yaml, ~8 GB)
#   ./09_more_scenes.sh list     -> which scenes can run now, and which are in the shared folder
#   ./09_more_scenes.sh fill     -> teacher, once (after ./03_spark_public.sh fill): cut the accident clips
#                                   out of the TUM zips, into the shared folder
# Then: ./06_run_extension.sh tum_r09_s01_south50   (or pick it in the "Scenario" list of the CCTV window)
# Not on the DGX Spark? ./03_download_data.sh --tum-scenes downloads the same clips from the internet.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_project
MODE="${1:-}"
PUB="$PUBLIC_LAB/datasets/TUMTraf-Accid3nD"
ZIPS=("$PUB"/zips/train*.zip)
umask 002

# The folder a config plays: its video.path, relative to the datasets folder (${DATASETS}/...).
video_dir() { sed -n "s|^video: *{path: *'\${DATASETS}/\([^']*\)'.*|\1|p" "$1"; }
# Unpack the zip entries under these prefixes (e.g. r09_s10/images/s040_camera_basler_north_50mm/) into DIR.
# The shared folder writes small files slowly, so this always unpacks onto the local disk.
# A camera can be split over two zips: every zip is searched.
unpack() {  # dir prefix...
  local dir="$1"; shift
  local pats=(); for p in "$@"; do pats+=("$p*"); done
  [ -f "${ZIPS[0]}" ] || { fail "No TUM zips in $PUB/zips: teacher runs ./03_spark_public.sh fill"; exit 1; }
  mkdir -p "$dir"
  for z in "${ZIPS[@]}"; do
    unzip -q -n "$z" "${pats[@]}" -d "$dir" 2>/dev/null &
  done
  wait || true
}

if [ "$MODE" = fill ]; then
  say "Accident clips: $PUB/zips -> $PUB/accident_scenes"
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/tum_fill.XXXXXX")"; trap 'rm -rf "$TMP"' EXIT
  SEQS=(); for c in "$PROJECT"/config/tum_r09_*.yaml; do s="$(basename "$c" .yaml)"; s="${s#tum_}"; SEQS+=("${s%_*}/"); done
  mapfile -t SEQS < <(printf '%s\n' "${SEQS[@]}" | sort -u)
  echo "  unpacking ${SEQS[*]} on the local disk"
  unpack "$TMP/data" "${SEQS[@]}"
  # The export tool also rewrites config/tum_*.yaml; run it on a copy so the project stays untouched.
  mkdir -p "$TMP/project"
  cp -r "$PROJECT/tools" "$PROJECT/analysis" "$PROJECT/cctv_utils" "$PROJECT/config" "$TMP/project/"
  ( cd "$TMP/project" && "$ISAAC_SIM/python.sh" tools/export_accident_scenes.py --root "$TMP/data" --out "$TMP/scenes" )
  mkdir -p "$PUB/accident_scenes"
  for d in "$TMP"/scenes/*/; do rsync -a "$d" "$PUB/accident_scenes/$(basename "$d")/" & done
  wait
  cp "$TMP/scenes/README.md" "$PUB/accident_scenes/"
  chmod -R g+rwX,o+rX "$PUB/accident_scenes" 2>/dev/null || true
  ok "$(ls -d "$PUB/accident_scenes"/*/ | wc -l) clips in $PUB/accident_scenes"
  exit 0
fi

if [ "$MODE" != list ]; then
  [ -d "$PUB/accident_scenes" ] || { fail "No $PUB/accident_scenes: teacher runs ./09_more_scenes.sh fill"; exit 1; }
  say "Accident clips: shared folder -> $CCTV_DATASETS/TUMTraf-Accid3nD/accident_scenes"
  for c in "$PROJECT"/config/tum_r09_*.yaml; do
    d="$(video_dir "$c")"; [ -n "$d" ] || continue
    [ -d "$PUBLIC_LAB/datasets/$d" ] || { fail "$(basename "$c" .yaml): not in the shared folder"; continue; }
    echo "  $(basename "$c" .yaml): $(du -sh "$PUBLIC_LAB/datasets/$d" | cut -f1)"
    mkdir -p "$CCTV_DATASETS/$d"
    rsync -a "$PUBLIC_LAB/datasets/$d/" "$CCTV_DATASETS/$d/"
  done
  # The scene description (scene.yaml: what happened, when) and a preview video next to each clip;
  # the 3D labels (2.6 GB) only with LABELS=1 (only the evaluation tools need them).
  X=(--exclude images); [ "${LABELS:-0}" = 1 ] || X+=(--exclude labels)
  rsync -a "${X[@]}" "$PUB/accident_scenes/" "$CCTV_DATASETS/TUMTraf-Accid3nD/accident_scenes/"

  if [ "$MODE" = all ]; then
    say "Full camera recordings: TUM zips -> $CCTV_DATASETS/TUMTraf-Accid3nD/data"
    WANT=()
    for c in "$PROJECT"/config/auto_r09_*.yaml "$PROJECT"/config/scene_r*.yaml; do
      d="$(video_dir "$c")"; [ -n "$d" ] || continue
      [ -n "$(ls -A "$CCTV_DATASETS/$d" 2>/dev/null)" ] || WANT+=("${d#TUMTraf-Accid3nD/data/}/")
    done
    if [ ${#WANT[@]} -gt 0 ]; then
      mapfile -t WANT < <(printf '%s\n' "${WANT[@]}" | sort -u)
      printf '  %s\n' "${WANT[@]}"
      unpack "$CCTV_DATASETS/TUMTraf-Accid3nD/data" "${WANT[@]}"
    fi
  fi
fi

say "Scenes ready to play (./06_run_extension.sh <name>)"
for c in "$PROJECT"/config/*.yaml; do
  d="$(video_dir "$c")"
  if [ -n "$d" ] && [ -n "$(ls -A "$CCTV_DATASETS/$d" 2>/dev/null)" ]; then
    printf '  %-44s %s photos\n' "$(basename "$c" .yaml)" "$(ls "$CCTV_DATASETS/$d" | wc -l)"
  fi
done
printf '  %-44s %s\n' detrac_MVI_20011 "(Step 3)" detrac_MVI_40191 "(Step 3)"
if [ -d "$PUB" ]; then
  say "In the shared folder, not on this computer yet"
  for c in "$PROJECT"/config/tum_r09_*.yaml "$PROJECT"/config/auto_r09_*.yaml "$PROJECT"/config/scene_r*.yaml; do
    d="$(video_dir "$c")"; [ -n "$d" ] || continue
    [ -n "$(ls -A "$CCTV_DATASETS/$d" 2>/dev/null)" ] && continue
    case "$d" in
      */accident_scenes/*) [ -d "$PUBLIC_LAB/datasets/$d" ] && echo "  $(basename "$c" .yaml)   (./09_more_scenes.sh)" ;;
      *) [ -f "${ZIPS[0]}" ] && echo "  $(basename "$c" .yaml)   (./09_more_scenes.sh all)" ;;
    esac
  done
fi
