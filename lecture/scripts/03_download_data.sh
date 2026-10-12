#!/usr/bin/env bash
# Step 3: download the CCTV videos and the YOLO models.
#   Models: YOLOv8n, YOLO11s (public Ultralytics weights), always.
#   UA-DETRAC (road CCTV in Beijing, CC BY 4.0): 60 training sequences in one 5.6 GB zip;
#     remote_zip.py pulls out just the sequences asked for (60-300 MB each).
#   TUM Traffic Accid3nD (real highway accidents, CC BY-NC-SA 4.0 = non-commercial): five ~9 GB zips;
#     tum_download.py pulls out just the accident scenes or cameras asked for.
#
#   ./03_download_data.sh                          MVI_20011 and MVI_40191 (the lab's two videos)
#   ./03_download_data.sh MVI_39761 40131          any UA-DETRAC training sequence ("MVI_" optional)
#   ./03_download_data.sh --detrac-all             all 60 UA-DETRAC sequences (~5.6 GB)
#   ./03_download_data.sh --tum-scenes             the 4 real accident scenes, 35 s each (~1.1 GB)
#   ./03_download_data.sh --tum-scenes r09_s02_north50     one of them
#   ./03_download_data.sh --tum r09_s10 north50    one whole TUM camera (~3000 photos, ~1 GB)
#   ./03_download_data.sh --tum-configs            every TUM camera a config/auto_*.yaml replays (~15 GB)
#   ./03_download_data.sh --list                   what can be downloaded
#   add --labels to a TUM option to get its 3D labels too (only needed by the evaluation tools)
# Options combine (e.g. MVI_20011 --tum-scenes). Run it again after a stop: finished photos are skipped.
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_project
HERE="$(cd "$(dirname "$0")" && pwd)"
ZIP_URL="https://huggingface.co/datasets/abhineet123/ua_detrac/resolve/main/ua_detrac_training_set.zip"
WEIGHTS_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0"
DETRAC="$PROJECT/data/UA-DETRAC/train"
TUM=(python3 -I "$HERE/tum_download.py" --project "$PROJECT" --datasets "$CCTV_DATASETS")

SEQS=(); DETRAC_ALL=0; LIST=0; TUM_ARGS=(); LABELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST=1 ;;
    --detrac-all) DETRAC_ALL=1 ;;
    --labels) LABELS=(--labels) ;;
    --tum-configs) TUM_ARGS+=(--configs) ;;
    --tum) [ $# -ge 3 ] || { fail "--tum needs SEQUENCE CAMERA, e.g. --tum r09_s10 north50"; exit 1; }
           TUM_ARGS+=(--camera "$2" "$3"); shift 2 ;;
    --tum-scenes) TUM_ARGS+=(--scenes)
           while [ $# -gt 1 ] && [[ "$2" == r09_* ]]; do TUM_ARGS+=("$2"); shift; done ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    MVI_*|[0-9]*) SEQS+=("MVI_${1#MVI_}") ;;
    *) fail "Unknown option $1  (see ./03_download_data.sh --help)"; exit 1 ;;
  esac
  shift
done
[ ${#SEQS[@]} -gt 0 ] || [ $DETRAC_ALL = 1 ] || [ ${#TUM_ARGS[@]} -gt 0 ] || [ $LIST = 1 ] || SEQS=(MVI_20011 MVI_40191)

if [ $LIST = 1 ]; then
  say "UA-DETRAC training sequences (./03_download_data.sh MVI_xxxxx)"
  DIRS="$(python3 -I "$HERE/remote_zip.py" "$ZIP_URL" --dirs)"
  echo "$DIRS" | sed 's/^detrac_[0-9]*_//' | sort | paste -sd' ' | fold -s -w 100
  echo "  Ready-made configs: MVI_20011, MVI_40191. The others have none: the extension can try to calibrate a picked video."
  say "TUM Traffic Accid3nD"
  "${TUM[@]}" --list
  exit 0
fi

say "YOLO models"
for w in yolov8n.pt yolo11s.pt; do
  if [ -s "$PROJECT/$w" ]; then ok "$w already here"
  else curl -fL --progress-bar -o "$PROJECT/$w.part" "$WEIGHTS_URL/$w" && mv "$PROJECT/$w.part" "$PROJECT/$w" && ok "$w"; fi
done

mkdir -p "$DETRAC" "$CCTV_DATASETS"
if [ ${#SEQS[@]} -gt 0 ] || [ $DETRAC_ALL = 1 ]; then
  say "UA-DETRAC: reading the zip's table of contents"
  mapfile -t FOLDERS < <(python3 -I "$HERE/remote_zip.py" "$ZIP_URL" --dirs | tr -d '\r')
  [ ${#FOLDERS[@]} -gt 0 ] || { fail "Could not read the UA-DETRAC zip (network?)  $ZIP_URL"; exit 1; }
  if [ $DETRAC_ALL = 1 ]; then SEQS=(); for f in "${FOLDERS[@]}"; do SEQS+=("${f#detrac_*_}"); done; fi
  for s in "${SEQS[@]}"; do
    f=""; for g in "${FOLDERS[@]}"; do [[ "$g" == *_"$s" ]] && f="$g"; done
    [ -n "$f" ] || { fail "Unknown sequence $s  (./03_download_data.sh --list)"; exit 1; }
    say "CCTV video $s -> $DETRAC/$f  (takes a few minutes)"
    python3 -I "$HERE/remote_zip.py" "$ZIP_URL" --extract "$f/" --out "$DETRAC"
    ok "$s: $(ls "$DETRAC/$f"/*.jpg | wc -l) photos"
  done
fi

if [ ${#TUM_ARGS[@]} -gt 0 ]; then
  say "TUM Traffic Accid3nD -> $CCTV_DATASETS/TUMTraf-Accid3nD  (CC BY-NC-SA 4.0: non-commercial use only)"
  "${TUM[@]}" "${TUM_ARGS[@]}" "${LABELS[@]}"
  ok "TUM Traffic Accid3nD: done"
fi
