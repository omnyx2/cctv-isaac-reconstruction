#!/usr/bin/env bash
# Step 3: download the CCTV videos and the YOLO models.
#   Videos: UA-DETRAC (road CCTV in Beijing, CC BY 4.0). Only the sequences we need:
#           the original zip is 5.6 GB, remote_zip.py pulls out just these photos (~60 MB per sequence).
#   Models: YOLOv8n, YOLO11s (public Ultralytics weights)
#   ./03_download_data.sh               -> MVI_20011 and MVI_40191
#   ./03_download_data.sh MVI_20011     -> just one
set -euo pipefail
source "$(dirname "$0")/env.sh"
need_project
HERE="$(cd "$(dirname "$0")" && pwd)"
ZIP_URL="https://huggingface.co/datasets/abhineet123/ua_detrac/resolve/main/ua_detrac_training_set.zip"
WEIGHTS_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0"
DETRAC="$PROJECT/data/UA-DETRAC/train"
SEQS=("$@"); [ ${#SEQS[@]} -gt 0 ] || SEQS=(MVI_20011 MVI_40191)

say "YOLO models"
for w in yolov8n.pt yolo11s.pt; do
  if [ -s "$PROJECT/$w" ]; then ok "$w already here"
  else curl -fL --progress-bar -o "$PROJECT/$w.part" "$WEIGHTS_URL/$w" && mv "$PROJECT/$w.part" "$PROJECT/$w" && ok "$w"; fi
done

declare -A FOLDER=([MVI_20011]=detrac_1_MVI_20011 [MVI_40191]=detrac_31_MVI_40191)
mkdir -p "$DETRAC" "$CCTV_DATASETS"
for s in "${SEQS[@]}"; do
  f="${FOLDER[$s]:-}"
  [ -n "$f" ] || { fail "Unknown sequence $s (choose MVI_20011 or MVI_40191)"; exit 1; }
  say "CCTV video $s -> $DETRAC/$f  (takes a few minutes)"
  python3 -I "$HERE/remote_zip.py" "$ZIP_URL" --extract "$f/" --out "$DETRAC"
  ok "$s: $(ls "$DETRAC/$f"/*.jpg | wc -l) photos"
done
