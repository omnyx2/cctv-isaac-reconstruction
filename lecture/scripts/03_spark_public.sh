#!/usr/bin/env bash
# DGX Spark only: Step 3 (videos) and the Step 7 download (VLM), from the shared folder instead of the internet.
# Every pod sees the same shared folder (PUBLIC_LAB in env.sh, /mnt/public/cctv_lab), so the big downloads
# happen once, there. Other computers keep using 03_download_data.sh and 07_vlm_setup.sh as before.
#   ./03_spark_public.sh          -> videos + YOLO models + the default VLM (VLM_MODEL), copied to this pod
#   ./03_spark_public.sh all      -> the same, with every VLM in VLM_MODELS
#   ./03_spark_public.sh fill     -> teacher, once: all datasets + all VLM_MODELS into the shared folder
#   ./03_spark_public.sh datasets -> teacher: only the datasets (UA-DETRAC train+test, TUMTraf-Accid3nD)
# After this, Step 7 (./07_vlm_setup.sh) finds the model already here and goes straight to the test.
set -euo pipefail
source "$(dirname "$0")/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-}"
SEQ_DIRS=(detrac_1_MVI_20011 detrac_31_MVI_40191)
PUB_DETRAC="$PUBLIC_LAB/UA-DETRAC/train"
PUB_OLLAMA="$PUBLIC_LAB/ollama_models"
LOCAL_OLLAMA="${OLLAMA_MODELS:-$HOME/.ollama/models}"
umask 002   # files in the shared folder stay usable by the other pods

[ -d "$(dirname "$PUBLIC_LAB")" ] || { fail "No shared folder $(dirname "$PUBLIC_LAB"): use 03_download_data.sh"; exit 1; }

# Teacher, once: every dataset the project's configs use, whole, into the shared folder (~57 GB download).
#   UA-DETRAC train + test   -> $PUBLIC_LAB/UA-DETRAC/{train,test}/detrac_*_MVI_*   (unpacked)
#   TUMTraf-Accid3nD (CC BY-NC-SA 4.0, accident-dataset.github.io) -> $PUBLIC_LAB/datasets/TUMTraf-Accid3nD/zips/train1-5.zip
#     kept as zips: the shared folder writes small files slowly (~30 photos/s, 2 hours for its 223,000 files),
#     so ./09_more_scenes.sh unpacks only the cameras asked for, on each computer's own disk (~400 photos/s).
# Downloads resume (curl -C -, into <name>.part); a rerun skips what is finished.
download() {  # url file
  [ -f "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 20 --retry-all-errors --retry-delay 10 --speed-limit 100000 --speed-time 120 \
       -C - -o "$2.part" "$1"
  mv "$2.part" "$2"
}
fetch_unzip() {  # url name dest
  local zip="$PUBLIC_LAB/downloads/$2"
  [ -f "$3/.done_$2" ] && { ok "$2 already unpacked"; return 0; }
  say "$2 -> $3"
  download "$1" "$zip"
  mkdir -p "$3"
  unzip -q -n "$zip" -d "$3"
  touch "$3/.done_$2" && rm -f "$zip"
  ok "$2"
}
if [ "$MODE" = fill ] || [ "$MODE" = datasets ]; then
  HF=https://huggingface.co/datasets/abhineet123/ua_detrac/resolve/main
  fetch_unzip "$HF/ua_detrac_training_set.zip" ua_detrac_training_set.zip "$PUBLIC_LAB/UA-DETRAC/train"
  fetch_unzip "$HF/ua_detrac_test_set.zip" ua_detrac_test_set.zip "$PUBLIC_LAB/UA-DETRAC/test"
  say "TUM Traffic Accid3nD: 5 zips, 47 GB, downloaded side by side (the server is slow per connection)"
  TUM=https://nx21496.your-storageshare.de/public.php/dav/files/Zd4rwsMwwXMaXiK/accid3nd_dataset
  pids=()
  for z in train1 train2 train3 train4 train5; do
    download "$TUM/$z.zip" "$PUBLIC_LAB/datasets/TUMTraf-Accid3nD/zips/$z.zip" & pids+=($!)
  done
  bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad + 1)); done
  [ "$bad" = 0 ] || { fail "$bad TUM zip(s) failed: run the same command again, it continues"; exit 1; }
  ok "TUM zips: $PUBLIC_LAB/datasets/TUMTraf-Accid3nD/zips  (next: ./09_more_scenes.sh fill)"
  chmod -R g+rwX,o+rX "$PUBLIC_LAB/UA-DETRAC" "$PUBLIC_LAB/datasets" 2>/dev/null || true
  [ "$MODE" = datasets ] && { ok "Datasets ready: $PUBLIC_LAB"; exit 0; }
fi

if [ "$MODE" = fill ]; then
  say "Videos -> $PUB_DETRAC"
  mkdir -p "$PUB_DETRAC"
  ZIP_URL="https://huggingface.co/datasets/abhineet123/ua_detrac/resolve/main/ua_detrac_training_set.zip"
  for f in "${SEQ_DIRS[@]}"; do
    python3 -I "$HERE/remote_zip.py" "$ZIP_URL" --extract "$f/" --out "$PUB_DETRAC"
    ok "$f: $(ls "$PUB_DETRAC/$f"/*.jpg | wc -l) photos"
  done

  say "VLM models -> $PUB_OLLAMA"
  # A second Ollama server on another port, storing into the shared folder; the normal one is not touched.
  command -v ollama >/dev/null || { fail "ollama is not installed"; exit 1; }
  mkdir -p "$PUB_OLLAMA"
  export OLLAMA_HOST=127.0.0.1:11435
  OLLAMA_MODELS="$PUB_OLLAMA" OLLAMA_NOPRUNE=1 ollama serve >"$PUBLIC_LAB/fill_ollama.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null' EXIT
  for _ in $(seq 30); do curl -fs "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 1; done
  for m in "${VLM_MODELS[@]}"; do
    say "Model $m"
    ollama pull "$m" && ok "$m"
  done
  ollama list
  chmod -R g+rwX,o+rX "$PUBLIC_LAB" 2>/dev/null || true
  ok "Shared folder ready: $PUBLIC_LAB"
  exit 0
fi

need_project
say "Videos: $PUB_DETRAC -> $PROJECT/data/UA-DETRAC/train"
mkdir -p "$PROJECT/data/UA-DETRAC/train" "$CCTV_DATASETS"
for f in "${SEQ_DIRS[@]}"; do
  [ -d "$PUB_DETRAC/$f" ] || { fail "$f is not in the shared folder: teacher runs ./03_spark_public.sh fill"; exit 1; }
  cp -ru "$PUB_DETRAC/$f" "$PROJECT/data/UA-DETRAC/train/"
  ok "$f: $(ls "$PROJECT/data/UA-DETRAC/train/$f"/*.jpg | wc -l) photos"
done

say "YOLO models (small, from GitHub as in Step 3)"
WEIGHTS_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0"
for w in yolov8n.pt yolo11s.pt; do
  if [ -s "$PROJECT/$w" ]; then ok "$w already here"
  else curl -fL --progress-bar -o "$PROJECT/$w.part" "$WEIGHTS_URL/$w" && mv "$PROJECT/$w.part" "$PROJECT/$w" && ok "$w"; fi
done

# Copy the model files (manifest + blobs) into this pod's own Ollama folder: Ollama then loads them from
# the local disk, and Step 7 works the same as on any other computer.
MODELS=("$VLM_MODEL"); [ "$MODE" = all ] && MODELS=("${VLM_MODELS[@]}")
for m in "${MODELS[@]}"; do
  say "VLM $m: shared folder -> $LOCAL_OLLAMA"
  name="${m%%:*}"; tag="${m#*:}"; [ "$tag" = "$m" ] && tag=latest
  man="manifests/registry.ollama.ai/library/$name/$tag"
  [ -f "$PUB_OLLAMA/$man" ] || { fail "$m is not in the shared folder: teacher runs ./03_spark_public.sh fill"; exit 1; }
  mkdir -p "$LOCAL_OLLAMA/blobs" "$(dirname "$LOCAL_OLLAMA/$man")"
  for b in $(python3 -I -c 'import json, sys
d = json.load(open(sys.argv[1]))
for l in [d["config"]] + d["layers"]: print(l["digest"].replace(":", "-"))' "$PUB_OLLAMA/$man"); do
    if [ "$(stat -c %s "$LOCAL_OLLAMA/blobs/$b" 2>/dev/null)" = "$(stat -c %s "$PUB_OLLAMA/blobs/$b")" ]; then continue; fi
    echo "  copying $b ($(du -h "$PUB_OLLAMA/blobs/$b" | cut -f1))"
    cp "$PUB_OLLAMA/blobs/$b" "$LOCAL_OLLAMA/blobs/$b.part" && mv "$LOCAL_OLLAMA/blobs/$b.part" "$LOCAL_OLLAMA/blobs/$b"
  done
  cp "$PUB_OLLAMA/$man" "$LOCAL_OLLAMA/$man"
  ok "$m"
done
ok "Done. Next: ./04_check_env.sh, and for the VLM ./07_vlm_setup.sh"
