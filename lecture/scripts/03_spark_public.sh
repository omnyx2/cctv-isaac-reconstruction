#!/usr/bin/env bash
# DGX Spark, teacher only: fill the shared folder once (PUBLIC_LAB in env.sh, /mnt/public/CCTV_recon_dL), which
# every pod sees. Students then run the usual steps: on a Spark, ./03_download_data.sh and ./07_vlm_setup.sh
# find the shared folder and copy from it instead of downloading. Other computers download as before.
#   ./03_spark_public.sh fill     -> all datasets + all VLM_MODELS into the shared folder
#   ./03_spark_public.sh datasets -> only the datasets (UA-DETRAC train+test, TUMTraf-Accid3nD)
#   ./03_spark_public.sh pack     -> only re-pack the UA-DETRAC videos the students get (see DETRAC_TAR)
set -euo pipefail
source "$(dirname "$0")/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-}"
SEQ_DIRS=(detrac_1_MVI_20011 detrac_31_MVI_40191)
PUB_DETRAC="$PUBLIC_LAB/UA-DETRAC/train"
DETRAC_TAR="$PUBLIC_LAB/UA-DETRAC/tar"
PUB_OLLAMA="$PUBLIC_LAB/ollama_models"
umask 002   # files in the shared folder stay usable by the other pods

[ -d "$(dirname "$PUBLIC_LAB")" ] || { fail "No shared folder $(dirname "$PUBLIC_LAB"): use 03_download_data.sh"; exit 1; }
case "$MODE" in fill|datasets|pack) ;; *) sed -n '2,8p' "$0"; exit 1 ;; esac

# The videos students get: 80 of the 100 (all 60 train + the first 20 test), one .tar per video in
# $DETRAC_TAR/{train,test}. One big file copies fast; 140,000 small photos crawl over the shared folder.
pack_detrac() {
  say "UA-DETRAC: 80 videos -> $DETRAC_TAR"
  ( cd "$PUBLIC_LAB/UA-DETRAC"
    mkdir -p "$DETRAC_TAR/train" "$DETRAC_TAR/test"
    { ls -d train/detrac_*_MVI_*; ls -d test/detrac_*_MVI_* | sort -t_ -k2 -n | head -20; } |
      xargs -P 16 -I{} sh -c '[ -f "tar/{}.tar" ] || { tar -cf "tar/{}.tar.part" -C "$(dirname {})" "$(basename {})" && mv "tar/{}.tar.part" "tar/{}.tar"; }' )
  ok "$(ls "$DETRAC_TAR"/*/*.tar | wc -l) videos packed"
}

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
[ "$MODE" = pack ] && { pack_detrac; exit 0; }
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
  pack_detrac
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
  # A second Ollama server on another port; the normal one is not touched. It downloads onto the local disk
  # first: Ollama writes each model in 16 chunks at once, and on the shared drive that stalls at 0 bytes.
  # Finished files are then copied to the shared folder. A model already there is skipped.
  command -v ollama >/dev/null || { fail "ollama is not installed"; exit 1; }
  STAGE="$HOME/.cache/cctv_lab_ollama_stage"
  mkdir -p "$PUB_OLLAMA/blobs" "$PUB_OLLAMA/manifests" "$STAGE"
  export OLLAMA_HOST=127.0.0.1:11435
  OLLAMA_MODELS="$STAGE" OLLAMA_NOPRUNE=1 ollama serve >"$STAGE.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null' EXIT
  for _ in $(seq 30); do curl -fs "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 1; done
  for m in "${VLM_MODELS[@]}"; do
    name="${m%%:*}"; tag="${m#*:}"; [ "$tag" = "$m" ] && tag=latest
    [ -f "$PUB_OLLAMA/manifests/registry.ollama.ai/library/$name/$tag" ] && { ok "$m already in the shared folder"; continue; }
    say "Model $m"
    ollama pull "$m" && ok "$m"
  done
  say "Copying the downloaded models to $PUB_OLLAMA"
  for b in "$STAGE"/blobs/sha256-*; do
    case "$b" in *partial*) continue ;; esac
    t="$PUB_OLLAMA/blobs/$(basename "$b")"
    [ "$(stat -c %s "$t" 2>/dev/null)" = "$(stat -c %s "$b")" ] || { cp "$b" "$t.part" && mv "$t.part" "$t"; }
  done
  [ -d "$STAGE/manifests" ] && cp -r "$STAGE/manifests/." "$PUB_OLLAMA/manifests/"
  echo "  (the local copy in $STAGE can be deleted)"
  ls "$PUB_OLLAMA/manifests/registry.ollama.ai/library"/*/*
  chmod -R g+rwX,o+rX "$PUBLIC_LAB" 2>/dev/null || true
  ok "Shared folder ready: $PUBLIC_LAB"
  exit 0
fi

