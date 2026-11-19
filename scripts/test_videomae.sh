#!/bin/bash
# Usage: bash test_videomae.sh /path/to/clips_folder [base|small]
# Sets up harshakalluri1403/Violence-Detection, downloads its public weights,
# runs it on every clip in the folder, and keeps each clip's results separately.
set -e
CLIPS="${1:?Give the folder that holds the clips}"
MODEL="${2:-base}"
ROOT="$HOME/fight-tests"
mkdir -p "$ROOT" && cd "$ROOT"

python3 -c 'import sys; assert sys.version_info >= (3,10), "Need Python 3.10+"'

[ -d Violence-Detection ] || git clone https://github.com/harshakalluri1403/Violence-Detection.git
cd Violence-Detection
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
[ -f checkpoints/vdet_base.pt ] || python scripts/download_weights.py

EXTRA=""
if [ "$MODEL" = "small" ]; then
  EXTRA="--checkpoint checkpoints/vdet_small.pt --stride 1.0"
fi

RES="$ROOT/results"
mkdir -p "$RES"
shopt -s nullglob nocaseglob
n=0
for f in "$CLIPS"/*.mp4 "$CLIPS"/*.avi "$CLIPS"/*.mov "$CLIPS"/*.mkv; do
  name="$(basename "${f%.*}")"
  echo "=== $name"
  rm -rf out
  python detect.py "$f" $EXTRA || echo "FAILED: $name"
  mkdir -p "$RES/$name"
  [ -d out ] && cp -R out/. "$RES/$name/"
  n=$((n+1))
done
echo
echo "Done: $n clips. Results are in $RES/<clip name>/ (JSON incidents, annotated MP4, timeline plot)."
ls -R "$RES" | head -60
