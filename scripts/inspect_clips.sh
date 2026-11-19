#!/bin/bash
# Run the public VideoMAE model with Grad-CAM on a few clips and inspect the raw output.
# Usage: bash inspect_clips.sh <folder-with-clips>
# Output: ~/fight-tests/inspect/out  (JSON, timeline PNG, annotated video, Grad-CAM)
set -e
IN="${1:?usage: bash inspect_clips.sh <folder-with-clips>}"
cd ~/fight-tests/Violence-Detection && source .venv/bin/activate
for f in "$IN"/*.mp4; do
  python detect.py "$f" --checkpoint checkpoints/vdet_small.pt --stride 1.0 --explain \
    --out-dir ~/fight-tests/inspect/out
done
