#!/bin/bash
# Usage: bash eval_videomae.sh [dataset ...] [--model base|small] [--limit N]
# Datasets: SURV AIRTLab RLVS  (default: SURV AIRTLab)
# --limit N = only N videos per class per dataset (quick test); omit for all.
# Needs ~/fight-datasets (from earlier). Setup of the model repo is done here.
set -e
ROOT="$HOME/fight-tests"
mkdir -p "$ROOT" && cd "$ROOT"
python3 -c 'import sys; assert sys.version_info >= (3,10), "Need Python 3.10+"'
[ -d Violence-Detection ] || git clone https://github.com/harshakalluri1403/Violence-Detection.git
cd Violence-Detection
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f checkpoints/vdet_base.pt ] || python scripts/download_weights.py

cat > "$ROOT/eval_videomae.py" <<'PYEOF'
import os, sys, json, subprocess, argparse, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="*", default=["SURV", "AIRTLab"])
ap.add_argument("--model", default="base")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
DATA = Path.home() / "fight-datasets"
REPO = Path.home() / "fight-tests" / "Violence-Detection"
RES = Path.home() / "fight-tests" / "results_eval"
EXT = {".mp4", ".avi", ".mov", ".mkv"}
NEG = {"nofight", "nonviolent", "non-violent", "nonviolence", "normal"}
POS = {"fight", "violent", "violence"}
def label(p):
    parts = [x.lower() for x in p.parts]
    if any(x in NEG for x in parts): return 0
    if any(x in POS for x in parts): return 1
    return None
extra = []
if a.model == "small":
    extra = ["--checkpoint", "checkpoints/vdet_small.pt", "--stride", "1.0"]
summary = []
for ds in a.datasets:
    root = DATA / ds
    vids = [p for p in root.rglob("*") if p.suffix.lower() in EXT and "_duplicate" not in p.parts]
    by = {0: [], 1: []}
    skipped = 0
    for p in sorted(vids):
        l = label(p.relative_to(root))
        if l is None: skipped += 1
        else: by[l].append(p)
    if a.limit:
        by = {k: v[:a.limit] for k, v in by.items()}
    items = [(p, l) for l in (0, 1) for p in by[l]]
    print(f"\n### {ds}: {len(by[1])} violent, {len(by[0])} non-violent (skipped, unlabelled: {skipped})", flush=True)
    out = RES / f"{ds}_{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, (p, l) in enumerate(items, 1):
        stem = f"{i:04d}_{p.stem}"
        # unique name per video: copy-free via symlink
        link = out / f"{stem}{p.suffix}"
        if not link.exists():
            try: link.symlink_to(p)
            except FileExistsError: pass
        js = out / f"{stem}.json"
        if not js.exists():
            r = subprocess.run([sys.executable, "detect.py", str(link), "--out-dir", str(out), "--no-video"] + extra,
                               cwd=REPO, capture_output=True, text=True)
            if not js.exists():
                print("FAILED", p.name, r.stderr[-300:], flush=True); continue
        rep = json.load(open(js))
        rows.append((str(p.relative_to(DATA)), l, float(rep["max_probability"])))
        if i % 10 == 0: print(f"  {i}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
    csv = out / "scores.csv"
    csv.write_text("video,label,max_probability\n" + "\n".join(f"{v},{l},{s}" for v, l, s in rows))
    for th in (0.6, 0.9):
        tp = sum(1 for _, l, s in rows if l == 1 and s >= th)
        fn = sum(1 for _, l, s in rows if l == 1 and s < th)
        fp = sum(1 for _, l, s in rows if l == 0 and s >= th)
        tn = sum(1 for _, l, s in rows if l == 0 and s < th)
        n = tp + fn + fp + tn
        acc = (tp + tn) / n if n else 0
        rec = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        prec = tp / (tp + fp) if tp + fp else 0
        summary.append((ds, th, n, acc, rec, prec, fpr))
print("\n=== SUMMARY (per-video, max score vs threshold) ===")
print(f"{'dataset':10} {'thr':>4} {'n':>5} {'acc':>6} {'recall':>7} {'prec':>6} {'false-pos rate':>14}")
for ds, th, n, acc, rec, prec, fpr in summary:
    print(f"{ds:10} {th:>4} {n:>5} {acc:6.1%} {rec:7.1%} {prec:6.1%} {fpr:14.1%}")
print(f"\nPer-video scores saved under {RES}/<dataset>_{a.model}/scores.csv")
PYEOF

python "$ROOT/eval_videomae.py" "$@"
