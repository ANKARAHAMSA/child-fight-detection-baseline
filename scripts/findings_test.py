#!/usr/bin/env python3
"""Test the paper findings against the VideoMAE-S baseline on AIRTLab and SURV.

What it compares (per dataset):
  A  VideoMAE score level        (max window score >= 0.6)
  B  VideoMAE alert level        (the tool's own events, unchanged)
  C  VideoMAE alert, fixed       (single-window clips padded to the full window)
  D  C + two-person gate         (alert only if 2 people were ever close)
  E  Pose only                   (height-normalised pose features, logistic regression)
  F  Fusion                      (VideoMAE scores + pose features)
plus cross-dataset tests (train AIRTLab -> test SURV and back) and a three-tier count.

Usage (inside the Violence-Detection venv):
  python findings_test.py all --limit 10      # quick check, 10 clips per class
  python findings_test.py all                 # full run, resumable
  python findings_test.py report              # rebuild the report only
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np

HOME = Path.home()
EXT = {".mp4", ".avi", ".mov", ".mkv"}
NEG = {"nofight", "nonviolent", "non-violent", "nonviolence", "normal"}
POS = {"fight", "violent", "violence"}
WIN = 3.2


def first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]


ROOT = first_existing(HOME / "fight-tests", HOME / "Desktop" / "fight-tests")
DATA = first_existing(HOME / "fight-datasets", HOME / "Desktop" / "fight-datasets")
REPO = ROOT / "Violence-Detection"
OUT = ROOT / "findings_test"


def label_of(rel):
    parts = [x.lower() for x in rel.parts]
    if any(x in NEG for x in parts):
        return 0
    if any(x in POS for x in parts):
        return 1
    return None


def list_clips(ds, limit):
    root = DATA / ds
    by = {0: [], 1: []}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXT or "_duplicate" in p.parts:
            continue
        lab = label_of(p.relative_to(root))
        if lab is not None:
            by[lab].append(p)
    if limit:
        by = {k: v[:limit] for k, v in by.items()}
    clips = []
    for lab in (0, 1):
        for p in by[lab]:
            if ds.lower().startswith("airt"):
                # same scene from both cameras shares a group, so CV never splits it
                key = f"{lab}_{p.parent.name}_{p.stem}"
                group = f"{lab}_{p.stem}"
            else:
                key = f"{lab}_{p.stem}"
                group = key
            clips.append({"path": p, "label": lab, "key": key, "group": group})
    return clips


# ---------------------------------------------------------------- scoring
def run_videomae(ds, clips):
    sys.path.insert(0, str(REPO))
    from vdet.stream import ViolenceDetector
    out = OUT / f"vm_{ds}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips if not (out / f"{c['key']}.json").exists()]
    if not todo:
        return
    det = ViolenceDetector(str(REPO / "checkpoints" / "vdet_small.pt"), device="cpu", stride_sec=1.0)
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        tl = det.analyze(str(c["path"]), batch=8)
        json.dump({"t_end": tl.t_end.tolist(), "prob": tl.prob.tolist(), "duration": tl.duration,
                   "events": [e.to_dict() for e in tl.events()]}, open(out / f"{c['key']}.json", "w"))
        if i % 10 == 0 or i == len(todo):
            print(f"  VideoMAE {ds}: {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


def run_pose(ds, clips, device):
    import cv2
    from ultralytics import YOLO
    out = OUT / f"pose_{ds}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips if not (out / f"{c['key']}.json").exists()]
    if not todo:
        return
    model = YOLO("yolo11n-pose.pt")  # downloads ~6 MB on first use
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        cap = cv2.VideoCapture(str(c["path"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps / 10))  # ~10 frames per second
        model.predictor = None          # fresh tracker for every clip
        frames, k = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if k % step == 0:
                if fr.shape[1] > 640:
                    fr = cv2.resize(fr, (640, int(fr.shape[0] * 640 / fr.shape[1])))
                r = model.track(fr, persist=True, tracker="bytetrack.yaml", verbose=False,
                                imgsz=640, conf=0.25, device=device)[0]
                people = []
                if r.keypoints is not None and len(r.boxes):
                    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
                    for pid, b, kxy, kc in zip(ids, r.boxes.xyxy.tolist(), r.keypoints.xy.tolist(),
                                               r.keypoints.conf.tolist()):
                        people.append({"id": pid, "box": [round(v, 1) for v in b],
                                       "kp": [[round(x, 1), round(y, 1), round(cf, 2)]
                                              for (x, y), cf in zip(kxy, kc)]})
                frames.append({"t": round(k / fps, 3), "p": people})
            k += 1
        cap.release()
        json.dump({"fps": fps, "frames": frames}, open(out / f"{c['key']}.json", "w"))
        if i % 10 == 0 or i == len(todo):
            print(f"  Pose {ds}: {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- pose features
LIMBS = [9, 10, 15, 16]  # wrists, ankles (COCO-17)
KC = 0.3


def clip_features(d):
    T = {}
    for fr in d["frames"]:
        for p in fr["p"]:
            if p["id"] < 0:
                continue
            x1, y1, x2, y2 = p["box"]
            T.setdefault(p["id"], []).append((fr["t"], np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                                              max(y2 - y1, 1.0), np.array(p["kp"], float)))
    times = sorted({fr["t"] for fr in d["frames"]})
    H = {i: np.median([r[2] for r in v]) for i, v in T.items()}  # person height -> scale
    at = {}
    for i, v in T.items():
        for t, c, h, kp in v:
            at.setdefault(t, {})[i] = (c, kp)
    limb_sp, toward, torso, acc = {}, {}, [], []
    for i, v in T.items():
        sp_i = []
        for a, b in zip(v[:-1], v[1:]):
            dt = b[0] - a[0]
            if dt <= 0 or dt > 0.5:
                continue
            ka, kb = a[3], b[3]
            ok = (ka[LIMBS, 2] > KC) & (kb[LIMBS, 2] > KC)
            if ok.any():
                vel = (kb[LIMBS, :2] - ka[LIMBS, :2]) / dt / H[i]
                s = np.linalg.norm(vel, axis=1)[ok].max()
                limb_sp.setdefault(i, []).append((b[0], s))
                sp_i.append(s)
                others = [c for j, (c, _) in at.get(b[0], {}).items() if j != i]
                if others:
                    oc = min(others, key=lambda c: np.linalg.norm(c - b[1]))
                    u = oc - kb[LIMBS, :2]
                    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-6)
                    toward.setdefault(i, []).append((b[0], (vel * u).sum(1)[ok].max()))
            torso.append(np.linalg.norm(b[1] - a[1]) / dt / H[i])
        if len(sp_i) > 1:
            acc.extend(np.abs(np.diff(sp_i)))
    close_frames, min_d, n_people = [], [], []
    for t in times:
        ppl = at.get(t, {})
        n_people.append(len(ppl))
        ids, best, pair = list(ppl), np.inf, None
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                dd = np.linalg.norm(ppl[i][0] - ppl[j][0]) / ((H[i] + H[j]) / 2)
                if dd < best:
                    best, pair = dd, (i, j)
        min_d.append(best)
        close_frames.append((t, best < 1.0, pair))  # centres within ~1 body height
    close = np.array([c for _, c, _ in close_frames], bool)
    run = longest = 0
    for c in close:
        run = run + 1 if c else 0
        longest = max(longest, run)
    dtf = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    close_ids = {}
    for t, c, pair in close_frames:
        if c:
            close_ids.setdefault(t, set()).update(pair)
    sp_close = [s for i, v in limb_sp.items() for t, s in v if i in close_ids.get(t, ())]
    tw_close = [s for i, v in toward.items() for t, s in v if i in close_ids.get(t, ())]
    all_sp = [s for v in limb_sp.values() for _, s in v]

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else 0.0

    md = np.array([m for m in min_d if np.isfinite(m)])
    return {
        "max_people": float(max(n_people) if n_people else 0),
        "frac_2plus": float(np.mean(np.array(n_people) >= 2)) if n_people else 0.0,
        "frac_close": float(close.mean()) if len(close) else 0.0,
        "longest_close_s": float(longest * dtf),
        "min_pair_dist": float(md.min()) if len(md) else 5.0,
        "med_pair_dist": float(np.median(md)) if len(md) else 5.0,
        "limb_sp_p90": q(all_sp, 90), "limb_sp_max": q(all_sp, 100),
        "limb_acc_p90": q(acc, 90), "torso_sp_p90": q(torso, 90),
        "close_limb_sp_p90": q(sp_close, 90), "close_limb_sp_max": q(sp_close, 100),
        "close_toward_p90": q(tw_close, 90), "close_toward_max": q(tw_close, 100),
        "fast_close_s": float(np.sum(np.array(sp_close) > 3.0) * dtf),
    }


POSE = ["max_people", "frac_2plus", "frac_close", "longest_close_s", "min_pair_dist", "med_pair_dist",
        "limb_sp_p90", "limb_sp_max", "limb_acc_p90", "torso_sp_p90", "close_limb_sp_p90",
        "close_limb_sp_max", "close_toward_p90", "close_toward_max", "fast_close_s"]
VM = ["vm_max", "vm_mean"]


# ---------------------------------------------------------------- evaluation
def events_fixed(t_end, prob):
    sys.path.insert(0, str(REPO))
    from vdet.stream import extract_events
    t_end, prob = np.array(t_end), np.array(prob)
    if len(prob) == 0:
        return []
    k = 3
    sm = prob if len(prob) < k else np.convolve(np.pad(prob, (1, 1), mode="edge"), np.ones(k) / k, "valid")
    tc = np.clip(t_end - WIN / 2, 0, None)
    pad = (t_end[1] - t_end[0]) / 2 if len(t_end) > 1 else WIN / 2  # the fix
    return extract_events(tc, sm, 0.6, 0.4, 0.8, 1.5, pad=pad)


def load_table(ds, clips):
    import pandas as pd
    rows = []
    for c in clips:
        vj, pj = OUT / f"vm_{ds}" / f"{c['key']}.json", OUT / f"pose_{ds}" / f"{c['key']}.json"
        if not vj.exists() or not pj.exists():
            continue
        v = json.load(open(vj))
        p = np.array(v["prob"])
        r = {"key": c["key"], "label": c["label"], "group": c["group"], "dur": v["duration"],
             "n_win": len(p), "vm_max": float(p.max()) if len(p) else 0.0,
             "vm_mean": float(p.mean()) if len(p) else 0.0,
             "alert_orig": int(len(v["events"]) > 0),
             "alert_fixed": int(len(events_fixed(v["t_end"], v["prob"])) > 0)}
        r.update(clip_features(json.load(open(pj))))
        rows.append(r)
    return pd.DataFrame(rows)


RNG = np.random.default_rng(0)


def summarize(y, pred, groups):
    y, pred, groups = np.asarray(y), np.asarray(pred).astype(int), np.asarray(groups)

    def m(idx):
        yy, pp = y[idx], pred[idx]
        tp, fn = ((yy == 1) & (pp == 1)).sum(), ((yy == 1) & (pp == 0)).sum()
        fp, tn = ((yy == 0) & (pp == 1)).sum(), ((yy == 0) & (pp == 0)).sum()
        return np.array([tp / max(tp + fn, 1), fp / max(fp + tn, 1)])

    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    bs = np.array([m(np.concatenate([gi[g] for g in RNG.choice(ug, len(ug))])) for _ in range(1000)])
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    tp, fp = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum())
    P, N = int((y == 1).sum()), int((y == 0).sum())
    acc = (tp + (N - fp)) / max(P + N, 1)
    prec = tp / max(tp + fp, 1)
    return (f"acc {acc:6.1%} | caught {tp:3d}/{P} {tp / max(P, 1):6.1%} [{lo[0]:.0%}-{hi[0]:.0%}] | "
            f"false alarms {fp:3d}/{N} {fp / max(N, 1):6.1%} [{lo[1]:.0%}-{hi[1]:.0%}] | precision {prec:6.1%}")


def make_clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))


def oof(df, cols, reps=5):
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, g = df[cols].values, df.label.values, df.group.values
    out = np.zeros(len(df))
    for s in range(reps):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            out[te] += make_clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] / reps
    return out


def ap(y, s):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s) if len(set(y)) == 2 else float("nan")


def report(datasets, limit):
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    tables = {}
    for ds in datasets:
        df = load_table(ds, list_clips(ds, limit))
        tables[ds] = df
        if df.empty:
            say(f"\n## {ds}: no scored clips yet")
            continue
        df.to_csv(OUT / f"features_{ds}.csv", index=False)
        y, g = df.label.values, df.group.values
        say(f"\n## {ds}: {int(y.sum())} fights, {int((1 - y).sum())} non-fights, "
            f"median length {df.dur.median():.1f}s, clips with one window {int((df.n_win <= 1).sum())}")
        say("   [brackets] = 95% bootstrap interval, resampled by scene")
        say(f"A VideoMAE score>=0.6        {summarize(y, df.vm_max >= 0.6, g)}")
        say(f"B VideoMAE alert (as is)     {summarize(y, df.alert_orig, g)}")
        say(f"C VideoMAE alert (fixed)     {summarize(y, df.alert_fixed, g)}")
        gate = df.frac_close > 0
        say(f"D C + two-person gate        {summarize(y, df.alert_fixed & gate, g)}")
        enough = min(int(y.sum()), int((1 - y).sum())) >= 10
        if enough:
            df["pose_p"] = oof(df, POSE)
            df["fuse_p"] = oof(df, VM + POSE)
            say(f"E Pose only (5-fold CV)      {summarize(y, df.pose_p >= 0.5, g)}")
            say(f"F Fusion (5-fold CV)         {summarize(y, df.fuse_p >= 0.5, g)}")
            say(f"  Average precision: VideoMAE {ap(y, df.vm_max):.3f} | pose {ap(y, df.pose_p):.3f} | "
                f"fusion {ap(y, df.fuse_p):.3f}")
        else:
            say("E/F skipped: need at least 10 clips per class for cross-validation")
        nf, f = df[df.label == 0], df[df.label == 1]
        say("  Three tiers on VideoMAE max score (fights | non-fights):")
        for lo_, hi_, name in ((0.9, 1.01, "danger  >=0.9"), (0.6, 0.9, "warning 0.6-0.9"), (0, 0.6, "normal  <0.6")):
            say(f"    {name}: {int(((f.vm_max >= lo_) & (f.vm_max < hi_)).sum()):3d} | "
                f"{int(((nf.vm_max >= lo_) & (nf.vm_max < hi_)).sum()):3d}")

    names = [d for d in datasets if not tables[d].empty]
    if len(names) == 2 and all(min(int(tables[d].label.sum()), int((1 - tables[d].label).sum())) >= 10 for d in names):
        say("\n## Cross-dataset (train on one, test on the other; threshold 0.5)")
        for a, b in (names, names[::-1]):
            ta, tb = tables[a], tables[b]
            for cols, nm in ((POSE, "pose  "), (VM + POSE, "fusion")):
                p = make_clf().fit(ta[cols].values, ta.label.values).predict_proba(tb[cols].values)[:, 1]
                say(f"{nm} {a}->{b}: {summarize(tb.label.values, p >= 0.5, tb.group.values)}  AP {ap(tb.label.values, p):.3f}")
            say(f"VideoMAE fixed alert on {b} for reference: "
                f"{summarize(tb.label.values, tb.alert_fixed, tb.group.values)}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved {OUT / 'report.txt'} and features_*.csv")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("step", choices=["all", "score", "pose", "report"])
    ap_.add_argument("--datasets", nargs="*", default=["AIRTLab", "SURV"])
    ap_.add_argument("--limit", type=int, default=0, help="clips per class (0 = all)")
    ap_.add_argument("--pose-device", default=None, help="mps or cpu (default: mps if available)")
    a = ap_.parse_args()
    if a.limit:
        global OUT
        OUT = OUT.parent / f"findings_test_limit{a.limit}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"root {ROOT}\ndata {DATA}\nout  {OUT}")
    for need in (REPO / "checkpoints" / "vdet_small.pt", DATA):
        if not need.exists():
            sys.exit(f"missing: {need}")
    dev = a.pose_device
    if dev is None:
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for ds in a.datasets:
        clips = list_clips(ds, a.limit)
        print(f"\n{ds}: {sum(c['label'] for c in clips)} fights, {sum(1 - c['label'] for c in clips)} non-fights")
        if a.step in ("all", "score"):
            run_videomae(ds, clips)
        if a.step in ("all", "pose"):
            try:
                run_pose(ds, clips, dev)
            except Exception as e:
                if dev == "cpu":
                    raise
                print(f"  pose on {dev} failed ({e}); retrying on cpu")
                run_pose(ds, clips, "cpu")
    if a.step in ("all", "report"):
        report(a.datasets, a.limit)


if __name__ == "__main__":
    main()
