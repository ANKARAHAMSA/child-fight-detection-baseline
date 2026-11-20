"""Second-stage check: a local vision-language model looks at every VideoMAE alert.

Runs fully on the Mac (Apple silicon, MLX). The only download is the model itself;
no frames or video leave the machine.

Needs findings_test.py in the same folder and its output (features_*.csv).

Usage (inside the Violence-Detection venv, after `pip install -U mlx-vlm`):
  python verify_vlm.py --limit 5                       # quick check: 5 fight + 5 non-fight alerts
  python verify_vlm.py                                 # all AIRTLab alerts (resumable)
  python verify_vlm.py --datasets SURV
  python verify_vlm.py --model mlx-community/Qwen3-VL-4B-Instruct-4bit   # for 8 GB Macs
  python verify_vlm.py --prompt simple                 # plain yes/no prompt, for comparison
"""
import argparse, json, re, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import findings_test as ft  # noqa: E402

PROMPTS = {
    "cues": (
        "These {n} frames are in time order from a {d:.0f}-second CCTV clip.\n"
        "Decide whether a real fight or assault is happening: someone trying to hurt another person "
        "(punching, kicking, slapping, shoving, choking, or hitting, stabbing or shooting with a weapon).\n"
        "Friendly or harmless contact is NOT a fight: hugging, greeting, high-fives, handshakes, "
        "dancing, jumping, horseplay where everyone looks relaxed.\n"
        "Look for these cues: Are faces relaxed or smiling? Does anyone try to escape, cover up or fall? "
        "Are blows forceful and aimed at someone? Do people take turns as in a game? Is a weapon visible?\n"
        'Reply with JSON only: {{"fight": true or false, "confidence": 0.0 to 1.0, '
        '"reason": "one short sentence"}}'
    ),
    "simple": (
        "These {n} frames are in time order from a {d:.0f}-second CCTV clip. Is there violence? "
        'Reply with JSON only: {{"fight": true or false, "confidence": 0.0 to 1.0, '
        '"reason": "one short sentence"}}'
    ),
}


def sample_frames(path, n, max_side, tmpdir):
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    idx = np.linspace(0, max(total - 1, 0), n).round().astype(int)
    files = []
    for k, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
        f = Path(tmpdir) / f"f{k:02d}.jpg"
        cv2.imwrite(str(f), fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        files.append(str(f))
    cap.release()
    return files, total / fps


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            return bool(j.get("fight")), float(j.get("confidence", 0.5)), str(j.get("reason", ""))
        except Exception:
            pass
    t = text.lower()
    fight = bool(re.search(r'"?fight"?\s*:\s*true', t)) or t.strip().startswith("yes")
    return fight, 0.5, "unparsed: " + text[:120]


def make_runner(model_id):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    model, processor = load(model_id)
    config = load_config(model_id)

    def run(prompt, images):
        formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
        try:
            out = generate(model, processor, formatted, images, max_tokens=150, temperature=0.0, verbose=False)
        except TypeError:
            out = generate(model, processor, formatted, images, max_tokens=150, verbose=False)
        return getattr(out, "text", out)
    return run


def action_labels():
    """AIRTLab ships per-clip action classes; used only to break down false alarms."""
    found = list(ft.DATA.rglob("nonviolent-action-classes.csv"))
    if not found:
        return {}
    df = pd.read_csv(found[0], sep=";")
    df.columns = ["file", "acts"]
    return {Path(f).stem: str(a).split(",") for f, a in zip(df.file, df.acts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["AIRTLab"])
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="cues")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="alerts per class to check (0 = all)")
    a = ap.parse_args()

    tag = f"{a.model.split('/')[-1]}_{a.prompt}_{a.frames}f"
    run = None
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    for ds in a.datasets:
        csv = ft.OUT / f"features_{ds}.csv"
        if not csv.exists():
            sys.exit(f"missing {csv}: run findings_test.py first")
        df = pd.read_csv(csv)
        paths = {c["key"]: c["path"] for c in ft.list_clips(ds, 0)}
        alerts = df[df.alert_fixed == 1]
        if a.limit:
            alerts = pd.concat([alerts[alerts.label == 1].head(a.limit), alerts[alerts.label == 0].head(a.limit)])
        out = ft.OUT / f"vlm_{tag}_{ds}"
        out.mkdir(parents=True, exist_ok=True)
        todo = [k for k in alerts.key if not (out / f"{k}.json").exists()]
        print(f"\n{ds}: {len(alerts)} alerts to verify, {len(todo)} not done yet -> {out}")
        t0 = time.time()
        for i, key in enumerate(todo, 1):
            if run is None:
                print(f"loading {a.model} (first run downloads it once)...", flush=True)
                run = make_runner(a.model)
            with tempfile.TemporaryDirectory() as tmp:
                imgs, dur = sample_frames(paths[key], a.frames, a.max_side, tmp)
                s = time.time()
                text = run(PROMPTS[a.prompt].format(n=len(imgs), d=max(dur, 1)), imgs)
            fight, conf, reason = parse(text)
            json.dump({"fight": fight, "confidence": conf, "reason": reason, "raw": text,
                       "secs": round(time.time() - s, 1)}, open(out / f"{key}.json", "w"))
            if i % 5 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el:.0f}s  (~{el / i:.1f}s per clip)", flush=True)

        # ---------------- report: VideoMAE alert (fixed) vs alert AND VLM says fight
        res = {}
        for k in alerts.key:
            f = out / f"{k}.json"
            if f.exists():
                res[k] = json.load(open(f))
        df["vlm_fight"] = df.key.map(lambda k: res[k]["fight"] if k in res else np.nan)
        chk = df[df.key.isin(res)]
        say(f"\n## {ds} - {a.model.split('/')[-1]}, prompt '{a.prompt}', {a.frames} frames "
            f"({len(chk)} alerts checked, median {np.median([r['secs'] for r in res.values()]):.1f}s each)")
        if chk.empty:
            continue
        fa, tf = chk[chk.label == 0], chk[chk.label == 1]
        say(f"False alarms the VLM removed: {int((fa.vlm_fight == 0).sum())}/{len(fa)}")
        say(f"Real fights the VLM wrongly removed: {int((tf.vlm_fight == 0).sum())}/{len(tf)}")
        if not a.limit:
            y, g = df.label.values, df.group.values
            two = df.alert_fixed.astype(bool) & (df.vlm_fight.fillna(0) == 1)
            say(f"C VideoMAE alert (fixed)    {ft.summarize(y, df.alert_fixed, g)}")
            say(f"G C + VLM verification      {ft.summarize(y, two, g)}")
        if ds.lower().startswith("airt"):
            acts = action_labels()
            if acts:
                fa = fa.assign(cid=fa.key.str.split("_").str[-1])
                say("False alarms by action (VideoMAE alerts -> left after VLM):")
                for act in sorted({x for v in acts.values() for x in v}):
                    m = fa.cid.map(lambda c: act in acts.get(c, []))
                    if m.sum():
                        say(f"  {act:14s} {int(m.sum()):3d} -> {int((fa[m].vlm_fight == 1).sum())}")
        wrong = chk[(chk.vlm_fight == 0) & (chk.label == 1)].key.head(5).tolist()
        for k in wrong:
            say(f"  missed fight {k}: {res[k]['reason']}")

    rep = ft.OUT / f"vlm_report_{tag}.txt"
    rep.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {rep}")


if __name__ == "__main__":
    main()
