"""Check one video: does VideoMAE raise an alert, and does Qwen confirm it?
Also asks Qwen for a scene description (school-safety prompt) and shows it under the video.
Saves an annotated video, a score timeline, the description and a JSON result, then opens the video.
Runs fully on the Mac. Usage:
  python check_clip.py ~/path/to/clip.mp4
  python check_clip.py clip.mp4 --always-ask     # ask Qwen even if VideoMAE stays quiet
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import findings_test as ft
import verify_vlm as vv

GREEN, RED, GREY = (80, 175, 76), (40, 40, 220), (90, 90, 90)

SCENE_PROMPT = (
    "You are a school safety monitoring AI. Analyze these surveillance frames from camera {feed_id} "
    "at timestamps {t_start}-{t_end}.\n"
    "Describe what is happening. Focus on:\n"
    "1. Physical interactions between people (fighting, shoving, grabbing)\n"
    "2. Medical emergencies (collapse, unconsciousness, injury)\n"
    "3. Crowd dynamics (surrounding, stampede, panic)\n"
    "4. Weapons or dangerous objects\n"
    "5. Unauthorized access or trespassing\n"
    "Keep it to 3-5 short sentences."
)


def make_runner(model_id):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    model, processor = load(model_id)
    config = load_config(model_id)

    def run(prompt, images, max_tokens=150):
        formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
        try:
            out = generate(model, processor, formatted, images, max_tokens=max_tokens, temperature=0.0, verbose=False)
        except TypeError:
            out = generate(model, processor, formatted, images, max_tokens=max_tokens, verbose=False)
        return getattr(out, "text", out)
    return run


def clean(text):
    return " ".join(text.replace("**", "").replace("#", "").split())


def stamp(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def frames_between(path, t0, t1, n, max_side, tmp):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    a, b = int(max(t0, 0) * fps), min(int(t1 * fps), total - 1)
    files = []
    for k, i in enumerate(np.linspace(a, max(a, b), n).round().astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
        f = Path(tmp) / f"f{k:02d}.jpg"
        cv2.imwrite(str(f), fr)
        files.append(str(f))
    cap.release()
    return files


def bottom_bar(frame, text, color):
    h, w = frame.shape[:2]
    bh = max(28, round(h * 0.08))
    fs = bh / 40
    over = frame.copy()
    cv2.rectangle(over, (0, h - bh), (w, h), color, -1)
    frame[:] = cv2.addWeighted(over, 0.85, frame, 0.15, 0)
    text = text.encode("ascii", "replace").decode()
    max_chars = int(w / (fs * 19))
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."
    cv2.putText(frame, text, (10, h - bh // 2 + int(8 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), max(1, int(fs * 2)), cv2.LINE_AA)


def text_panel(width, lines_text, fs, n_lines):
    import textwrap
    line_h = int(28 * fs)
    panel = np.full((line_h * n_lines + 16, width, 3), 24, np.uint8)
    for k, line in enumerate(lines_text[:n_lines]):
        cv2.putText(panel, line, (10, 8 + line_h * (k + 1) - int(8 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return panel


def wrap(text, width, fs):
    import textwrap
    text = text.encode("ascii", "replace").decode()
    return textwrap.wrap(text, max(20, int(width / (fs * 21))))


def render(src, out_path, tl, events, checks, final, final_color, max_width=960):
    from vdet.viz import _hud, _timeline_strip, _writer
    cap = cv2.VideoCapture(str(src))
    fps = tl.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / max(W, 1))
    w, h = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    strip_h = max(32, h // 9) // 2 * 2
    fs = max(0.45, min(0.7, w / 1100))
    panels = {}
    for k, c in enumerate(checks):
        if c.get("scene"):
            panels[k] = ["SCENE " + stamp(c["from"]) + "-" + stamp(c["to"]) + ":"] + wrap(c["scene"], w, fs)
    n_lines = min(12, max([len(v) for v in panels.values()], default=0))
    panel_h = text_panel(w, [], fs, n_lines).shape[0] if n_lines else 0
    write, close = _writer(str(out_path), (w, h + strip_h + panel_h), fps)
    strip = _timeline_strip(tl, events, w, strip_h, 0.6)
    shown = min(panels) if panels else None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        alarm = any(e.start <= t <= e.end for e in events)
        frame = np.ascontiguousarray(_hud(frame, t, tl.score_at(t), alarm))
        active = [k for k, c in enumerate(checks) if c["from"] <= t <= c["to"]]
        if active:
            c = checks[active[0]]
            if active[0] in panels:
                shown = active[0]
        if active and "fight" in c:
            bottom_bar(frame, f"Qwen: {'FIGHT' if c['fight'] else 'not a fight'} "
                              f"({c['confidence']:.2f}) - {c['reason']}", RED if c["fight"] else GREEN)
        else:
            bottom_bar(frame, final, final_color)
        s = strip.copy()
        cx = int(np.clip(t / max(tl.duration, 1e-3) * (w - 1), 0, w - 1))
        s[:, cx + 1:] = (s[:, cx + 1:] * 0.5).astype(np.uint8)
        cv2.line(s, (cx, 0), (cx, strip_h - 1), (255, 255, 255), 2)
        parts = [frame, s]
        if n_lines:
            parts.append(text_panel(w, panels[shown], fs, n_lines))
        write(np.vstack(parts))
        i += 1
    cap.release()
    close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    ap.add_argument("--prompt", choices=list(vv.PROMPTS), default="simple")
    ap.add_argument("--always-ask", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open the video when done")
    ap.add_argument("--feed-id", default=None, help="camera name for the scene description (default: file name)")
    ap.add_argument("--no-describe", action="store_true", help="skip the scene description")
    a = ap.parse_args()
    path = Path(a.video).expanduser()
    if not path.exists():
        sys.exit(f"not found: {path}")
    out_dir = Path(__file__).resolve().parent / "clip_checks" / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ft.REPO))
    from vdet.stream import ViolenceDetector
    from vdet.viz import plot_timeline
    det = ViolenceDetector(str(ft.REPO / "checkpoints" / "vdet_small.pt"), device="cpu", stride_sec=1.0)
    tl = det.analyze(str(path), batch=8)
    events = tl.events()
    top = float(np.max(tl.prob)) if len(tl.prob) else 0.0
    print(f"\nClip: {path.name}  ({tl.duration:.1f}s)")
    print(f"Stage 1  VideoMAE: highest score {top:.2f}, alerts: {len(events)}")
    for e in events:
        print(f"   alert {e.start:.1f}s - {e.end:.1f}s  peak {e.to_dict()['peak']:.2f}")

    segs = [(max(e.start - 1, 0), min(e.end + 1, tl.duration)) for e in events]
    if not segs and (a.always_ask or not a.no_describe):
        segs = [(0, tl.duration)]
    ask_verdict = bool(events) or a.always_ask
    checks = []
    if segs:
        run = make_runner(a.model)
        feed = a.feed_id or path.stem[:40]
        for t0, t1 in segs:
            c = {"from": t0, "to": t1}
            with tempfile.TemporaryDirectory() as tmp:
                imgs = frames_between(path, t0, t1, 8, 448, tmp)
                if ask_verdict:
                    text = run(vv.PROMPTS[a.prompt].format(n=len(imgs), d=max(t1 - t0, 1)), imgs)
                    fight, conf, reason = vv.parse(text)
                    c.update({"fight": fight, "confidence": conf, "reason": reason})
                    print(f"Stage 2  Qwen on {t0:.1f}s - {t1:.1f}s: {'FIGHT' if fight else 'not a fight'} "
                          f"(confidence {conf:.2f}) - {reason}")
                if not a.no_describe:
                    scene = clean(run(SCENE_PROMPT.format(feed_id=feed, t_start=stamp(t0), t_end=stamp(t1)),
                                      imgs, max_tokens=300))
                    c["scene"] = scene
                    print(f"Scene    {stamp(t0)}-{stamp(t1)}: {scene}")
            checks.append(c)
    else:
        print("Stage 2  Qwen: not asked (no alert). Add --always-ask to hear Qwen anyway.")

    confirmed = any(c.get("fight") for c in checks)
    asked = any("fight" in c for c in checks)
    if events:
        final = "FINAL: FIGHT ALERT" if confirmed else "FINAL: alert dismissed by Qwen"
    else:
        final = "FINAL: no fight" + (f" (Qwen alone says {'fight' if confirmed else 'no fight'})" if asked else "")
    color = RED if (events and confirmed) else GREY if events else GREEN
    print(final)

    video = out_dir / f"{path.stem}_checked.mp4"
    print("Writing annotated video...")
    render(path, video, tl, events, checks, final, color)
    plot_timeline(tl, events, 0.6, title=path.name, path=str(out_dir / f"{path.stem}_timeline.png"))
    (out_dir / "scene_description.txt").write_text(
        "\n\n".join(f"{stamp(c['from'])}-{stamp(c['to'])}: {c['scene']}" for c in checks if c.get("scene")) + "\n")
    json.dump({"clip": str(path), "duration": tl.duration, "videomae_max": top,
               "alerts": [e.to_dict() for e in events], "qwen": checks, "final": final,
               "timeline": {"t": np.round(tl.t_center, 2).tolist(), "p": np.round(tl.prob, 3).tolist()}},
              open(out_dir / "result.json", "w"), indent=1)
    print(f"Saved in {out_dir}")
    if not a.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(video)])


if __name__ == "__main__":
    main()
