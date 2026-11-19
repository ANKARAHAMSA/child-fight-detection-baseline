"""Builds assets/results.png and assets/overview.gif from results/*.csv (no model needed)."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
def load(n):
    return [(int(r["label"]), float(r["max_probability"])) for r in csv.DictReader(open(ROOT/"results"/n))]
D = {"SURV": load("surv_scores.csv"), "AIRTLab": load("airtlab_scores.csv")}

def metrics(rows, t):
    pos = [p for l, p in rows if l == 1]; neg = [p for l, p in rows if l == 0]
    tp = sum(p >= t for p in pos); fp = sum(p >= t for p in neg)
    return dict(acc=(tp + len(neg) - fp) / len(rows) * 100, caught=tp / len(pos) * 100,
                fa=fp / len(neg) * 100, tp=tp, npos=len(pos), fp=fp, nneg=len(neg))
M = {k: metrics(v, 0.6) for k, v in D.items()}
for k, m in M.items(): print(k, {a: round(b, 1) for a, b in m.items()})

BLUE, RED, GREEN, INK, MUTED, BG = "#2563eb", "#dc2626", "#16a34a", "#0f172a", "#64748b", "#ffffff"

# ---------- static chart ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
names = ["SURV\n(adult CCTV, 300 clips)", "AIRTLab\n(close contact, 350 clips)"]
x = range(2); w = 0.26
for i, (key, lab, col) in enumerate([("acc", "Accuracy", BLUE), ("caught", "Fights caught", GREEN), ("fa", "False alarms", RED)]):
    vals = [M["SURV"][key], M["AIRTLab"][key]]
    bars = ax[0].bar([j + (i - 1) * w for j in x], vals, w, label=lab, color=col)
    for b, v in zip(bars, vals): ax[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(names); ax[0].set_ylim(0, 108)
ax[0].set_title("Score level: clip flagged if any window >= 0.6", loc="left", fontweight="bold", fontsize=11)
ax[0].legend(frameon=False, loc="upper right", fontsize=9, ncol=3, bbox_to_anchor=(1, 1.0))
neg = [p for l, p in D["AIRTLab"] if l == 0]
ax[1].hist(neg, bins=20, range=(0, 1), color=RED, alpha=.85)
ax[1].axvline(0.6, color=INK, ls="--", lw=1); ax[1].text(0.58, ax[1].get_ylim()[1] * .92, "threshold 0.6", ha="right", fontsize=9)
ax[1].set_title("AIRTLab non-violent clips (hugs, high-fives):\nmodel is confidently wrong", loc="left", fontweight="bold", fontsize=11)
ax[1].set_xlabel("max fight score per clip"); ax[1].set_ylabel("clips")
for a in ax:
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.text(0.01, 0.015, "Alert level (what the tool actually raises): SURV caught only 2/150 fights (its clips are ~2 s); AIRTLab 65.7% accuracy, 34.2% false alarms.", fontsize=9, color=MUTED)
fig.savefig(ROOT/"assets/results.png", dpi=150); plt.close(fig)

# ---------- animated overview ----------
W, H = 960, 540
def font(sz, bold=False):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), sz)
def ctext(d, xy, s, f, fill):
    d.text(xy, s, font=f, fill=fill, anchor="mm")
boxes = [("CCTV clip", "adult test sets"), ("Sliding window", "3.2 s, 1 s stride"), ("VideoMAE-S", "public, adult-trained"),
         ("Fight score", "0 to 1 per window"), ("Alert?", "max score >= 0.6")]
bx, bw, bh, by = 40, 156, 120, 200
gap = (W - 80 - 5 * bw) / 4
pos = [(bx + i * (bw + gap), by) for i in range(5)]
def base(title, sub):
    im = Image.new("RGB", (W, H), "#f8fafc"); d = ImageDraw.Draw(im)
    d.text((40, 34), title, font=font(30, True), fill=INK); d.text((40, 78), sub, font=font(17), fill=MUTED)
    return im, d
frames, durs = [], []
# scene 1: packet travels through pipeline
for step in range(5):
    for sub in range(3):
        im, d = base("Child fight detection: baseline test", "Does a public adult-trained model work on CCTV-style clips?")
        for i, (a, b) in enumerate(boxes):
            x0, y0 = pos[i]; on = i <= step
            d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], 14, fill=BLUE if on else "#e2e8f0", outline=BLUE, width=2)
            ctext(d, (x0 + bw / 2, y0 + 44), a, font(17, True), "white" if on else MUTED)
            ctext(d, (x0 + bw / 2, y0 + 78), b, font(12), "white" if on else MUTED)
            if i < 4: d.line([x0 + bw + 6, y0 + bh / 2, x0 + bw + gap - 6, y0 + bh / 2], fill=BLUE if i < step else "#cbd5e1", width=4)
        if step < 4:
            t = sub / 3; px = pos[step][0] + bw + 6 + t * (gap - 12); d.ellipse([px - 8, by + bh / 2 - 8, px + 8, by + bh / 2 + 8], fill=RED)
        else:
            d.rounded_rectangle([pos[4][0], by + bh + 30, pos[4][0] + bw, by + bh + 74], 10, fill=RED if sub else "#fee2e2")
            ctext(d, (pos[4][0] + bw / 2, by + bh + 52), "ALERT", font(20, True), "white" if sub else RED)
        d.text((40, 440), "Tested on 650 labelled clips. No child footage used.", font=font(16), fill=MUTED)
        frames.append(im); durs.append(110 if step < 4 else 260)
# scene 2: bars grow
bars = [("SURV  score-level accuracy", 88.0, BLUE), ("SURV  fights alerted (~2 s clips)", 1.3, RED),
        ("AIRTLab  score-level accuracy", M["AIRTLab"]["acc"], BLUE), ("AIRTLab  alert-level accuracy", 65.7, BLUE),
        ("AIRTLab  alert false alarms", 34.2, RED)]
for k in list(range(0, 11)) + [10] * 10:
    f = min(k / 8, 1); f = 1 - (1 - f) ** 3
    im, d = base("Result: scores look fine, real alerts do not", "Public VideoMAE-S, threshold 0.6: score level vs alert level")
    for i, (lab, v, col) in enumerate(bars):
        y = 118 + i * 58
        d.text((40, y), lab, font=font(16, True), fill=INK)
        d.rounded_rectangle([360, y - 4, 900, y + 28], 8, fill="#e2e8f0")
        w_ = max(2, 540 * v / 100 * f); d.rounded_rectangle([360, y - 4, 360 + w_, y + 28], 8, fill=col)
        d.text((370 + w_ if w_ < 440 else 360 + w_ - 66, y + 1), f"{v * f:.0f}%", font=font(17, True), fill=INK if w_ < 440 else "white")
    d.text((40, 424), "Short ~2 s clips almost never raise an alert; hugs and high-fives do (34% of AIRTLab non-violent clips).", font=font(16), fill=INK)
    d.text((40, 456), "Next: play-fight class, real child-scene data, LLM check of flagged clips.", font=font(15), fill=MUTED)
    frames.append(im); durs.append(60 if k < 10 else 150)
durs[-1] = 3500
frames[0].save(ROOT/"assets/overview.gif", save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
print("done")
