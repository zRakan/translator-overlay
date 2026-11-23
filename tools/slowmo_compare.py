"""ONNX/CUDA vs RapidOCR (TRT), rendered by the real pipeline: arrival analysis + slow-motion comparison video.
python slowmo.py CLIP.mp4 ONSETS.json A.mp4 B.mp4 OUT.mp4 FPS [PLAYBACK_FPS] [MAXN]"""
import sys, json, cv2, numpy as np, statistics as st, collections, subprocess
from PIL import Image, ImageDraw, ImageFont
clip, onj, pa, pb, out, fps = sys.argv[1:7]; fps = float(fps); play = float(sys.argv[7]) if len(sys.argv) > 7 else fps / 6; maxn = int(sys.argv[8]) if len(sys.argv) > 8 else 8
Y0, Y1, X0, X1 = 900, 1060, 260, 1660
def load_small(path):  # only a 140x16 grey thumbnail per frame (a full band per frame of a 4-minute clip would not fit in memory)
    c = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.resize(cv2.cvtColor(f[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY), (140, 16), interpolation=cv2.INTER_AREA).astype(np.int16))
    return fr
def load_some(path, idx):
    c = cv2.VideoCapture(path); out = {}; i = 0; last = max(idx)
    while i <= last:
        ok, f = c.read()
        if not ok: break
        if i in idx: out[i] = f[Y0:Y1, X0:X1].copy()
        i += 1
    return out
sa, sb = load_small(pa), load_small(pb); n = min(len(sa), len(sb)); print("frames", len(sa), len(sb))
onsets = json.load(open(onj)); win = int(1.2 * fps)
def arrival(bd):
    d = [0.0] + [float(np.abs(bd[i] - bd[i-1]).mean()) for i in range(1, len(bd))]; res = {}
    for o in onsets:
        ev = [i for i in range(o, min(o + win, len(d))) if d[i] > 2.0]; g = []
        for i in ev:
            if g and i - g[-1][-1] <= 1: g[-1].append(i)
            else: g.append([i])
        if len(g) >= 2: res[o] = g[1][0]
    return res
ra, rb = arrival(sa), arrival(sb); both = sorted(set(ra) & set(rb))
print(f"A) Arabic arrival after the English appears (frames @{fps:.0f} fps): resolved ONNX {len(ra)}, TensorRT {len(rb)}, both {len(both)} of {len(onsets)}")
fa, fb = [ra[o] - o for o in both], [rb[o] - o for o in both]
print(f"   ONNX     median +{st.median(fa):.0f} frames ({1e3*st.median(fa)/fps:.0f} ms), mean +{st.mean(fa):.1f}")
print(f"   TensorRT median +{st.median(fb):.0f} frames ({1e3*st.median(fb)/fps:.0f} ms), mean +{st.mean(fb):.1f}")
dd = collections.Counter(ra[o] - rb[o] for o in both); print("   per subtitle, ONNX arrival - TensorRT arrival (frames; + = TensorRT earlier):", dict(sorted(dd.items())))
# B) direct pairwise comparison of the two renders, per subtitle: the first frame where they differ, and which one just changed
verdict, unresolved = {}, []
for o in onsets:
    w = range(o, min(o + win, n)); dif = [f for f in w if float(np.abs(sa[f] - sb[f]).mean()) > 2.0]
    if not dif: verdict[o] = 0; continue
    end = min(o + win - 1, n - 1)
    if float(np.abs(sa[end] - sb[end]).mean()) > 1.0: unresolved.append(o); continue   # different Arabic text, never converges
    f = dif[0]; ca, cb = float(np.abs(sa[f] - sa[f-1]).mean()), float(np.abs(sb[f] - sb[f-1]).mean())
    verdict[o] = len(dif) if cb > ca else -len(dif)
same = sum(1 for d in verdict.values() if d == 0); trt = sum(1 for d in verdict.values() if d > 0); onx = sum(1 for d in verdict.values() if d < 0)
print(f"B) direct A-vs-B comparison of the two renders, per subtitle ({len(onsets)}): same {same}, TensorRT earlier {trt} (by {sorted(d for d in verdict.values() if d > 0)} frames), ONNX earlier {onx} (by {sorted(-d for d in verdict.values() if d < 0)}), different Arabic text {len(unresolved)}")
diff_text = len(unresolved)
both = [o for o in onsets if o in verdict]
for o in both: ra.setdefault(o, o); rb.setdefault(o, o)
# slow-motion video: the subtitles with the biggest differences first, then a few identical ones
pick = sorted(both, key=lambda o: (-abs(verdict[o]), o))
chosen = sorted(pick[:maxn - 2] + [o for o in pick if verdict[o] == 0][:2]) if len(pick) > maxn else sorted(pick)
W = 1120; sc = W / (X1 - X0); rh = int((Y1 - Y0) * sc)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
def row(img, label, color):
    im = Image.new("RGB", (W, rh + 26), (28, 28, 28)); d = ImageDraw.Draw(im); d.text((8, 4), label, fill=color, font=font)
    im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(img, (W, rh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)), (0, 26)); return np.asarray(im)
spans = {}
for o in chosen:
    dd_ = [f for f in range(o, min(o + win, n)) if float(np.abs(sa[f] - sb[f]).mean()) > 2.0]; spans[o] = (o - 2, (dd_[-1] if dd_ else max(ra[o], rb[o])) + 8)
need = {f for lo, hi in spans.values() for f in range(max(0, lo), min(n, hi + 1))}
S_, A, B = load_some(clip, need), load_some(pa, need), load_some(pb, need)
frames = []
for k, o in enumerate(chosen):
    lo = o - 2; dd_ = [f for f in range(o, min(o + win, n)) if float(np.abs(sa[f] - sb[f]).mean()) > 2.0]; hi = (dd_[-1] if dd_ else max(ra[o], rb[o])) + 8
    d = verdict[o]; msg = "same frame" if d == 0 else f"RapidOCR (TRT) shows the Arabic {d} frame{'s' if d > 1 else ''} earlier" if d > 0 else f"ONNX shows the Arabic {-d} frame{'s' if -d > 1 else ''} earlier"
    for f in range(max(0, lo), min(n, hi + 1)):
        head = Image.new("RGB", (W, 40), (20, 20, 20)); ImageDraw.Draw(head).text((8, 10), f"subtitle {k+1}/{len(chosen)}  t={o/fps:.1f}s  frame {f-o:+d} ({1e3*(f-o)/fps:+.0f} ms @{fps:.0f}fps)  [{msg}]", fill=(80, 255, 80), font=font)
        frames.append(np.vstack([np.asarray(head), row(S_[f], "SOURCE (English)", (255, 215, 0)), row(A[f], "ONNX / CUDA OCR", (255, 215, 0)), row(B[f], "RAPIDOCR (TRT)", (255, 215, 0))]))
    frames += [frames[-1]] * 2
h, w = frames[0].shape[:2]; tmp = out + ".tmp.mp4"
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), play, (w, h)); [vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) for f in frames]; vw.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
import os; os.remove(tmp); print("slowmo:", out, len(frames), "frames @", play, "fps =", round(len(frames) / play, 1), "s,", len(chosen), "subtitles")
json.dump({str(o): verdict[o] for o in both}, open(out + ".verdict.json", "w"))
