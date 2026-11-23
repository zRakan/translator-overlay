"""Cost of the render and display steps, per component.
  python tools/render_bench.py -c config.capture.toml [--gui] [--frame tests/cropped.mp4]
cold  = a new translation arrives: build the Arabic bitmap (font, wrap, shape, draw) once
warm  = every displayed frame afterwards: place + blur backdrop + blend
scale = resizing the picture to the screen (cubic + sharpen / area)
--gui = imshow and waitKey on a real window (needs a desktop)."""
import argparse
import json
import statistics as st
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("tools", 1)[0])
from translator_overlay import config  # noqa: E402
from translator_overlay.overlay import Overlay  # noqa: E402


def timeit(f, n=200, warm=10):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return st.median(ts), ts[int(0.95 * (len(ts) - 1))], ts[-1]


def row(label, r):
    print(f"  {label:<46} median {r[0]:7.3f}   p95 {r[1]:7.3f}   max {r[2]:7.3f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--frame", default="tests/cropped.mp4")
    ap.add_argument("--gui", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    cap = cv2.VideoCapture(a.frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    lines = ["حسناً، سأكون هناك بعد قليل.", "هل تعرف أين يمكنني أن أجد سيارة أجرة؟", "لا تقلق، كل شيء تحت السيطرة الآن.", "أخبرني ماذا حدث بالضبط في ذلك المساء.",
             "إذا انتظرت هنا قليلاً فسأعود ومعي ما نحتاجه.", "لم أرَ أحداً يدخل أو يخرج من المبنى طوال الصباح.", "شكراً لك على كل ما فعلته من أجلنا.", "أين وضعت المفاتيح؟",
             "سنتحدث عن هذا لاحقاً، ليس الآن.", "هل أنت متأكد أنك تريد أن تفعل ذلك؟"] * 3  # a sample of typical subtitle lengths
    box, line_h = (int(0.22 * w), int(0.875 * h), int(0.74 * w), int(0.915 * h)), 34.0  # a typical subtitle box and text height at 1080p
    print(f"{sys.platform}  {w}x{h}  mode={cfg.overlay.mode} blur={cfg.overlay.blur} display.scale={cfg.display.scale}  {len(lines)} Arabic lines")

    ov = Overlay(cfg.overlay)
    print("cold: a new translation arrives (bitmap built once)")
    def cold(text):
        ov._cache.clear()
        ov.set(text, box, line_h)
        t = time.perf_counter()
        ov._place(h, w, 1.0)
        return (time.perf_counter() - t) * 1e3
    for text in lines[:5]:
        cold(text)  # settle imports / font cache
    ts = sorted(cold(t) for t in lines for _ in range(3))
    row(f"build bitmap, {len(lines)} real lines", (st.median(ts), ts[int(0.95 * (len(ts) - 1))], ts[-1]))
    from PIL import ImageFont
    row("  of which ImageFont.truetype() alone", timeit(lambda: ImageFont.truetype(ov.font_path, 28), 200))

    print("warm: every displayed frame while a subtitle is up")
    ov.set(lines[10], box, line_h)
    ov._place(h, w, 1.0)
    row("place (bitmap cached)", timeit(lambda: ov._place(h, w, 1.0)))
    bmp, x, y = ov._place(h, w, 1.0)
    print(f"  (bitmap {bmp.shape[1]}x{bmp.shape[0]} px)")
    roi = frame[y:y + bmp.shape[0], x:x + bmp.shape[1]].copy()
    row("blur backdrop only", timeit(lambda: ov._blur(roi)))
    f2 = frame.copy()
    row("draw() total, 1080p (place+blur+blend)", timeit(lambda: ov.draw(f2, 1.0)))
    ov.cfg.blur = False
    row("draw() total, 1080p, blur off (solid box)", timeit(lambda: ov.draw(f2, 1.0)))
    ov.cfg.blur = True
    row("patch() (click-through overlay window path)", timeit(lambda: ov.patch(frame, 1.0)))
    ov.current = None
    row("draw() with no subtitle (nothing to do)", timeit(lambda: ov.draw(f2, 1.0)))

    print("scale: picture to screen")
    sh = cfg.display.sharpen
    for name, (tw, th) in (("1440p", (2560, 1440)), ("4K", (3840, 2160))):
        row(f"cubic {w}x{h} -> {name}", timeit(lambda: cv2.resize(frame, (tw, th), interpolation=cv2.INTER_CUBIC), 60))
        big = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_CUBIC)
        row(f"  + sharpen at {name} (blur + addWeighted)", timeit(lambda: cv2.addWeighted(big, 1 + sh, cv2.GaussianBlur(big, (0, 0), 1.0), -sh, 0), 60))
    row("area 1080p -> 720p (window smaller)", timeit(lambda: cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA), 100))

    if a.gui:
        print("window: imshow + waitKey on a real window (1080p image)")
        cv2.namedWindow("bench", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("bench", 960, 540)
        im, wk = [], []
        for i in range(300):
            t = time.perf_counter()
            cv2.imshow("bench", frame)
            t1 = time.perf_counter()
            cv2.waitKey(1)
            t2 = time.perf_counter()
            if i >= 20:
                im.append((t1 - t) * 1e3)
                wk.append((t2 - t1) * 1e3)
        cv2.destroyAllWindows()
        for label, v in (("imshow", im), ("waitKey(1)", wk)):
            v.sort()
            row(label, (st.median(v), v[int(0.95 * (len(v) - 1))], v[-1]))
