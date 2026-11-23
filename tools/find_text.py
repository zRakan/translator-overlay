"""OCR the WHOLE frame of any source a few times and print every piece of text with its position as fractions of the frame
(x0 y0 x1 y1). Use it with a subtitle on screen to set [region] in the config.
python tools/find_text.py <source> [frames=8] [seconds_between=0.7]      source: cam:0:1920x1080@60:MJPG:msmf | window:<title> | file.mp4"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from translator_overlay.capture import open_source  # noqa: E402  (first: sets an OpenCV env var)
from translator_overlay.ocr import Ocr  # noqa: E402

import numpy as np  # noqa: E402

src = open_source(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
ocr = Ocr()
for _ in range(10):  # let the source settle
    src.read()
seen = {}
for i in range(n):
    ok, f = src.read()
    if not ok:
        print("no frame")
        break
    h, w = f.shape[:2]
    g = f.mean()
    res = ocr._engine(f, use_cls=False)
    items = []
    if res.txts is not None:
        for b, t, s in zip(res.boxes, res.txts, res.scores):
            b = np.asarray(b)
            items.append((round(b[:, 0].min() / w, 3), round(b[:, 1].min() / h, 3), round(b[:, 0].max() / w, 3), round(b[:, 1].max() / h, 3), t, round(float(s), 2)))
    print(f"frame {i}: {w}x{h}, mean brightness {g:.0f}, {len(items)} text boxes")
    for x0, y0, x1, y1, t, s in items:
        seen[(t, round(y0, 2))] = (x0, y0, x1, y1, s)
        print(f"   x {x0:.3f}-{x1:.3f}  y {y0:.3f}-{y1:.3f}  score {s:.2f}  {t!r}")
    time.sleep(gap)
src.release()
if seen:
    ys = sorted(v[1] for v in seen.values())
    print("\ntext seen between y =", ys[0], "and", max(v[3] for v in seen.values()))
