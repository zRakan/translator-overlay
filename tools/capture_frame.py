"""Windows only: capture a window for a few seconds and report whether the capture is usable (not black, changing,
frame rate, size); saves the last frame as a PNG.   python tools/capture_frame.py "<title piece>" [out.png] [seconds]"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from translator_overlay import winutil  # noqa: E402
from translator_overlay.capture import WindowSource  # noqa: E402

winutil.make_dpi_aware()
title, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "capture.png")
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 4
hwnd = winutil.find_window(title)
if not hwnd:
    sys.exit(f"no visible window with a title containing {title!r}")
left, top, right, bottom = winutil.frame_bounds(hwnd)
print(f"window hwnd={hwnd}  on-screen {right-left}x{bottom-top} at ({left},{top})")
src = WindowSource(hwnd=hwnd)
frames, t0, prev = [], time.perf_counter(), None
motion = []
while time.perf_counter() - t0 < secs:
    ok, f = src.read()
    if ok:
        frames.append(f)
        small = cv2.resize(f, (96, 54), interpolation=cv2.INTER_AREA).astype(np.int16)
        if prev is not None:
            motion.append(float(np.abs(small - prev).mean()))
        prev = small
n_new = src._seq
src.release()
if not frames:
    sys.exit("no frames were delivered")
f = frames[-1]
cv2.imwrite(out, f)
g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
print(f"frames delivered: {n_new} in {secs:.0f} s  ({n_new/secs:.0f} fps)   frame size {f.shape[1]}x{f.shape[0]}")
print(f"last frame: mean brightness {g.mean():.0f}/255, brightest pixel {g.max()}, contrast (std) {g.std():.0f}")
print(f"frame-to-frame change: mean {np.mean(motion):.2f}  (0 = frozen picture)" if motion else "only one frame arrived")
black = g.max() < 12 or g.std() < 2
print("VERDICT:", "BLACK/BLANK capture (content protection or wrong surface?)" if black else "capture looks usable (real picture)")
print("saved", out)
