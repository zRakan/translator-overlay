"""Windows only: put a sample Arabic subtitle over a window and check the overlay behaves.
  python tools/overlay_demo.py "<window title>" [--visible-shot out.png]
Checks: it is click-through (WindowFromPoint skips it), takes no focus, follows the window, and is excluded from
capture. --visible-shot disables the capture exclusion for a moment and saves a small screenshot of the patch area
(only that rectangle) so you can see it."""
import ctypes
import ctypes.wintypes as wt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from translator_overlay import config, winutil  # noqa: E402
from translator_overlay.capture import WindowSource  # noqa: E402
from translator_overlay.overlay import Overlay  # noqa: E402
from translator_overlay.winoverlay import OverlayWindow  # noqa: E402

title = sys.argv[1]
shot = sys.argv[sys.argv.index("--visible-shot") + 1] if "--visible-shot" in sys.argv else None
winutil.make_dpi_aware()
user32 = winutil.user32
user32.WindowFromPoint.argtypes = [wt.POINT]
user32.WindowFromPoint.restype = wt.HWND
user32.GetForegroundWindow.restype = wt.HWND
user32.GetAncestor.argtypes = [wt.HWND, wt.UINT]
user32.GetAncestor.restype = wt.HWND
target = winutil.find_window(title)
assert target, "window not found"
src = WindowSource(hwnd=target)
ov = Overlay(config.OverlayCfg(mode="bar"))
ov.set("\u0623\u0647\u0644\u0627\u064b \u0628\u0643 \u0641\u064a \u0627\u0644\u0644\u0639\u0628\u0629\u060c \u0647\u0630\u0627 \u0627\u062e\u062a\u0628\u0627\u0631 \u0644\u0644\u062a\u0631\u062c\u0645\u0629 \u0627\u0644\u0641\u0648\u0631\u064a\u0629", None, 28)
fg_before = user32.GetForegroundWindow()
win = OverlayWindow(target, exclude_from_capture=shot is None)
print("capture exclusion active:", win.excluded if shot is None else "off for the screenshot")
for _ in range(30):
    ok, frame = src.read()
    patch = ov.patch(frame)
    win.show(*patch)
    time.sleep(0.016)
time.sleep(0.2)
patch, fx, fy = ov.patch(frame)
left, top, _, _ = winutil.frame_bounds(target)
sx, sy = left + fx, top + fy
h, w = patch.shape[:2]
pt = wt.POINT(sx + w // 2, sy + h // 2)
hit = user32.WindowFromPoint(pt)
root = user32.GetAncestor(hit, 2) if hit else None  # GA_ROOT
print(f"patch {w}x{h} at screen ({sx},{sy}); window under the patch centre is the {'GAME window (click-through OK)' if root == target else 'OVERLAY or other window (NOT click-through)'}")
print("focus unchanged:", user32.GetForegroundWindow() == fg_before)
from PIL import ImageGrab
img = ImageGrab.grab(bbox=(sx - 10, sy - 10, sx + w + 10, sy + h + 10), all_screens=True)  # only the patch rectangle
if shot:
    img.save(shot)
    print("saved", shot)
else:
    import numpy as np
    print("mean brightness of the screen grab at the patch (a visible dark subtitle box lowers it):", round(float(np.asarray(img).mean()), 1))
win.close()
src.release()
