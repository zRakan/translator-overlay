"""Windows only: does the OS keep the overlay out of screen captures?  Puts the overlay over a plain white window and
screen-grabs just the patch rectangle with the capture exclusion ON and OFF.  python tools/overlay_selfcheck.py"""
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import ImageGrab

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from translator_overlay import config, winutil  # noqa: E402
from translator_overlay.overlay import Overlay  # noqa: E402
from translator_overlay.winoverlay import OverlayWindow  # noqa: E402

winutil.make_dpi_aware()
root = tk.Tk()
root.title("TRO selfcheck")
root.geometry("900x300+60+60")
root.configure(bg="white")
root.update()
time.sleep(0.4)
hwnd = winutil.find_window("TRO selfcheck")
left, top, right, bottom = winutil.frame_bounds(hwnd)
blank = np.full((bottom - top, right - left, 3), 255, np.uint8)
ov = Overlay(config.OverlayCfg(mode="bar", bar_y=0.8))
ov.set("\u0627\u062e\u062a\u0628\u0627\u0631 \u0627\u0644\u062a\u0631\u062c\u0645\u0629", None, 30)
patch, fx, fy = ov.patch(blank)
h, w = patch.shape[:2]
box = (left + fx + 4, top + fy + 4, left + fx + w - 4, top + fy + h - 4)
for exclude in (True, False):
    win = OverlayWindow(hwnd, exclude_from_capture=exclude)
    for _ in range(10):
        win.show(patch, fx, fy)
        root.update()
        time.sleep(0.03)
    time.sleep(0.2)
    g = np.asarray(ImageGrab.grab(bbox=box, all_screens=True))
    print(f"exclusion {'ON ' if exclude else 'OFF'} (flag reports {win.excluded}): screen grab of the patch area, mean brightness {g.mean():.0f} of 255  "
          f"-> {'overlay NOT in the capture' if g.mean() > 240 else 'overlay IS in the capture'}")
    win.close()
root.destroy()
