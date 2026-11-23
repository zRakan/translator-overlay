"""Windows only: measure the live-capture path against a test window we draw ourselves.
  1) capture latency: flip the window black/white, time until a captured frame shows it
  2) delivery rate while the window repaints continuously
python tools/capture_probe.py"""
import statistics as st
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from translator_overlay.capture import WindowSource  # noqa: E402

TITLE = "TRO capture test"
root = tk.Tk()
root.title(TITLE)
root.geometry("640x360+40+40")
canvas = tk.Canvas(root, width=640, height=360, highlightthickness=0, bg="black")
canvas.pack()
root.update()
time.sleep(0.5)
src = WindowSource(TITLE)
for _ in range(5):                                  # let capture spin up
    src.read()
    root.update()


def paint(white: bool):
    canvas.configure(bg="white" if white else "black")
    root.update_idletasks()
    root.update()


lat = []
white = False
for _ in range(24):
    time.sleep(0.25)
    white = not white
    t0 = time.perf_counter()
    paint(white)
    t1 = time.perf_counter()
    while True:                                     # first captured frame that shows the new colour
        ok, f = src.read()
        if ok and (f.mean() > 127) == white:
            lat.append((time.perf_counter() - t1) * 1e3)
            break
        if time.perf_counter() - t1 > 1.0:
            lat.append(float("nan"))
            break
good = [x for x in lat if x == x]
print(f"capture latency (paint call returned -> frame in our process): median {st.median(good):.1f} ms, p95 {sorted(good)[int(.95*len(good))]:.1f}, max {max(good):.1f}  (n={len(good)}/{len(lat)})")

n, t_end, i = 0, time.perf_counter() + 4.0, 0
last = src._seq
while time.perf_counter() < t_end:                  # repaint continuously as fast as Tk allows
    i += 1
    canvas.delete("all")
    canvas.create_rectangle(i % 600, 100, i % 600 + 40, 200, fill="white")
    root.update()
frames = src._seq - last
print(f"frames delivered while repainting for 4 s: {frames}  ({frames/4:.0f} fps), frame size {src._frame.shape[1]}x{src._frame.shape[0]}, measured fps={src.get(5):.0f}")
src.release()
root.destroy()
