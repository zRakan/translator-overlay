"""Windows only: find which video mode a capture card really delivers, and how fast.
python tools/camera_probe.py <index> [seconds_per_mode]
Each mode runs in its own subprocess with a timeout, because some drivers hang when a device is re-opened repeatedly.
Reports the delivered frame rate, size and whether the picture is real (not black), then prints the best --source."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
MODES = [("msmf", "MJPG", 1920, 1080, 60), ("msmf", "MJPG", 1920, 1080, 30), ("msmf", "MJPG", 1280, 720, 60), ("msmf", "YUY2", 1280, 720, 60),
         ("dshow", "MJPG", 1920, 1080, 60), ("dshow", "YUY2", 1280, 720, 60)]


def one(idx, backend, fourcc, w, h, fps, secs):
    sys.path.insert(0, str(HERE.parent.parent))
    from translator_overlay.capture import CameraSource  # first: it sets an OpenCV env var that must precede `import cv2`

    import cv2

    cam = CameraSource(idx, w, h, fps, fourcc, backend)
    t0, last, n0 = time.perf_counter(), None, None
    while time.perf_counter() - t0 < secs:
        ok, f = cam.read()
        if ok:
            last = f
            if n0 is None and time.perf_counter() - t0 > 0.5:  # ignore the first half second (driver start-up)
                n0, t_start = cam._seq, time.perf_counter()
    n1, t_end = cam._seq, time.perf_counter()
    info = cam.info()
    g = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY) if last is not None else None
    res = {"size": f"{last.shape[1]}x{last.shape[0]}" if last is not None else "-", "fps": round((n1 - n0) / (t_end - t_start), 1) if n0 else 0,
           "fourcc": info["fourcc"], "real": bool(g is not None and g.max() >= 12 and g.std() >= 2)}
    if last is not None and res["real"]:
        cv2.imwrite(f"cam_{idx}.png", last)
    print(json.dumps(res), flush=True)
    import os
    os._exit(0)  # skip release(): re-opening/closing is exactly what some drivers choke on


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        a = sys.argv[2:]
        one(int(a[0]), a[1], a[2], int(a[3]), int(a[4]), int(a[5]), float(a[6]))
    idx, secs = int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"{'mode asked':32s} {'size':10s} {'delivered fps':>13s}  picture")
    best = None
    for backend, fourcc, w, h, fps in MODES:
        label = f"{backend} {fourcc} {w}x{h}@{fps}"
        try:
            r = subprocess.run([sys.executable, str(HERE), "--one", str(idx), backend, fourcc, str(w), str(h), str(fps), str(secs)],
                               capture_output=True, text=True, timeout=secs + 12)
            res = json.loads(r.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            print(f"{label:32s} {'-':10s} {'-':>13s}  HUNG (killed after {secs + 12:.0f} s)")
            continue
        except Exception as e:
            print(f"{label:32s} failed: {(r.stderr or str(e)).strip().splitlines()[-1][:80]}")
            continue
        print(f"{label:32s} {res['size']:10s} {res['fps']:13.1f}  {'real picture' if res['real'] else 'black/blank (no signal?)'}")
        w_, h_ = (int(x) for x in res["size"].split("x")) if res["size"] != "-" else (0, 0)
        score = (w_ * h_ >= 1920 * 1080 and w_ * h_ > 0, res["fps"])
        if res["real"] and (best is None or score > best[0]):
            best = (score, f"cam:{idx}:{w}x{h}@{fps}:{fourcc}:{backend}")
    print(f"\nbest: --source \"{best[1]}\"" if best else "\nno mode gave a real picture: check HDMI/console output and that no other app has the card open")
