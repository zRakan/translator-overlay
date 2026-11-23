"""Cost of the map state on the subtitle path: runs the real pipeline (OCR, translation server) over a video with the map code absent (A) and present (B),
alternating, and compares per-frame loop time and subtitle latency.
python tools/map_overhead_bench.py [video] [config] [rounds] [seconds]   (results also written to bench_result.txt)"""
import statistics as st
import sys
import time

import numpy as np

from translator_overlay import config, translate
from translator_overlay.capture import open_source
from translator_overlay.mapstate import MapDetector
from translator_overlay.ocr import Ocr
from translator_overlay.pipeline import run

video = sys.argv[1] if len(sys.argv) > 1 else "tests/cropped.mp4"
cfgfile = sys.argv[2] if len(sys.argv) > 2 else "config.plamo.toml"
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 40
ocr = Ocr("v5-en", "auto", "")


def one(map_code: bool):
    cfg = config.load(cfgfile)
    cfg.translate.cache_size = 0  # every subtitle really goes to the model
    cfg.map.enabled = map_code
    cfg.map.cache = ""
    cap = open_source(video)
    stamps, lat, subs = [], [], []
    run(cfg, cap, ocr, translate.build(cfg.translate), emit=lambda *_: None, on_subtitle=lambda *a: subs.append(a),
        on_frame=lambda f: stamps.append(time.perf_counter()), on_timing=lambda *t: lat.append(t),
        map_switch=[True] if map_code else None, max_frames=int(seconds * (cap.get(5) or 30)))  # B: the full map machinery is switched on and running
    d = np.diff(stamps) * 1e3
    return {"frame_ms": float(np.median(d)), "frame_p95": float(np.percentile(d, 95)), "frames": len(stamps),
            "subs": len(lat), "ocr": st.median(x[1] for x in lat) if lat else 0, "mt": st.median(x[2] for x in lat) if lat else 0,
            "total": st.median(x[3] for x in lat) if lat else 0, "total_p95": float(np.percentile([x[3] for x in lat], 95)) if lat else 0}


one(False)  # warm-up (models, server cache, disk)
res = {False: [], True: []}
for r in range(rounds):
    for flag in ((False, True) if r % 2 == 0 else (True, False)):
        res[flag].append(one(flag))
cols = ("frame_ms", "frame_p95", "subs", "ocr", "mt", "total", "total_p95")
out = [f"video {video}, first {seconds:g} s ({res[False][0]['frames']} frames) played in real time, {rounds} rounds each, medians of rounds (ms)",
       f"{'':26s}" + "".join(f"{c:>11s}" for c in cols)]
for flag, name in ((False, "A: map code absent"), (True, "B: map code running")):
    out.append(f"{name:26s}" + "".join(f"{st.median(x[c] for x in res[flag]):11.2f}" for c in cols))
out.append("per-round totals A: " + ", ".join(f"{x['total']:.1f}" for x in res[False]) + "   B: " + ", ".join(f"{x['total']:.1f}" for x in res[True]))
f = MapDetector(config.MapCfg())
frame = np.zeros((1080, 1920, 3), np.uint8)
t = time.perf_counter()
for _ in range(3000):
    f.update(frame)
out.append(f"map tab check: {(time.perf_counter() - t) / 3000 * 1e6:.1f} us per call (it runs on every 3rd frame)")
text = "\n".join(out)
print(text)
open("bench_result.txt", "w", encoding="utf-8").write(text + "\n")
