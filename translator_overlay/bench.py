"""End-to-end latency: synthetic 30fps video with known subtitle onsets, real OCR + translator.
Latency = (translation emitted) - (wall time the onset frame was read). Run:
    python -m translator_overlay.bench [-c config.toml] [--limit-ms 500]"""
import argparse
import statistics
import time

import cv2
import numpy as np

from . import config, guard, translate
from .ocr import Ocr
from .pipeline import run

LINES = ["Where are you going, Captain?", "We leave at dawn.", "Run!",
         "Do not trust him, he lies to everyone.", "I never wanted any of this.",
         "The bridge is out, we must go around.", "Stay close to me.", "Did you hear that?"]
FPS, HOLD, GAP = 30, 60, 30  # frames a subtitle is shown / blank between


def synth(w=1280, h=720):
    frames, onsets = [], {}
    for text in LINES:
        for _ in range(GAP):
            frames.append(np.full((h, w, 3), 40, np.uint8))
        onsets[len(frames)] = text
        for _ in range(HOLD):
            f = np.full((h, w, 3), 40, np.uint8)
            cv2.putText(f, text, (w // 4, int(h * 0.88)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            frames.append(f)
    return frames + [np.full((h, w, 3), 40, np.uint8)] * GAP, onsets


class Cap:
    def __init__(self, frames):
        self.frames, self.i, self.read_at, self.t = frames, 0, [], time.perf_counter()

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        time.sleep(max(0, self.t + self.i / FPS - time.perf_counter()))
        self.read_at.append(time.perf_counter())
        self.i += 1
        return True, self.frames[self.i - 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config")
    p.add_argument("--limit-ms", type=float, default=500)
    a = p.parse_args()
    cfg = config.load(a.config)
    guard.preflight()
    peak = guard.watchdog()
    ocr, tr = Ocr(cfg.ocr.rec, cfg.ocr.engine, cfg.ocr.cache_dir), translate.build(cfg.translate)
    cap = Cap(synth()[0])
    onsets = synth()[1]
    got = []
    cap.t = time.perf_counter()  # models are warm; start the clock now
    run(cfg, cap, ocr, tr, emit=lambda m: got.append((time.perf_counter(), m)))
    onset_t = [cap.read_at[i] for i in onsets]
    lat = []  # each emitted line belongs to the latest onset before it
    for t_emit, _ in got:
        lat.append(1e3 * (t_emit - [o for o in onset_t if o <= t_emit][-1]))
    for (_, m), l in zip(got, lat):
        print(f"{l:5.0f}ms  {m.split('  (')[0]}")
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else float("inf")
    print(f"subtitles {len(got)}/{len(onsets)}  p50 {statistics.median(lat) if lat else 0:.0f}ms  "
          f"p95 {p95:.0f}ms  ({'fast-first: no stability wait on instant subtitles' if cfg.detect.fast else f'stability wait alone = {1e3 * cfg.detect.stable_frames / FPS:.0f}ms'})")
    print(f"peak VRAM taken {peak['vram_used']} MB, RAM free now {guard.free_ram_mb()} MB")
    if len(got) < len(onsets) or p95 > a.limit_ms:
        raise SystemExit(f"FAIL: need all subtitles and p95 <= {a.limit_ms}ms")
    print("PASS")


if __name__ == "__main__":
    main()
