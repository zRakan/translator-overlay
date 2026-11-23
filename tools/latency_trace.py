"""Per-stage latency trace of the real pipeline on a video file (played at its native speed).
  python tools/latency_trace.py -c config.plamo.toml clip.mp4 [--json out.json]

Pass 1 reads every frame offline to find the first frame each subtitle is readable. Pass 2 runs the pipeline with hooks on the change
detector, OCR and translator and reports, per subtitle: first readable frame -> detector fire -> queue -> OCR -> translation (client time,
and the server's own prompt / generation time) -> hand-off to the display."""
import argparse
import collections
import difflib
import json
import os
import re
import statistics as st
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("tools", 1)[0])
from translator_overlay import config, translate  # noqa: E402
from translator_overlay.capture import open_source  # noqa: E402
from translator_overlay.change import ChangeDetector, crop  # noqa: E402
from translator_overlay.ocr import Ocr  # noqa: E402
from translator_overlay.pipeline import run  # noqa: E402


def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def offline_reads(cfg, ocr, video):
    """Accepted text per frame ("" = no subtitle, None = shaky read), using the worker's own filters."""
    cap, out = cv2.VideoCapture(video), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        w = f.shape[1]
        ox, tol = int(cfg.region.x * w), cfg.detect.center_tol
        t = ocr.read(crop(f, cfg.region), x_range=((0.5 - tol) * w - ox, (0.5 + tol) * w - ox) if tol else None, skip_caps=cfg.detect.skip_caps)
        if len(re.findall("[A-Za-z]", t)) < 2:
            t = ""
        out.append(None if t and ocr.last_score is not None and ocr.last_score < cfg.ocr.min_score else t)


def subtitles(reads, fps=30.0, min_frames=5):
    """[(mode_text, first_readable_frame_index)] for each run of similar text. Runs of the same text separated by a gap
    shorter than half a second are one subtitle (a flickering frame must not split it)."""
    runs, cur = [], None
    for i, t in enumerate(reads):
        if t is None:
            continue
        if not t:
            cur = runs.append(cur) if cur else None
            continue
        if cur and ratio(t, cur[-1][1]) >= 0.9:
            cur.append((i, t))
        else:
            cur and runs.append(cur)
            cur = [(i, t)]
    cur and runs.append(cur)
    out = []
    for r in runs:
        if len(r) < min_frames:
            continue
        mode = collections.Counter(t for _, t in r).most_common(1)[0][0]
        first, last = next(i for i, t in r if ratio(t, mode) >= 0.9), r[-1][0]
        if out and ratio(out[-1][0], mode) >= 0.9 and first - out[-1][2] <= 0.5 * fps:
            out[-1] = (out[-1][0], out[-1][1], last)
        else:
            out.append((mode, first, last))
    return [(m, f) for m, f, _ in out]


class Cap:
    def __init__(self, cap):
        self.cap, self.n, self.ts = cap, 0, []

    def read(self):
        ok, f = self.cap.read()
        if ok:
            self.n += 1
            self.ts.append(time.perf_counter())
        return ok, f

    def get(self, p):
        return self.cap.get(p)

    def release(self):
        self.cap.release()


def trace(cfg, ocr, tr, video):
    cap = Cap(open_source(video))
    fires, ocrs, trs, shown = [], [], [], []
    local = threading.local()

    orig_update = ChangeDetector.update
    def update(self, img):
        r = orig_update(self, img)
        if r:
            fires.append((time.perf_counter(), cap.n))
        return r
    ChangeDetector.update = update

    orig_read = ocr.read
    def read(*a, **k):
        t = time.perf_counter()
        r = orig_read(*a, **k)
        ocrs.append((t, time.perf_counter(), r, ocr.last_score))
        return r
    ocr.read = read

    orig_post = translate._post_json
    def post(*a, **k):
        r = orig_post(*a, **k)
        local.server = r.get("timings") if isinstance(r, dict) else None
        return r
    translate._post_json = post

    orig_tr = tr.translate
    def translate_(src, *a, **k):
        local.server = None
        t = time.perf_counter()
        r = orig_tr(src, *a, **k)
        trs.append((t, time.perf_counter(), src, local.server))
        return r
    tr.translate = translate_

    try:
        run(cfg, cap, ocr, tr, emit=lambda *a: None, on_subtitle=lambda out, *a: out and shown.append(time.perf_counter()))
    finally:
        ChangeDetector.update = orig_update
        ocr.read = orig_read
        translate._post_json = orig_post
    return cap, fires, ocrs, trs


def rows_for(subs, cap, fires, ocrs, trs, strip=True, window=2.5):
    """One row per subtitle. A translation call belongs to the first subtitle (in order) it matches within `window` seconds of the
    moment that subtitle became readable; each call is used once. Speaker labels are stripped as the pipeline does."""
    rows, used = [], set()
    for text, fi in subs:
        t_r = cap.ts[fi]
        want = translate.strip_speaker(text) if strip else text
        tr = next((x for k, x in enumerate(trs) if k not in used and t_r - 0.001 <= x[0] <= t_r + window and ratio(x[2], want) >= 0.75), None)
        if tr is not None:
            used.add(next(k for k, x in enumerate(trs) if x is tr))
        if tr is None:
            rows.append({"text": text, "frame": fi, "missed": True})
            continue
        o = max((x for x in ocrs if x[1] <= tr[0] + 1e-4), key=lambda x: x[1])
        f = max((x for x in fires if x[0] <= o[0] + 1e-4), key=lambda x: x[0])
        srv = tr[3] or {}
        rows.append({
            "text": text, "frame": fi, "missed": False, "cache_hit": not srv,
            "detect_ms": 1e3 * (f[0] - t_r), "fires_before": sum(1 for x in fires if t_r <= x[0] < f[0]),
            "qwait_ms": 1e3 * (o[0] - f[0]), "ocr_ms": 1e3 * (o[1] - o[0]), "handoff_ms": 1e3 * (tr[0] - o[1]),
            "mt_ms": 1e3 * (tr[1] - tr[0]), "srv_prompt_ms": srv.get("prompt_ms"), "srv_gen_ms": srv.get("predicted_ms"),
            "prompt_n": srv.get("prompt_n"), "cache_n": srv.get("cache_n"), "gen_n": srv.get("predicted_n"),
            "total_ms": 1e3 * (tr[1] - t_r)})
    return rows


def summary(rows):
    ok = [r for r in rows if not r["missed"]]
    live = [r for r in ok if not r["cache_hit"]]
    print(f"subtitles found offline: {len(rows)}   translated in the live run: {len(ok)}   missed: {len(rows) - len(ok)}   "
          f"cache hits: {len(ok) - len(live)}")
    def line(label, key, rs=live):
        v = sorted(x[key] for x in rs if x.get(key) is not None)
        if v:
            print(f"  {label:<34} median {st.median(v):7.1f}   p90 {v[int(0.9 * (len(v) - 1))]:7.1f}   max {v[-1]:7.1f}")
    print("per subtitle, ms (cache hits excluded):")
    line("first readable frame -> detector fire", "detect_ms")
    line("detector fires before this one", "fires_before")
    line("queue wait", "qwait_ms")
    line("OCR", "ocr_ms")
    line("OCR done -> translate call", "handoff_ms")
    line("translation, client side", "mt_ms")
    line("  server: prompt processing", "srv_prompt_ms")
    line("  server: generation", "srv_gen_ms")
    line("  prompt tokens (new)", "prompt_n")
    line("  prompt tokens (cached)", "cache_n")
    line("  generated tokens", "gen_n")
    line("TOTAL first readable -> displayed", "total_ms")
    for r in sorted(live, key=lambda x: -x["total_ms"])[:4]:
        print(f"  slowest: {r['total_ms']:.0f} ms  detect {r['detect_ms']:.0f}  fires_before {r['fires_before']}  ocr {r['ocr_ms']:.0f}  mt {r['mt_ms']:.0f}  {r['text'][:48]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("video")
    ap.add_argument("--json")
    ap.add_argument("--reads", help="cache file for the offline pass (read if present, else written)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    ocr = Ocr(cfg.ocr.rec, cfg.ocr.engine, cfg.ocr.cache_dir)
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30.0
    t0 = time.time()
    if a.reads and os.path.exists(a.reads):
        reads = json.load(open(a.reads))
    else:
        reads = offline_reads(cfg, ocr, a.video)
        a.reads and json.dump(reads, open(a.reads, "w"))
    subs = subtitles(reads, fps)
    print(f"offline pass: {len(subs)} subtitles in {time.time() - t0:.0f} s ({fps:g} fps)", flush=True)
    tr = translate.build(cfg.translate)
    cap, fires, ocrs, trs = trace(cfg, ocr, tr, a.video)
    rows = rows_for(subs, cap, fires, ocrs, trs, cfg.translate.strip_speaker)
    summary(rows)
    print(f"detector fires {len(fires)}, OCR reads {len(ocrs)}, translation calls {len(trs)}, frames {cap.n}")
    calls = [(round(x[0] - cap.ts[0], 3), x[2]) for x in trs]
    a.json and json.dump({"rows": rows, "calls": calls}, open(a.json, "w"), indent=1)
