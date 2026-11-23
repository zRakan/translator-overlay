"""Per-stage OCR timing on subtitle crops (same crops on every platform).
  python tools/ocr_profile.py make clip.mp4 crops.npz      # cut subtitle-band crops that contain text
  python tools/ocr_profile.py run crops.npz [--engine trt] [--no-patch] [--n 150]
"""
import argparse
import collections
import statistics as st
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("tools", 1)[0])


def make(video, out, region=(0.20, 0.855, 0.58, 0.08), step=12, limit=120):
    import cv2
    from translator_overlay.ocr import Ocr

    ocr, cap, crops, i = Ocr(), cv2.VideoCapture(video), [], 0
    while len(crops) < limit:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        h, w = f.shape[:2]
        x, y, cw, ch = int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h)
        c = f[y:y + ch, x:x + cw].copy()
        if len(ocr.read(c)) > 3:
            crops.append(c)
    np.savez_compressed(out, crops=np.stack(crops))
    print(len(crops), "crops", crops[0].shape)


class Timed:
    """Proxy whose __call__ is timed (dunder methods cannot be patched per instance)."""
    def __init__(self, inner, label, acc):
        self._inner, self._label, self._acc = inner, label, acc

    def __call__(self, *a, **k):
        t = time.perf_counter()
        try:
            return self._inner(*a, **k)
        finally:
            self._acc[self._label].append((time.perf_counter() - t) * 1e3)

    def __getattr__(self, n):
        return getattr(self._inner, n)


def hook(obj, name, label, acc):
    f = getattr(obj, name)

    def w(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            acc[label].append((time.perf_counter() - t) * 1e3)
    setattr(obj, name, w)


def run(path, engine, patch, n):
    import onnxruntime as ort
    import translator_overlay.ocr as O
    if not patch:
        O._speed_up_det_preprocess = lambda: False
    ocr = O.Ocr(engine=engine)
    crops = list(np.load(path)["crops"])
    e, acc = ocr._engine, collections.defaultdict(list)
    for _ in range(10):  # settle
        ocr.read(crops[0])
    det, rec = e.text_det, e.text_rec
    from rapidocr.ch_ppocr_det.utils import DetPreProcess as D
    from rapidocr.ch_ppocr_det import main as dm
    hook(e, "load_img", "load_img", acc)
    hook(e, "preprocess_img", "resize(pre)", acc)
    hook(e, "detect_and_crop", "det_total+crop", acc)
    hook(e, "crop_text_regions", "  crop_regions", acc)
    hook(e, "recognize_txt", "rec_total", acc)
    hook(e, "build_final_output", "build_output", acc)
    det.session = Timed(det.session, "  det_session(ort)", acc)
    det.postprocess_op = Timed(det.postprocess_op, "  det_postproc", acc)
    rec.session = Timed(rec.session, "  rec_session(ort)", acc)
    rec.postprocess_op = Timed(rec.postprocess_op, "  rec_postproc", acc)
    hook(rec, "resize_norm_img", "  rec_resize_norm", acc)
    hook(D, "__call__", "  det_preprocess", acc)
    tot, rows = [], []
    for i in range(n):
        c = crops[i % len(crops)]
        t = time.perf_counter()
        ocr.read(c)
        tot.append((time.perf_counter() - t) * 1e3)
    print(f"{sys.platform} engine={ocr.engine_name} patched={ocr.fast_preprocess} ort={ort.__version__} n={n} crop={crops[0].shape}")
    print(f"{'read() total':<22} med {st.median(tot):6.2f}  mean {st.mean(tot):6.2f}  p95 {np.percentile(tot, 95):6.2f}")
    for k, v in acc.items():
        print(f"{k:<22} med {st.median(v):6.2f}  mean {st.mean(v):6.2f}  p95 {np.percentile(v, 95):6.2f}  calls/read {len(v)/n:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "run"])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--no-patch", action="store_true")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    make(a.a, a.b) if a.cmd == "make" else run(a.a, a.engine, not a.no_patch, a.n)
