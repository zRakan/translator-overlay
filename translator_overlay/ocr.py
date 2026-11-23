import os
import re
import sys

import cv2
import numpy as np

# TensorRT shape profiles (min / opt / max): one dynamic engine per model. Det: longest side <= 1280.
# Rec: height 48, width grows with line length (4096 covers a full-width line at 4K).
_DET_PROFILE = ("x:1x3x32x32", "x:1x3x128x1152", "x:1x3x1280x1280")
_REC_PROFILE = ("x:1x3x48x32", "x:2x3x48x800", "x:6x3x48x4096")


def _speed_up_det_preprocess() -> bool:
    """Replace the detector's resize -> normalize -> permute -> astype chain, which allocates ~4 fresh float64/float32 arrays per frame
    (5 ms on Windows, where fresh pages are slow, 2 ms on Linux), with per-channel writes into a reused contiguous CHW float32 buffer (0.3 ms).
    Applied only if its output is bit-identical to the original on probe images; returns whether the fast version is active."""
    import threading

    try:
        from rapidocr.ch_ppocr_det.utils import DetPreProcess as D
    except Exception:
        return False
    if getattr(D, "_fast_preprocess", False):
        return True
    original, local = D.__call__, threading.local()

    def fast(self, img):
        r = self.resize(img)
        if r is None:
            return None
        h, w = r.shape[:2]
        bufs = local.__dict__.setdefault("bufs", {})
        buf = bufs.get((h, w))
        if buf is None:
            buf = bufs[(h, w)] = np.empty((1, 3, h, w), np.float32)
        a = (self.scale / self.std).astype(np.float32)
        b = (-self.mean / self.std).astype(np.float32)
        for c in range(3):
            np.multiply(r[..., c], a[c], out=buf[0, c], dtype=np.float32)
            buf[0, c] += b[c]
        return buf  # reused by the next call of this shape on this thread; ONNX Runtime has copied it by then

    try:
        ramp = (np.arange(256, dtype=np.uint16)[:, None, None] + np.array([0, 85, 170], np.uint16)) % 256  # every byte value in every channel
        probes = [np.repeat(ramp, 64, axis=1).astype(np.uint8), np.random.default_rng(0).integers(0, 256, (96, 1113, 3), dtype=np.uint8)]
        inst = D()
        for p in probes:
            if not np.array_equal(original(inst, p.copy()), fast(inst, p.copy())):
                return False
    except Exception:  # signature or behaviour changed in this RapidOCR version: keep its own code
        return False
    D.__call__, D._fast_preprocess = fast, True
    return True


def _preload_tensorrt() -> bool:
    """Preload the pip-installed TensorRT 10 libraries so the ONNX Runtime TensorRT provider finds them; False if absent."""
    import ctypes
    import glob
    import site

    dirs = [d for s in site.getsitepackages() for d in glob.glob(os.path.join(s, "tensorrt_libs"))]
    win = sys.platform == "win32"
    for d in dirs:
        if win:  # DLLs are found via the search path, so register the folder, then load by absolute path
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        pats = ("nvinfer_*.dll", "nvonnxparser_*.dll") if win else ("libnvinfer.so*", "libnvonnxparser.so*")
        for f in sorted(g for p in pats for g in glob.glob(os.path.join(d, p))):
            try:
                ctypes.WinDLL(f) if win else ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                return False
    return bool(dirs)


class Ocr:
    """OCR via ONNX Runtime, tuned for English subtitles.
    engine: auto (TensorRT -> CUDA -> CPU) | trt | cuda | cpu; one 1113x86 crop read takes ~5 / ~11 / ~24 ms (CPU: 8 threads), same text on Linux and Windows.
    rec: v5-en (English-only, default) | v6 (multilingual)."""

    def __init__(self, rec: str = "v5-en", engine: str = "auto", cache_dir: str = ""):
        import onnxruntime as ort
        import rapidocr.inference_engine.onnxruntime.main as ort_main
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR  # optional dependency

        if rec not in ("v6", "v5-en"):
            raise ValueError(f"unknown ocr rec model {rec!r}")
        if engine not in ("auto", "trt", "cuda", "cpu"):
            raise ValueError(f"unknown ocr engine {engine!r}")
        ort.preload_dlls()  # load pip-installed CUDA libs; otherwise ORT silently falls back to CPU
        self.fast_preprocess = _speed_up_det_preprocess()
        have = ort.get_available_providers()
        trt_ok = engine in ("auto", "trt") and "TensorrtExecutionProvider" in have and _preload_tensorrt()
        cuda_ok = "CUDAExecutionProvider" in have
        if engine == "trt" and not trt_ok:
            raise RuntimeError("ocr engine 'trt' needs onnxruntime-gpu with its TensorRT provider and `pip install tensorrt-cu13-libs`")
        if engine == "cuda" and not cuda_ok:
            raise RuntimeError("onnxruntime-gpu has no CUDA provider")
        self.engine_name = "trt" if trt_ok else "cuda" if (cuda_ok and engine != "cpu") else "cpu"

        params = {
            # default upscales the ~90 px subtitle crop ~8x before detection (5x slower, same text)
            "Det.limit_type": "max", "Det.limit_side_len": 1280,
            "Det.max_candidates": 20,  # a subtitle band never has more boxes; same text, trims post-processing
        }
        if rec == "v5-en":
            params |= {"Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.lang_type": LangRec.EN, "Rec.model_type": ModelType.MOBILE}
        cuda_opts = {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo",
                     "cudnn_conv_algo_search": "HEURISTIC",   # EXHAUSTIVE re-benchmarks per new shape: +300-500 ms per subtitle width
                     "gpu_mem_limit": 700 * 1024 * 1024}  # bounds arena growth (OOM next to the desktop's GPU share)
        if self.engine_name == "cuda":
            params |= {"EngineConfig.onnxruntime.use_cuda": True,
                       "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": cuda_opts["cudnn_conv_algo_search"],
                       "EngineConfig.onnxruntime.cuda_ep_cfg.gpu_mem_limit": cuda_opts["gpu_mem_limit"]}
        elif self.engine_name == "cpu":  # 8 threads is the sweet spot; all 16 (SMT) is 3x slower
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = min(8, max(1, (os.cpu_count() or 2) // 2))

        shim = None
        if self.engine_name == "trt":
            cache = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "translator-overlay", "trt")
            os.makedirs(cache, exist_ok=True)
            orig = ort_main.InferenceSession

            def with_tensorrt(path, sess_options=None, providers=None, **kw):
                name = os.path.basename(str(path)).lower()
                prof = _DET_PROFILE if "det" in name else _REC_PROFILE if "rec" in name else None
                if prof is None:
                    return orig(path, sess_options=sess_options, providers=providers, **kw)
                trt = ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": False,  # FP32: same text as the ONNX path; FP16 is unreliable for these models
                    "trt_engine_cache_enable": True, "trt_engine_cache_path": cache, "trt_max_workspace_size": 1 << 30,
                    "trt_profile_min_shapes": prof[0], "trt_profile_opt_shapes": prof[1], "trt_profile_max_shapes": prof[2]})
                chain = [trt] + ([("CUDAExecutionProvider", cuda_opts)] if cuda_ok else []) + ["CPUExecutionProvider"]
                return orig(path, sess_options=sess_options, providers=chain, **kw)

            # models load lazily on first call: keep the session patch through warm-up
            ort_main.InferenceSession = shim = with_tensorrt
            orig_session = orig
        try:
            self._engine = RapidOCR(params=params)
            # warm-up: first CUDA calls ~1 s; the first-ever run also builds TensorRT engines (~25 s, cached)
            dummy = np.zeros((108, 896, 3), np.uint8)
            cv2.putText(dummy, "Warm up the engine.", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            for _ in range(3):
                self.read(dummy)
        finally:
            if shim is not None:
                ort_main.InferenceSession = orig_session

    extend_up = True  # the pipeline may re-read a taller crop when the text touches the top of the region (read(..., max_gap=...))

    def read(self, img: np.ndarray, x_range=None, skip_caps=False, max_gap=None) -> str:
        """x_range=(lo, hi) in crop pixels: keep lines whose centre lies inside (subtitles are centred; world text usually is not).
        skip_caps drops ALL-CAPS lines (signs); leave off for all-caps subtitles.
        max_gap: keep only the block of lines that starts at the bottom-most one and has no gap above a line bigger than max_gap line heights
        (a tall crop also holds scenery: only the connected subtitle lines are wanted)."""
        res = self._engine(img, use_cls=False)  # subtitles are upright; skips the angle classifier
        self.last_box = self.last_line_h = self.last_score = None
        if res.txts is None:
            return ""
        keep = [i for i, b in enumerate(res.boxes)
                if (x_range is None or x_range[0] <= np.asarray(b)[:, 0].mean() <= x_range[1])
                and not (skip_caps and re.search("[A-Z]", res.txts[i]) and not re.search("[a-z]", res.txts[i]))]
        if keep and max_gap:
            hs = [np.ptp(np.asarray(res.boxes[i])[:, 1]) for i in keep]
            lh = float(np.median(hs))
            order = sorted(keep, key=lambda i: -np.asarray(res.boxes[i])[:, 1].max())  # bottom-most first
            chain, top = [order[0]], np.asarray(res.boxes[order[0]])[:, 1].min()
            for i in order[1:]:
                b = np.asarray(res.boxes[i])[:, 1]
                if top - b.max() <= max_gap * lh:
                    chain.append(i)
                    top = min(top, b.min())
                else:
                    break
            keep = chain
        if not keep:
            return ""
        pts = np.concatenate([np.asarray(res.boxes[i]) for i in keep])
        self.last_box = (*pts.min(0).astype(int), *pts.max(0).astype(int))  # x0,y0,x1,y1 in crop
        self.last_score = float(min(res.scores[i] for i in keep))  # weakest kept line: low = garbled/partial read
        # median line height = the game's text size, so the overlay can match it
        self.last_line_h = float(np.median([np.ptp(np.asarray(res.boxes[i])[:, 1]) for i in keep]))
        keep.sort(key=lambda i: res.boxes[i][0][1])  # top-to-bottom
        return " ".join(res.txts[i] for i in keep).strip()

    def lines(self, img: np.ndarray):
        """Every text line in `img` as (text, (x0, y0, x1, y1), score, chars), unfiltered, crop pixels; chars = [(char, box)].
        For screens with separate text pieces (map cards, labels) where read() would join them."""
        # The detector keeps only the first `max_candidates` contours (unordered): 20 suffices for a subtitle band but drops real lines on a full map screen.
        # Lifted for these reads (not latency critical).
        det = getattr(self._engine, "text_det", None)  # created lazily by the first call (warm-up)
        post = getattr(det, "postprocess_op", None)
        old = post.max_candidates if post is not None else None
        try:
            if post is not None:
                post.max_candidates = 1000
            res = self._engine(img, use_cls=False, return_word_box=True, return_single_char_box=True)
        finally:
            if post is not None:
                post.max_candidates = old
        if res.txts is None:
            return []
        out = []
        words = res.word_results or ()
        for i, (t, b, sc) in enumerate(zip(res.txts, res.boxes, res.scores)):
            b = np.asarray(b)
            chars = []
            for ch in (words[i] if i < len(words) else ()):  # (char, score, 4-point box)
                q = np.asarray(ch[2]).reshape(-1, 2)
                chars.append((ch[0], (int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max()))))
            out.append((t, (int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 0].max()), int(b[:, 1].max())), float(sc), chars))
        return out
