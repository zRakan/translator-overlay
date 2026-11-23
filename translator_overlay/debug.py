"""--debug: a profiling / quality log of the live pipeline, written for later analysis (tools/diagnose.py reads it).

  debug/trace-<time>.jsonl   one JSON object per line, all times in ms:
      start      the whole config (api key removed), platform, OCR engine, capture source info
      read       one per OCR read, whatever became of it: outcome = translated | same_text | low_score | no_text | ocr_error | translate_error
                 | no_translation | map_card.  A "translated" read has the source text, the Arabic, the conversation context that was given to
                 the model, the stage times (queue, OCR, translation, the server's own prompt / generation time and tokens, display) and quality
                 flags (no Arabic letters, English left in the Arabic, length ratio, model copied the source)
      capture    every few seconds: frame rate, frame gaps, stalls, frames our loop skipped, pipeline counters
      mark       the k key (marks a problem: the clean frame is saved to debug/marks/)
      summary    written at exit
  debug/crops/<time>-NNNNN.png   the crop the OCR read (what it actually saw)
  debug/LATEST               name of the newest trace file

"on screen" is the moment our window / overlay was handed the picture that contains the Arabic. The capture card's own delay, the USB
transfer and the monitor's scan-out cannot be measured in software and are not in any number."""
import json
import os
import platform
import queue
import re
import statistics
import sys
import threading
import time
from collections import deque

SAMPLE_EVERY_S = 5.0
CROP_OUTCOMES = ("translated", "low_score", "translate_error", "no_translation")


def _q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else 0.0


def quality_flags(src: str, out: str) -> list:
    """Cheap checks on a translation that point at a problem (not a verdict on the wording)."""
    flags = []
    has_ar = bool(re.search("[؀-ۿ]", out))
    if not has_ar:
        flags.append("no_arabic")
    if out.strip().lower() == src.strip().lower():
        flags.append("copied_source")
    if has_ar and re.search("[A-Za-z]{3,}", out):
        flags.append("english_left")  # words that stayed in English (names are often fine)
    n_src = max(1, len(src))
    if len(out) > 2.5 * n_src + 10:
        flags.append("much_longer_than_source")
    if len(out) < 0.3 * n_src and n_src > 12:
        flags.append("much_shorter_than_source")
    return flags


class Debug:
    def __init__(self, cfg, info: dict | None = None):
        """cfg: config.DebugCfg. info: static facts for the start line (config, ocr engine, source...)."""
        self.cfg, self.info = cfg, dict(info or {})
        os.makedirs(cfg.dir, exist_ok=True)
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(cfg.dir, f"trace-{self.stamp}.jsonl")
        with open(os.path.join(cfg.dir, "LATEST"), "w", encoding="utf-8", newline="\n") as f:
            f.write(os.path.basename(self.path) + "\n")
        self.t0 = time.perf_counter()
        self.frames = deque(maxlen=240)   # perf_counter of the last frames handed to the pipeline
        self.n_frames = self.stalls = 0
        self.all = []                      # finished "translated" records (for the summary)
        self._pending = []                 # translated, not yet on screen
        self._lock = threading.Lock()
        self.n = self.marks = self._crops = 0
        self.outcomes: dict = {}
        self.stats: dict = {}              # pipeline counters (pipeline.run fills it)
        self.cap = None                    # the capture source (optional: dropped frames)
        self._last_sample = self.t0
        self._w: queue.Queue = queue.Queue()
        self._file = open(self.path, "w", encoding="utf-8")
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        self._put({"event": "start", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.platform(), "python": sys.version.split()[0],
                   "units": "ms; t = seconds since start", **self.info})

    # -- writer thread: file output never slows the pipeline -----------------------------------------------------------------------------
    def _put(self, d, png=None):
        self._w.put((d, png))

    def _writer(self):
        import cv2

        while (item := self._w.get()) is not None:
            d, png = item
            try:
                if png is not None:
                    os.makedirs(os.path.dirname(png[0]), exist_ok=True)
                    cv2.imwrite(png[0], png[1])
                self._file.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                self._file.flush()
            except Exception:
                pass  # a full disk must not stop the pipeline

    def _rel(self, t):
        return round(t - self.t0, 4)

    @staticmethod
    def _clean(d):
        return {k: (round(v, 1) if isinstance(v, float) and k.endswith("_ms") else v) for k, v in d.items() if not k.startswith("_")}

    # -- feeds ---------------------------------------------------------------------------------------------------------------------------
    def frame(self, t: float | None = None):
        """One real captured frame was handed to the pipeline."""
        t = time.perf_counter() if t is None else t
        if len(self.frames) >= 10:
            iv = list(self.frames)
            med = statistics.median(b - a for a, b in zip(iv, iv[1:]))
            if t - iv[-1] > 2.5 * med:
                self.stalls += 1
        self.frames.append(t)
        self.n_frames += 1
        if t - self._last_sample >= SAMPLE_EVERY_S:
            self._last_sample = t
            self._put({"event": "capture", "t": self._rel(t), **(self.capture_stats() or {}), "stalls": self.stalls, "frames": self.n_frames,
                       **({"skipped_by_loop": self.cap.dropped} if hasattr(self.cap, "dropped") else {}), **self.stats, "translated": self.n})

    def trace(self, r: dict):
        """Worker thread: one OCR read and what became of it (pipeline's on_trace)."""
        r = dict(r)
        img = r.pop("img", None)
        outcome = r.pop("outcome")
        srv = r.pop("server", None) or {}
        self.n += outcome == "translated"
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        t_fire, t_o0, t_o1, t_m = r.pop("t_fire"), r.pop("t_ocr0"), r.pop("t_ocr1"), r.pop("t_mt1", None)
        d = {"event": "read", "outcome": outcome, "t": self._rel(t_fire), "frame": r.pop("frame", None), "frame_age_ms": r.pop("frame_age_ms", None),
             "queue_ms": 1e3 * (t_o0 - t_fire), "ocr_ms": 1e3 * (t_o1 - t_o0)}
        if "score" in r:
            sc = r.pop("score")
            d["ocr_score"] = None if sc is None else round(float(sc), 3)
        d.update(r)
        if outcome == "translated":
            d["n"] = self.n
            d["translate_ms"] = 1e3 * (t_m - t_o1)
            d["worker_ms"] = 1e3 * (t_m - t_fire)
            d["server_prompt_ms"], d["server_gen_ms"] = srv.get("prompt_ms"), srv.get("predicted_ms")
            d["prompt_tokens"], d["cached_tokens"], d["gen_tokens"] = srv.get("prompt_n"), srv.get("cache_n"), srv.get("predicted_n")
            d["flags"] = quality_flags(d.get("src", ""), d.get("out", ""))
        elif t_m is not None:
            d["translate_ms"] = 1e3 * (t_m - t_o1)
        if "history" in d:
            d["history"] = [list(h) for h in (d["history"] or [])]
        png = None
        if self.cfg.crops and img is not None and outcome in CROP_OUTCOMES:
            self._crops += 1
            d["crop"] = f"crops/{self.stamp}-{self._crops:05d}.png"
            png = (os.path.join(self.cfg.dir, d["crop"]), img.copy())
        if outcome == "translated":
            d["_t_mt1"], d["_png"] = t_m, png
            with self._lock:
                self._pending.append(d)
        else:
            self._put(self._clean(d), png)

    def shown(self, t: float | None = None):
        """The picture with the newest overlay state was just handed to the window / overlay: closes every pending subtitle."""
        t = time.perf_counter() if t is None else t
        with self._lock:
            done, self._pending = self._pending, []
        for d in done:
            d["handoff_ms"] = 1e3 * (t - d["_t_mt1"])             # translation ready -> on screen
            d["on_screen_ms"] = d["worker_ms"] + d["handoff_ms"]  # detector fire -> on screen
            self.all.append(self._clean(d))
            self._put(self._clean(d), d.get("_png"))

    def mark(self, frame, note: str = "", shown=None, overlay=None):
        """Hotkey k: save the clean frame (and the picture as it was on screen, with the overlay) and log which subtitle was the latest + the overlay state."""
        self.marks += 1
        path = os.path.join(self.cfg.dir, "marks", f"{self.stamp}-{self.marks:03d}.png")
        spath = path[:-4] + "-shown.png"
        if shown is not None:
            self._put({"event": "mark_shown", "n": self.marks}, (spath, shown.copy()))
        self._put({"event": "mark", "n": self.marks, "t": self._rel(time.perf_counter()), "frame_file": os.path.relpath(path, self.cfg.dir).replace("\\", "/"),
                   "shown_file": os.path.relpath(spath, self.cfg.dir).replace("\\", "/") if shown is not None else None, "overlay": overlay,
                   "last_subtitle": self.all[-1]["n"] if self.all else None, "note": note}, (path, frame.copy()) if frame is not None else None)
        return self.marks

    # -- views ---------------------------------------------------------------------------------------------------------------------------
    def capture_stats(self):
        iv = [b - a for a, b in zip(list(self.frames), list(self.frames)[1:])]
        if len(iv) < 5:
            return None
        return {"fps": round(1 / statistics.median(iv), 1), "gap_med_ms": round(1e3 * statistics.median(iv), 1),
                "gap_p95_ms": round(1e3 * _q(iv, .95), 1), "gap_max_ms": round(1e3 * max(iv), 1)}

    def summary(self) -> dict:
        live = [d for d in self.all if not d["cache_hit"]]
        out = {"event": "summary", "seconds": round(time.perf_counter() - self.t0, 1), "frames": self.n_frames, "stalls": self.stalls,
               "translated": len(self.all), "cache_hits": len(self.all) - len(live), "marks": self.marks, "outcomes": self.outcomes, **self.stats}
        c = self.capture_stats()
        if c:
            out["capture"] = c
        if hasattr(self.cap, "dropped"):
            out["skipped_by_loop"] = self.cap.dropped
        for k in ("queue_ms", "ocr_ms", "translate_ms", "server_prompt_ms", "server_gen_ms", "handoff_ms", "worker_ms", "on_screen_ms"):
            v = [d[k] for d in live if d.get(k) is not None]
            if v:
                out[k] = {"median": round(statistics.median(v), 1), "p90": round(_q(v, .9), 1), "max": round(max(v), 1)}
        return out

    def text_summary(self) -> str:
        s = self.summary()
        lines = [f"[debug] {s['translated']} subtitles ({s['cache_hits']} from the cache), {s['frames']} frames in {s['seconds']} s, stalls {s['stalls']}, marks {s['marks']}"]
        for k in ("ocr_ms", "translate_ms", "on_screen_ms"):
            if k in s:
                lines.append(f"[debug]   {k[:-3]:<12} median {s[k]['median']:6.1f}   p90 {s[k]['p90']:6.1f}   max {s[k]['max']:6.1f} ms")
        lines.append(f"[debug] log: {self.path}   (diagnose it with: python tools/diagnose.py)")
        return "\n".join(lines)

    def close(self):
        with self._lock:
            left, self._pending = self._pending, []
        for d in left:  # translated but never shown (the run ended first)
            self._put(self._clean(d) | {"on_screen_ms": None}, d.get("_png"))
        self._put(self.summary())
        self._w.put(None)
        self._thread.join(3)
        self._file.close()
