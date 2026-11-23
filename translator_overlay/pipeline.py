import queue

import cv2
import numpy as np
import difflib
import re
import threading
import time
from collections import deque

from .change import ChangeDetector, crop
from .mapstate import MapDetector, PhraseBook, fix_case, card_present, erase_text, group_blocks, make_probe, panel_color
from .speakers import arabic_name, is_listed
from .translate import split_segments
from .translate import last as _tl


def erased_at(e, ox, oy):
    """erase_text() result moved from search-area to frame coordinates."""
    return None if e is None else (e[0], (e[1][0] + ox, e[1][1] + oy), e[2])  # patch, position, mask (the ink colour, e[3], is separate)


def _same_card(a, b, tol=6) -> bool:
    """Same texts at (nearly) the same place: OCR boxes jitter by a pixel or two between reads and must not redraw the card."""
    return bool(b) and len(a) == len(b) and all(x[0] == y[0] and all(abs(p - q) <= tol for p, q in zip(x[1], y[1])) for x, y in zip(a, b))


def _unfinished(src: str) -> bool:
    """True if a subtitle line stops mid-sentence (no sentence-ending mark): the next line probably continues it."""
    return src.rstrip()[-1:] not in tuple(".?!…\"”’)]")


class ConversationContext:
    """Previous subtitle lines passed to the translator as context, only while they belong to one conversation: the previous subtitle was on screen
    within `gap_s` seconds. After a longer pause the history is dropped.
    when="continuation" (default): history only for the unfinished-sentence tail the current line continues (previous line lacks a sentence-ending
    mark, or the current line starts lowercase). when="always": the last `lines` lines of the conversation."""

    def __init__(self, lines: int, gap_s: float, when: str = "continuation"):
        self._n, self.gap, self._seen, self.when = max(0, lines), gap_s, None, when
        self._h: dict = {}      # speaker -> that speaker's previous (source, translation) lines: two people talking in turn do not share a thread
        self._memo: dict = {}   # (speaker, source) -> translation, for lines already translated in this conversation
        self.last_speaker = None  # the speaker who was labelled last: a line shown without its label (cut off at the top) is theirs

    def visible(self, now: float):
        """Call on every valid subtitle read (new or repeated): it is on screen right now."""
        if self._seen is None or now - self._seen > self.gap:
            self._h.clear()
            self._memo.clear()
            self.last_speaker = None
        self._seen = now

    def history(self, current: str = "", speaker=None):
        h = list(self._h.get(speaker, ()))
        if self.when == "always" or not h:
            return h
        tail, nxt = [], current
        for src, out in reversed(h):  # walk back while each line runs on into the next one
            if _unfinished(src) or nxt[:1].islower():
                tail.append((src, out))
                nxt = src
            else:
                break
        return tail[::-1]

    @staticmethod
    def _norm(src: str) -> str:
        return " ".join(src.lower().split()).rstrip(" .,;:!?…\"'”’-")  # the game's line end varies between reads ("family." / "family,")

    def recall(self, src: str, speaker=None):
        """The translation of exactly this line, if this speaker's line was already translated in this conversation (a caption that changes around it
        must not make the line translate again, differently)."""
        return self._memo.get((speaker, self._norm(src)))

    def remember(self, src: str, out: str, speaker=None):
        if self._n:
            self._memo[(speaker, self._norm(src))] = out

    def add(self, src: str, out: str, speaker=None):
        if self._n:
            self._h.setdefault(speaker, deque(maxlen=self._n)).append((src, out))
        self.remember(src, out, speaker)


def run(cfg, cap, ocr, translator, emit=print, max_frames=None, on_subtitle=None, on_frame=None, on_timing=None, on_map=None, on_state=None, map_switch=None, on_redraw=None, on_trace=None, stats=None):
    """Capture loop (never blocks; diffs a tiny crop). OCR + translation run on a worker fed by a size-1 queue where the newest job replaces a stale pending one."""
    sink = emit
    stats = stats if stats is not None else {}
    for k in ("fires", "reads", "same_text", "low_score", "errors"):
        stats.setdefault(k, 0)

    def emit(msg):  # a failing log sink (e.g. stdout redirected to a cp1252 file that cannot encode Arabic) must not kill the worker thread
        try:
            sink(msg)
        except Exception:
            pass

    q: queue.Queue = queue.Queue(maxsize=1)
    fps = (cap.get(5) if hasattr(cap, "get") else 0) or 30  # CAP_PROP_FPS; 30 if the source does not say
    poll = max(1, round(cfg.detect.poll_ms / 1000 * fps)) if cfg.detect.poll_ms else 0
    detector = ChangeDetector(cfg.detect.threshold, cfg.detect.stable_frames, cfg.detect.fast, poll)
    last = [""]
    map_on = map_switch if map_switch is not None else [cfg.map.enabled]  # runtime switch for the map state (hotkey)
    mapdet = MapDetector(cfg.map) if (cfg.map.enabled or map_switch is not None) else None
    book = PhraseBook(cfg.map.cache) if mapdet else None
    in_map, map_state = [False], {"items": None, "empty": 0, "pending": None, "probes": None, "bad": 0, "seen": set(), "flat": None, "nblocks": 0, "rect": None}  # map_state: what is on screen, and the number of consecutive empty reads
    force_poll, last_ocr_at = [False], [0.0]  # force_poll: the worker wants another read now (confirming a new card)
    last_poll, last_small, last_card_small = [0.0], [None], [None]  # last_small: tiny grey copy of the search area at the last OCR (skips OCR on a still screen)
    ctx = ConversationContext(cfg.translate.context_lines if cfg.translate.context else 0, cfg.translate.context_gap_s, cfg.translate.context_when)

    def trace(outcome, **kw):
        """--debug only: one record per OCR read, whatever became of it (translated, same text, low score, empty, error...)."""
        if on_trace:
            try:
                on_trace({"outcome": outcome, **kw})
            except Exception as e:
                emit(f"[trace callback error: {e}]")

    def read_map(img, t0, ox, oy, fw, fh):
        """Find the blip card's blocks, translate each (saved translations first), hand them to the overlay."""
        tw = time.perf_counter()
        keep, dropped = [], []
        all_lines = [(ln[0], ln[1], ln[2], ln[3] if len(ln) > 3 else None) for ln in ocr.lines(img)]
        for text, b, score, chars in all_lines:
            cx, cy = (ox + (b[0] + b[2]) / 2) / fw, (oy + (b[1] + b[3]) / 2) / fh
            if score < cfg.map.min_score or any(x <= cx <= x + w_ and y <= cy <= y + h_ for x, y, w_, h_ in cfg.map.exclude):
                if len(re.findall("[A-Za-z]", text)) >= 2:
                    dropped.append(f"{text[:24]!r}:{score:.2f}")
            else:
                keep.append((text, b, chars))
        blocks = group_blocks(keep)
        t1 = time.perf_counter()
        if not blocks:  # two empty reads in a row = the card is gone (one miss must not flicker it away)
            map_state["empty"] += 1
            map_state["pending"] = None
            if map_state["empty"] >= 2 and map_state["items"]:
                map_state.update(items=None, flat=None, nblocks=0, rect=None)
                on_map and on_map([])
            return
        map_state["empty"] = 0
        flat = " ".join(" ".join(b.text.lower().split()) for b in blocks)
        if map_state["flat"] is not None and map_state["items"] and len(blocks) >= map_state["nblocks"] \
                and difflib.SequenceMatcher(None, flat, map_state["flat"]).ratio() >= 0.97:
            map_state["pending"] = None  # same card read again (possibly split differently): leave what is shown
            return
        sig = tuple(" ".join(b.text.lower().split()) for b in blocks)
        if sig != map_state.get("shown_sig") and sig != map_state["pending"] and sig not in map_state["seen"]:
            # new card: confirm with a second read, so a half-faded read never reaches the screen
            map_state["pending"] = sig
            force_poll[0] = True
            return
        map_state["pending"] = None
        if cfg.map.debug and sig != map_state.get("shown_sig"):
            import os

            import cv2 as _cv

            os.makedirs("map_debug", exist_ok=True)
            n_ = len(os.listdir("map_debug")) // 2
            _cv.imwrite(f"map_debug/{n_:03d}.png", img)
            with open(f"map_debug/{n_:03d}.txt", "w", encoding="utf-8") as f_:
                f_.write("offset in frame: %d,%d  frame %dx%d\n" % (ox, oy, fw, fh))
                for text, b, score, _c in all_lines:
                    f_.write(f"{score:.2f}  {b}  {text!r}  chars: {[(c_, bb[0], bb[2]) for c_, bb in (_c or [])][:3]}..{[(c_, bb[0], bb[2]) for c_, bb in (_c or [])][-2:]}\n")
                f_.write("blocks: " + repr([(b.text, b.box) for b in blocks]) + "\n")
        map_state["shown_sig"] = sig
        map_state["seen"].add(sig)  # a card read twice before is shown at once next time
        items, probes = [], []
        for blk in blocks:
            e = erase_text(img, blk.lines, blk.box)  # the English removed from a copy of the card (once per card)
            er, pr = erased_at(e, ox, oy), make_probe(img, e)
            if pr is not None:
                probes.append((pr[0] + ox, pr[1] + oy, pr[2], pr[3]))
            src = fix_case(blk.text)
            ar = book.get(src)
            if ar is None:
                ar = translator.translate(src)
                if ar and re.search("[\u0600-\u06FF]", ar):  # only real Arabic is saved
                    book.put(src, ar)
            if ar:
                bx = blk.box
                items.append((ar, (bx[0] + ox, bx[1] + oy, bx[2] + ox, bx[3] + oy), blk.line_h, panel_color(img, bx), er, blk.anchor, e[3] if e is not None else None))
        if not _same_card(items, map_state["items"]):
            map_state["items"], map_state["probes"], map_state["bad"] = items, probes, 0
            map_state["flat"], map_state["nblocks"] = flat, len(blocks)
            xs = [it[1] for it in items]
            map_state["rect"] = (min(b[0] for b in xs) - 20, min(b[1] for b in xs) - 20, max(b[2] for b in xs) + 20, max(b[3] for b in xs) + 20)
            last_card_small[0] = None
            on_map and on_map(items)
            t2 = time.perf_counter()
            on_timing and on_timing(1e3 * (tw - t0), 1e3 * (t1 - tw), 1e3 * (t2 - t1), 1e3 * (t2 - t0))
            trace("map_card", t_fire=t0, t_ocr0=tw, t_ocr1=t1, t_mt1=t2, blocks=[b.text for b in blocks], out=[it[0] for it in items], dropped=dropped, img=img)
            emit(f"[map] {len(items)} blocks: {[b.text[:28] for b in blocks]}  (ocr {1e3*(t1-tw):.0f}ms, translate {1e3*(t2-t1):.0f}ms)"
                 + (f"  dropped: {dropped}" if dropped else ""))

    tall = [False]  # the caption on screen is taller than the region: read the taller picture right away

    def worker_loop():
        while (item := q.get()) is not None:
            kind, img, t0, (ox, oy), (fw, fh), *rest = item
            meta = rest[0] if rest else {}
            strip = rest[1] if len(rest) > 1 else None  # the picture above the region (copied at detection time), for subtitles taller than the region
            if kind == "map":
                if in_map[0]:
                    try:
                        read_map(img, t0, ox, oy, fw, fh)
                    except Exception as e:
                        emit(f"[map error: {e}]")
                continue
            if in_map[0]:  # a subtitle job queued just before the map opened
                continue
            tw = time.perf_counter()  # tw - t0 = time the job sat in the queue
            tol = cfg.detect.center_tol
            xr = ((0.5 - tol) * fw - ox, (0.5 + tol) * fw - ox) if tol else None
            try:
                if strip is not None and tall[0]:  # the caption on screen needed the taller read: go straight to it (the normal read would only be thrown away)
                    big = np.vstack([strip, img])
                    text = ocr.read(big, x_range=xr, skip_caps=cfg.detect.skip_caps, max_gap=1.8)
                    if text:
                        img, oy = big, oy - strip.shape[0]
                        stats["tall_direct"] = stats.get("tall_direct", 0) + 1
                    else:
                        tall[0] = False  # the caption is gone
                        text = ocr.read(img, x_range=xr, skip_caps=cfg.detect.skip_caps)
                else:
                    text = ocr.read(img, x_range=xr, skip_caps=cfg.detect.skip_caps)
            except Exception as e:  # a dead worker would silently stop all translation
                stats["errors"] += 1
                emit(f"[ocr error: {e}]")
                trace("ocr_error", t_fire=t0, t_ocr0=tw, t_ocr1=time.perf_counter(), error=str(e), **meta)
                continue
            if strip is not None and text and not tall[0] and ocr.last_box and ocr.last_box[1] <= 3:
                # the text touches the top of the region: the subtitle goes on above it (several speakers at once). Read again with the strip above added.
                big = np.vstack([strip, img])
                try:
                    text2 = ocr.read(big, x_range=xr, skip_caps=cfg.detect.skip_caps, max_gap=1.8)
                except Exception as e:
                    text2 = ""
                    emit(f"[ocr error (taller read): {e}]")
                if text2:
                    text, img, oy = text2, big, oy - strip.shape[0]
                    stats["extended"] = stats.get("extended", 0) + 1
                    tall[0] = True
                else:
                    text = ocr.read(img, x_range=xr, skip_caps=cfg.detect.skip_caps)  # keep the normal read (and the OCR box that belongs to it)
            t_o = time.perf_counter()
            stats["reads"] += 1
            raw = text
            # Latin-only check (source is English): drops OCR junk; 2 letters keeps "No." / "Hi."
            if len(re.findall("[A-Za-z]", text)) < 2:
                text = ""
            score = getattr(ocr, "last_score", None)
            if text and score is not None and score < cfg.ocr.min_score:
                stats["low_score"] += 1
                trace("low_score", t_fire=t0, t_ocr0=tw, t_ocr1=t_o, text=text, score=score, img=img, **meta)
                continue  # a shaky read is not "no subtitle": keep what is on screen
            t1 = time.perf_counter()
            if not text:  # subtitle gone: clear the overlay
                last[0] = ""
                on_subtitle and on_subtitle(None, None, None)
                trace("no_text", t_fire=t0, t_ocr0=tw, t_ocr1=t_o, text=raw, **meta)
                continue
            ctx.visible(time.perf_counter())
            # OCR noise reads the same subtitle slightly differently frame to frame; do not retranslate
            if last[0] and difflib.SequenceMatcher(None, text, last[0]).ratio() >= 0.9:
                stats["same_text"] += 1
                trace("same_text", t_fire=t0, t_ocr0=tw, t_ocr1=t_o, text=text, previous=last[0], score=score, **meta)
                continue
            last[0] = text
            retry = False
            try:
                known = lambda n: arabic_name(n) is not None
                segs = split_segments(text, known, is_listed) if cfg.translate.strip_speaker else [(None, text)]  # one per speaker; the model never sees a label
                done, past, reused, prev_speaker = [], None, 0, ctx.last_speaker
                _tl.timings = None
                for name, body in segs:
                    key = (name or "").lower() or prev_speaker  # a line without its label (cut off at the top) belongs to whoever spoke before
                    o = ctx.recall(body, key)
                    if o is not None:
                        reused += 1  # already translated in this conversation: same Arabic as before, no model call
                    else:
                        h = ctx.history(body, key)
                        past = h if past is None else past
                        o = (translator.translate(body, history=h) if h else translator.translate(body)) if body else ""
                        if h and o.strip().lower() == body.strip().lower():
                            # the model copied the line; copied lines in the history make it copy again, so translate this one alone
                            o = translator.translate(body)
                            retry = True
                        if o and o.strip().lower() != body.strip().lower():  # never feed an untranslated copy back as an example
                            ctx.add(body, o, key)
                    if name:
                        ctx.last_speaker = name.lower()
                    lab = name if (name and cfg.overlay.speaker) else None
                    if lab and cfg.overlay.speaker_arabic:
                        lab = arabic_name(lab) or lab  # a known name in Arabic: Latin letters would break the right-to-left line
                    if o:
                        done.append((lab, o, body))
            except Exception as e:  # network/model failure must not kill the loop or reach the screen
                last[0] = ""  # retry on the next detection
                stats["errors"] += 1
                emit(f"[translate error: {e}]")
                trace("translate_error", t_fire=t0, t_ocr0=tw, t_ocr1=t_o, text=text, error=str(e), score=score, img=img, **meta)
                continue
            t2 = time.perf_counter()
            src = " ".join(b for _, b in segs)
            out = " ".join(o for _, o, _ in done)
            speaker = done[0][0] if len(done) == 1 else None
            if not out:  # nothing usable: clear the old overlay instead of showing junk
                on_subtitle and on_subtitle(None, None, None)
                emit(f"{text!r} -> (no usable translation)")
                trace("no_translation", t_fire=t0, t_ocr0=tw, t_ocr1=t_o, t_mt1=t2, text=text, src=src, score=score, img=img, history=past, **meta)
                continue
            trace("translated", text=text, src=src, out=out, speaker=speaker, segments=[[l, o] for l, o, _ in done] if len(done) > 1 else None, reused_lines=reused,
                  box=[int(v) for v in ocr.last_box] if getattr(ocr, "last_box", None) is not None else None, line_h=getattr(ocr, "last_line_h", None), crop_origin=[ox, oy], t_fire=t0, t_ocr0=tw, t_ocr1=t1, t_mt1=t2, score=score, img=img, history=past, retried_alone=retry,
                  server=getattr(_tl, "timings", None), cache_hit=bool(getattr(_tl, "hit", False)), **meta)
            if on_subtitle:  # (the trace is registered first so that a display between the two calls cannot miss it)
                b = getattr(ocr, "last_box", None)
                try:
                    on_subtitle(out if len(done) == 1 else [(l, o) for l, o, _ in done], b and (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy),
                                getattr(ocr, "last_line_h", None), *([speaker] if speaker else []))
                except Exception as e:
                    emit(f"[overlay error: {e}]")
            try:  # a failing stats/display callback must not take the worker down
                on_timing and on_timing(1e3 * (tw - t0), 1e3 * (t1 - tw), 1e3 * (t2 - t1), 1e3 * (t2 - t0))
            except Exception as e:
                emit(f"[timing callback error: {e}]")
            emit(f"{text!r} -> {out!r}  (qwait {1e3*(tw-t0):.0f}ms, ocr {1e3*(t1-tw):.0f}ms, mt {1e3*(t2-t1):.0f}ms, "
                 f"total {1e3*(t2-t0):.0f}ms)")

    def worker():
        """An unexpected error while handling one job must cost that job only, not every later subtitle (a dead worker fails silently and the queue fills up)."""
        while True:
            try:
                return worker_loop()
            except Exception as e:
                stats["worker_restarts"] = stats.get("worker_restarts", 0) + 1
                emit(f"[worker error, job dropped, continuing: {type(e).__name__}: {e}]")
                try:
                    trace("worker_error", t_fire=time.perf_counter(), t_ocr0=time.perf_counter(), t_ocr1=time.perf_counter(), error=f"{type(e).__name__}: {e}")
                except Exception:
                    pass

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    n, last_frame = 0, None
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:  # the overlay changed while waiting for the next frame: show it now, without feeding the detector
            if last_frame is not None and on_redraw:
                on_redraw(last_frame)
            continue
        last_frame = frame
        n += 1
        if n == 60 and cfg.detect.poll_ms and hasattr(cap, "get"):  # live sources only know their FPS after about a second
            detector.poll_frames = max(1, round(cfg.detect.poll_ms / 1000 * (cap.get(5) or fps)))
        fh, fw = frame.shape[:2]
        if mapdet:
            was = in_map[0]
            if map_on[0]:
                in_map[0] = mapdet.update(frame)
            else:  # switched off: behave as if the map were closed (subtitles resume, card overlay removed)
                mapdet.reset()
                in_map[0] = False
            if in_map[0] != was:
                emit(f"[state] {'MAP opened: subtitle translation paused, translating the blip card' if in_map[0] else 'MAP closed: subtitle translation resumed'}")
                on_state and on_state(in_map[0])
                last[0] = ""
                map_state.update(items=None, empty=0, pending=None, shown_sig=None, flat=None, nblocks=0, rect=None)
                last_small[0] = None
                if in_map[0]:
                    on_subtitle and on_subtitle(None, None, None)
                else:
                    on_map and on_map([])
                    detector = ChangeDetector(cfg.detect.threshold, cfg.detect.stable_frames, cfg.detect.fast, poll)
        if in_map[0] and map_state["items"] and map_state["probes"]:
            # the English card must still be on screen; when it goes, remove the Arabic at once
            map_state["bad"] = 0 if card_present(frame, map_state["probes"]) else map_state["bad"] + 1
            if map_state["bad"] >= 2:
                map_state.update(items=None, probes=None, bad=0, shown_sig=None, pending=None, flat=None, nblocks=0, rect=None)
                last_small[0], force_poll[0] = None, True
                on_map and on_map([])
        if in_map[0]:
            now = time.perf_counter()
            if now - last_poll[0] >= cfg.map.poll_ms / 1000:
                last_poll[0] = now
                ax, ay, aw, ah = cfg.map.area
                ix, iy = int(ax * fw), int(ay * fh)
                area_img = frame[iy : iy + int(ah * fh), ix : ix + int(aw * fw)]
                small = cv2.resize(cv2.cvtColor(area_img, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.int16)
                changed = last_small[0] is None or (np.abs(small - last_small[0]) > 10).mean() > 0.002  # >0.2% of the area changed noticeably
                if map_state["rect"] is not None and last_small[0] is not None:
                    # a card is on screen: only its own pixels count (animated map markers must not trigger reads)
                    rx0, ry0, rx1, ry1 = (max(0, map_state["rect"][0]), max(0, map_state["rect"][1]), min(fw, map_state["rect"][2]), min(fh, map_state["rect"][3]))
                    if rx1 > rx0 and ry1 > ry0:
                        cs = cv2.resize(cv2.cvtColor(frame[ry0:ry1, rx0:rx1], cv2.COLOR_BGR2GRAY), (64, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
                        changed = last_card_small[0] is None or (np.abs(cs - last_card_small[0]) > 10).mean() > 0.003
                        last_card_small[0] = cs if changed else last_card_small[0]
                if not changed and not force_poll[0] and now - last_ocr_at[0] < 1.5:
                    if on_frame:
                        on_frame(frame)
                    continue  # nothing moved since the last read: skip OCR
                last_small[0], last_ocr_at[0], force_poll[0] = small, now, False
                item = ("map", area_img.copy(), now, (ix, iy), (fw, fh))
                try:
                    q.put_nowait(item)
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put_nowait(item)
            if on_frame:
                on_frame(frame)
            continue
        img = crop(frame, cfg.region)
        off = (int(cfg.region.x * fw), int(cfg.region.y * fh))
        if detector.update(img):
            now = time.perf_counter()
            stats["fires"] += 1
            sh = min(off[1], 2 * img.shape[0]) if getattr(ocr, "extend_up", False) else 0  # the strip above the region (its copy, the frame gets drawn on afterwards)
            strip = frame[off[1] - sh : off[1], off[0] : off[0] + img.shape[1]].copy() if sh > 0 else None
            item = ("sub", img.copy(), now, off, (fw, fh), {"frame": n, "frame_age_ms": 1e3 * (now - cap.arrived) if getattr(cap, "arrived", 0) else None}, strip)
            try:
                q.put_nowait(item)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait(item)
        if on_frame:  # draw last: OCR/detector see only the clean frame
            on_frame(frame)
    q.put(None)
    th.join()
