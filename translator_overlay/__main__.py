import argparse
import math
from collections import deque
import statistics
import os
import sys
import time

import cv2
import numpy as np

from . import config, guard, translate
from .overlay import Overlay
from .capture import open_source
from .ocr import Ocr
from .pipeline import run


def open_when_ready(device: str, show: bool):
    """Open the video source; if a capture card is not ready (no signal, held by another app) retry every 2 s, showing a waiting window with --show
    (q / Esc closes it). Returns None if that window was closed."""
    wait_title = "translator-overlay (waiting for the capture card)"
    first = True
    while True:
        try:
            cap = open_source(device)
            if not first and show:
                cv2.destroyWindow(wait_title)
            return cap
        except RuntimeError as e:
            if not (device.startswith("cam:") or device.isdigit()):
                raise  # a missing file/window is a configuration error: do not wait
            if first:
                print(f"[capture] {e}  Waiting for the card (turn the console on / close other apps using it)...", flush=True)
                first = False
            if show:
                img = np.zeros((360, 960, 3), np.uint8)
                cv2.putText(img, "Waiting for the capture card...", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, "Turn the console on. q / Esc closes this.", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
                cv2.imshow(wait_title, img)
                if cv2.waitKey(2000) & 0xFF in (27, ord("q")):
                    cv2.destroyWindow(wait_title)
                    return None
            else:
                time.sleep(2)


def main():
    p = argparse.ArgumentParser(prog="translator-overlay")
    p.add_argument("-c", "--config", help="TOML config (see config.example.toml)")
    p.add_argument("--source", help="override source.device")
    p.add_argument("--out", help="write the composited video to this file (mp4)")
    p.add_argument("--mode", choices=["bar", "replace"], help="override overlay.mode")
    p.add_argument("--show", action="store_true", help="display the video with subtitles in a window of our own (needed for a capture card; starts fullscreen. q/Esc quits, f = fullscreen/windowed, r = show/edit the subtitle region (drag / WASD / Enter saves), p = performance panel, m = blip/map translation on/off, a = mute audio)")
    p.add_argument("--scale", choices=["off", "cubic", "sharp"], help="override display.scale")
    p.add_argument("--fps", type=float, help="override display.fps (window frame-rate cap)")
    p.add_argument("--windowed", action="store_true", help="with --show: a normal window sized to the capture resolution instead of fullscreen")
    p.add_argument("--overlay", action="store_true", help="Windows: show the subtitles in a transparent click-through window over the captured window (source must be window:<title>)")
    p.add_argument("--record", metavar="FILE.mkv|auto", help="record the captured video (NVENC, constant frame rate, the card's sound next to it as .mka) while showing it; 'auto' = [record] dir/rec-<time>.mkv; k = mark the time (FILE.marks.txt)")
    p.add_argument("--no-ocr", action="store_true", help="no OCR and no translation: only show (and with --record, record) the video, to run the pipeline on the recording afterwards")
    p.add_argument("--debug", action="store_true", help="write a profiling / translation-quality log for later analysis (debug/trace-<time>.jsonl + OCR crops; k marks a problem frame); nothing is logged without this flag")
    a = p.parse_args()
    cfg = config.load(a.config)

    if a.source:
        cfg.source.device = a.source
    if a.mode:
        cfg.overlay.mode = a.mode
    if a.scale:
        cfg.display.scale = a.scale
    if a.fps is not None:
        cfg.display.fps = a.fps
    guard.preflight()
    guard.watchdog()
    if a.overlay and not cfg.source.device.startswith("window:"):
        p.error("--overlay needs a window source, e.g. --source \"window:Xbox\" (list windows with tools/list_windows.py)")
    cap = open_when_ready(cfg.source.device, a.show)
    if cap is None:  # closed while waiting for the card
        return
    win = rec = None
    try:
        ov, w = Overlay(cfg.overlay), [None]
        ov.map_align, ov.map_size_scale = cfg.map.align, cfg.map.size_scale
        if a.overlay:
            from .winoverlay import OverlayWindow

            win = OverlayWindow(cap.hwnd)

        shown_title = "translator-overlay"
        map_open = [False]  # map state (see [map])
        map_sw = [cfg.map.enabled]  # m = blip/map translation on/off at runtime
        audio = None
        if cfg.audio.enabled:  # capture card sound -> speakers; a = mute
            try:
                from .audio import AudioPass

                audio = AudioPass(cfg.audio.input, cfg.audio.output, cfg.audio.delay_ms, cfg.audio.volume).start()
                print(f"[audio] {audio.in_name!r} -> {audio.out_name!r}, delay {cfg.audio.delay_ms:g} ms", flush=True)
            except Exception as e:  # a missing device or package must not stop the video
                print(f"[audio] disabled: {e}", flush=True)
        if a.no_ocr:
            cfg.map.enabled = False

            class NoOcr:  # nothing to read: every detection finds no subtitle
                engine_name, last_box, last_line_h, last_score = "off", None, None, None

                def read(self, *args, **kw):
                    return ""

                def lines(self, *args, **kw):
                    return []

            ocr = NoOcr()
        else:
            ocr = Ocr(cfg.ocr.rec, cfg.ocr.engine, cfg.ocr.cache_dir)
        if a.record:
            from .recorder import Recorder

            if a.record == "auto" or os.path.isdir(a.record):
                a.record = os.path.join(cfg.record.dir if a.record == "auto" else a.record, time.strftime("rec-%Y%m%d-%H%M%S.mkv"))
            rec = Recorder(a.record, audio_match=cfg.audio.input if cfg.audio.enabled else "", ffmpeg=cfg.record.ffmpeg)
            print(f"[record] {a.record}   (k = mark the time, q = stop and close the file)", flush=True)
        dbg = None
        if a.debug:  # only the flag turns the log on
            import dataclasses

            from .debug import Debug

            conf = dataclasses.asdict(cfg)
            conf["translate"]["api_key"] = "***" if conf["translate"].get("api_key") else ""
            dbg = Debug(cfg.debug, {"config": conf, "config_file": a.config, "ocr": {"engine": ocr.engine_name, "rec": cfg.ocr.rec, "det_patch": bool(getattr(ocr, "fast_preprocess", False))},
                                    "source": cap.info() if hasattr(cap, "info") else {"device": cfg.source.device},
                                    "fps_reported": (cap.get(5) if hasattr(cap, "get") else 0) or None, "args": {"show": a.show, "overlay": a.overlay, "windowed": a.windowed, "out": bool(a.out)}})
            dbg.cap = cap
            print(f"[debug] logging to {dbg.path}   (k = mark a problem frame)", flush=True)
        cur = [None]  # the newest clean frame (k = mark)
        last_img = [None]  # the newest picture handed to the window, overlay included (k = mark)
        hud_on = [False]  # --show: p = performance panel (latest + median of the last 30 subtitles, display ms, fps)
        lat = deque(maxlen=30)  # (qwait, ocr, mt, total) ms per translated subtitle
        disp = deque(maxlen=120)
        stamps = deque(maxlen=60)
        note = [("", 0.0)]  # transient on-screen message (text, until)
        drag = {"a": None, "b": None}  # mouse-drag corners in displayed-image pixels
        shown_wh = [1, 1]  # size of the last displayed image: mouse coordinates are in these pixels

        def on_mouse(ev, mx, my, flags, _):
            if not region_on[0]:
                return
            if ev == cv2.EVENT_LBUTTONDOWN:
                drag["a"] = drag["b"] = (mx, my)
            elif ev == cv2.EVENT_MOUSEMOVE and drag["a"] and flags & cv2.EVENT_FLAG_LBUTTON:
                drag["b"] = (mx, my)
            elif ev == cv2.EVENT_LBUTTONUP and drag["a"]:
                (x0, y0), (x1, y1), (iw_, ih_) = drag["a"], (mx, my), shown_wh
                drag["a"] = drag["b"] = None
                if abs(x1 - x0) > 0.02 * iw_ and abs(y1 - y0) > 0.01 * ih_:  # ignore plain clicks
                    r = cfg.region
                    r.x, r.y, r.w, r.h = min(x0, x1) / iw_, min(y0, y1) / ih_, abs(x1 - x0) / iw_, abs(y1 - y0) / ih_
                    config.clamp_region(r)

        region_on = [False]  # --show: r draws the watched subtitle region
        if a.show:
            cv2.namedWindow(shown_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback(shown_title, on_mouse)  # KEEPRATIO: fullscreen letterboxes instead of stretching
            if not a.windowed:
                cv2.setWindowProperty(shown_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        sized = [not a.windowed]  # windowed: size to the capture resolution at the first frame

        shown_at = [0.0]
        times = []  # --show: how long our per-frame display work takes (logged every 300 frames)

        def display(f, redraw=False):
            """Scale the clean frame to the window (bicubic + light sharpening) and draw the subtitle at screen resolution.
            redraw: the same frame again because the overlay changed (shown at once, not counted as a new frame)."""
            t0 = time.perf_counter()
            if not redraw and cfg.display.fps and t0 - shown_at[0] < 0.92 / cfg.display.fps:  # over the cap: skip the work, only service the keyboard
                key_check(cv2.waitKey(1) & 0xFF)
                return
            shown_at[0] = t0
            fh, fw = f.shape[:2]
            _, _, ww, wh = cv2.getWindowImageRect(shown_title)
            k = min(ww / fw, wh / fh) if ww > 0 and wh > 0 else 1.0
            img = f
            if cfg.display.scale != "off" and k > 1.01:  # window bigger than the capture: bicubic (+ sharpening) beats the window's own stretch
                img = cv2.resize(f, (round(fw * k), round(fh * k)), interpolation=cv2.INTER_CUBIC)
                if cfg.display.scale == "sharp":
                    img = cv2.addWeighted(img, 1 + cfg.display.sharpen, cv2.GaussianBlur(img, (0, 0), 1.0), -cfg.display.sharpen, 0)
            elif cfg.display.scale != "off" and k < 0.99:  # window smaller: area averaging (no aliasing, no sharpening needed)
                img = cv2.resize(f, (round(fw * k), round(fh * k)), interpolation=cv2.INTER_AREA)
            else:
                k = 1.0
            if img is f and ov.has_content():
                img = f.copy()  # f is shown again on a redraw: never draw into it
            img = ov.draw(img, k, cfg.map.align)
            ih, iw = img.shape[:2]
            shown_wh[:] = [iw, ih]
            if region_on[0]:
                r = cfg.region
                cv2.rectangle(img, (int(r.x * iw), int(r.y * ih)), (int((r.x + r.w) * iw), int((r.y + r.h) * ih)), (0, 255, 255), 1)
                if drag["a"] and drag["b"]:
                    cv2.rectangle(img, drag["a"], drag["b"], (255, 255, 0), 1)
                help_ = (f"region x={r.x:.3f} y={r.y:.3f} w={r.w:.3f} h={r.h:.3f}  |  drag = new box, WASD = move (shift = bigger step), "
                         f"-/= height, [/] width, Enter = save to config")
                cv2.rectangle(img, (0, ih - 26), (iw, ih), (0, 0, 0), -1)
                cv2.putText(img, help_, (8, ih - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            if note[0][1] > time.perf_counter():
                cv2.putText(img, note[0][0], (12, ih - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2, cv2.LINE_AA)
            if hud_on[0]:
                not redraw and stamps.append(t0)
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0
                med = lambda i: statistics.median(x[i] for x in lat)
                rows = ([f"state: {'MAP open - translating the blip card' if map_open[0] else 'subtitles'}  (blip translation {'ON' if map_sw[0] else 'OFF'}, m)"] if cfg.map.enabled or not map_sw[0] else []) + [f"capture {fw}x{fh} -> window {img.shape[1]}x{img.shape[0]} (x{k:.2f}, {cfg.display.scale})", f"display fps {fps:.0f}"]
                if disp:
                    rows.append(f"display path {disp[-1]:.1f} ms (median {statistics.median(disp):.1f})")
                if lat:
                    l = lat[-1]
                    rows += [f"last {'card' if map_open[0] else 'subtitle'}: ocr {l[1]:.0f} + translate {l[2]:.0f} = {l[3]:.0f} ms (queue wait {l[0]:.0f} ms)",
                             f"median of {len(lat)}: ocr {med(1):.0f}  translate {med(2):.0f}  total {med(3):.0f} ms"]
                else:
                    rows.append("no subtitle translated yet")
                if audio:
                    rows.append(f"audio: {20 * math.log10(max(audio.level, 1e-5)):.0f} dB{' (muted)' if audio.muted else ''}, dropouts {audio.xruns}")
                y0 = 24
                cv2.rectangle(img, (8, 8), (14 + 9 * max(map(len, rows)), 16 + 22 * len(rows)), (0, 0, 0), -1)
                for i, r in enumerate(rows):
                    cv2.putText(img, r, (12, y0 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 1, cv2.LINE_AA)
            cv2.imshow(shown_title, img)
            last_img[0] = img
            dbg and dbg.shown()
            if not redraw:
                disp.append((time.perf_counter() - t0) * 1e3)
                times.append((time.perf_counter() - t0) * 1e3)
            if len(times) >= 300:
                q = sorted(times)
                print(f"[display] scale={cfg.display.scale}: {statistics.median(times):.1f} ms median, {q[int(.95 * len(q))]:.1f} ms p95 per frame "
                      f"({fw}x{fh} -> {img.shape[1]}x{img.shape[0]})", flush=True)
                times.clear()
            if not sized[0]:
                sized[0] = True
                scale = 1.0
                if sys.platform == "win32":  # windowed: shrink (never enlarge) so the whole picture fits on the screen
                    import ctypes
                    sw, sh = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
                    scale = min(1.0, 0.92 * sw / fw, 0.85 * sh / fh)
                cv2.resizeWindow(shown_title, int(fw * scale), int(fh * scale))
            key_check(cv2.waitKey(1) & 0xFF)

        def key_check(key):
            if key in (27, ord("q")):
                raise KeyboardInterrupt
            if key == ord("a") and audio:
                note[0] = (f"audio {'MUTED' if audio.toggle_mute() else 'ON'}", time.perf_counter() + 2.5)
            if key == ord("m"):
                map_sw[0] = not map_sw[0]
                note[0] = (f"blip / map translation {'ON' if map_sw[0] else 'OFF'}", time.perf_counter() + 2.5)
            if key == ord("p"):
                hud_on[0] = not hud_on[0]
            if key == ord("k") and rec and not dbg:
                el = rec.mark()
                note[0] = (f"marked {int(el // 60)}:{int(el % 60):02d} (recording)", time.perf_counter() + 3)
            if key == ord("k") and dbg:
                n_ = dbg.mark(cur[0], shown=last_img[0], overlay=repr(ov.current)[:600])
                note[0] = (f"marked #{n_} (frame saved in {cfg.debug.dir}/marks)", time.perf_counter() + 3)
            if key == ord("r"):
                region_on[0] = not region_on[0]
            if region_on[0] and key != 255:
                r, st = cfg.region, 0.02 if key < 97 else 0.004  # uppercase (shift) = bigger step
                ch = chr(key).lower() if 0 < key < 128 else ""
                dx, dy, dw, dh = {"a": (-st, 0, 0, 0), "d": (st, 0, 0, 0), "w": (0, -st, 0, 0), "s": (0, st, 0, 0),
                                  "-": (0, 0, 0, -st), "=": (0, 0, 0, st), "_": (0, 0, 0, -st), "+": (0, 0, 0, st),
                                  "[": (0, 0, -st, 0), "]": (0, 0, st, 0), "{": (0, 0, -st, 0), "}": (0, 0, st, 0)}.get(ch, (0, 0, 0, 0))
                if dx or dy or dw or dh:
                    r.x, r.y, r.w, r.h = r.x + dx, r.y + dy, r.w + dw, r.h + dh
                    config.clamp_region(r)
                if key == 13:  # Enter = save
                    try:
                        config.save_region(a.config, r) if a.config else (_ for _ in ()).throw(RuntimeError("no -c config file to save to"))
                        note[0] = (f"region saved to {a.config}", time.perf_counter() + 3)
                    except Exception as e:
                        note[0] = (f"save failed: {e}", time.perf_counter() + 5)
            if key == ord("f"):
                full = cv2.getWindowProperty(shown_title, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(shown_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)

        def on_frame(f):
            cur[0] = f
            rec and rec.write(f)  # the clean picture, before anything is drawn on it
            dbg and dbg.frame()
            if win:  # captured frames never contain the overlay window: f is the clean picture
                patch = ov.patch(f)
                win.show(*patch) if patch else win.hide()
                dbg and dbg.shown()
            if a.out:
                if w[0] is None:
                    h, wd = f.shape[:2]
                    w[0] = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30, (wd, h))
                w[0].write(ov.draw(f.copy() if a.show else f, 1.0, cfg.map.align))
                dbg and not (a.show or win) and dbg.shown()
            if a.show:
                display(f)

        def redraw(f):
            """A translation arrived between two frames: show it now instead of waiting for the next one."""
            if win:
                patch = ov.patch(f)
                win.show(*patch) if patch else win.hide()
                dbg and dbg.shown()
            if a.show:
                display(f, True)

        if (a.show or win) and hasattr(cap, "wake"):
            ov.on_change = cap.wake
        run(cfg, cap, ocr, translate.Echo() if a.no_ocr else translate.build(cfg.translate), on_subtitle=ov.set, on_frame=on_frame, on_timing=lambda *t: lat.append(t),
            on_map=ov.set_map, on_state=lambda on: map_open.__setitem__(0, on), map_switch=map_sw, on_redraw=redraw,
            on_trace=dbg and dbg.trace, stats=dbg.stats if dbg else None)
        w[0] and w[0].release()
    except KeyboardInterrupt:
        pass
    finally:
        if rec:
            rec.close()
            print(f"[record] {a.record}: {rec.frames} frames, {rec.dropped} dropped by the recorder, {rec.elapsed():.0f} s, {rec.marks} marks", flush=True)
        if dbg:
            dbg.close()
            print(dbg.text_summary(), flush=True)
        win and win.close()
        cap.release()
        try:
            audio and audio.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
