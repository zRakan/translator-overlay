import os
import threading
import time
from collections import deque

# MSMF hardware transforms add ~20 s to opening a UVC card; OpenCV reads this at open time, so set it first
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
import numpy as np


class _Paced:
    """Plays a video file at its native FPS (a live card is already real-time)."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap, self.dt, self.next = cap, 1 / (cap.get(cv2.CAP_PROP_FPS) or 30), None
        self._wake = threading.Event()

    def wake(self):
        """Ends the wait for the next frame early (any thread): read() returns (True, None) = nothing new, redraw the last frame."""
        self._wake.set()

    def read(self):
        self.next = self.next or time.perf_counter()  # clock starts at first read, not at open
        while (left := self.next - time.perf_counter()) > 0:
            if self._wake.wait(left):
                self._wake.clear()
                return True, None
        self._wake.clear()  # a frame is due now and will be drawn with the current overlay
        self.next += self.dt
        return self.cap.read()

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.cap.release()


class WindowSource:
    """Live capture of one window (Windows Graphics Capture). Captures the window's own pixels, so windows above it (the overlay) are not in the frames.
    read() waits for the next new frame and drops any backlog; a static window repeats the last frame."""

    def __init__(self, title: str = "", hwnd: int | None = None):
        from windows_capture import WindowsCapture

        from . import winutil

        if hwnd is None:
            hwnd = winutil.find_window(title)
            if hwnd is None:
                raise RuntimeError(f"no visible window with a title containing {title!r}; list them with: python tools/list_windows.py")
        self.hwnd = hwnd  # the overlay window positions itself from this window's bounds

        self._cv, self._frame, self._seq, self._seen, self.closed = threading.Condition(), None, 0, 0, False
        self._woken = False
        self.arrived = 0.0                      # perf_counter() of the newest frame's arrival
        self._times = deque(maxlen=120)
        target = {"window_hwnd": hwnd}

        def build(**kw):
            cap = WindowsCapture(cursor_capture=False, **target, **kw)

            @cap.event
            def on_frame_arrived(frame, control):
                t = time.perf_counter()
                bgr = np.ascontiguousarray(frame.frame_buffer[:, :, :3])  # BGRA -> BGR copy: the capture buffer is reused
                with self._cv:
                    self._frame, self._seq, self.arrived = bgr, self._seq + 1, t
                    self._times.append(t)
                    self._cv.notify_all()

            @cap.event
            def on_closed():
                with self._cv:
                    self.closed = True
                    self._cv.notify_all()

            return cap

        try:
            cap = build(draw_border=False)
            self._ctl = cap.start_free_threaded()
        except Exception:  # Windows 10 cannot hide the capture border; accepted
            cap = build()
            self._ctl = cap.start_free_threaded()

    def wake(self):
        """Ends a pending read() early (any thread): it returns (True, None) = nothing new, redraw the last frame. A new frame always wins."""
        with self._cv:
            self._woken = True
            self._cv.notify_all()

    def read(self):
        with self._cv:
            self._cv.wait_for(lambda: self._seq != self._seen or self._woken or self.closed, timeout=0.25)
            if self._frame is None or (self.closed and self._seq == self._seen):
                return False, None
            if self._woken and self._seq == self._seen:
                self._woken = False
                return True, None
            self._woken = False
            self.dropped += max(0, self._seq - self._seen - 1) if self._seen else 0
            self._seen = self._seq
            return True, self._frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:  # observed delivery rate; 60 until enough frames
            t = list(self._times)
            return (len(t) - 1) / (t[-1] - t[0]) if len(t) >= 10 and t[-1] > t[0] else 60.0
        return 0.0

    def release(self):
        self._ctl.stop()


def parse_cam(spec: str) -> dict:
    """'cam:0:1920x1080@60:MJPG:msmf' -> {index, width, height, fps, fourcc, backend}. Everything after the index is
    optional and order-free: WxH@fps (or WxH), a 4-letter FOURCC (MJPG, YUY2, NV12...), a backend (dshow | msmf)."""
    parts = spec.split(":")[1:]
    out = {"index": int(parts[0]), "width": 0, "height": 0, "fps": 0, "fourcc": "", "backend": "dshow"}
    for tok in parts[1:]:
        t = tok.strip()
        if "x" in t.lower() and t[0].isdigit():
            wh, _, fps = t.lower().partition("@")
            out["width"], out["height"] = (int(v) for v in wh.split("x"))
            out["fps"] = int(fps) if fps else 0
        elif t.lower() in ("dshow", "msmf"):
            out["backend"] = t.lower()
        elif len(t) == 4:
            out["fourcc"] = t.upper()
        else:
            raise ValueError(f"cannot parse {tok!r} in {spec!r}")
    return out


class CameraSource:
    """UVC capture card / webcam (DirectShow or Media Foundation). A background thread keeps only the newest frame;
    read() waits for the next new frame."""

    BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}

    def __init__(self, index=0, width=0, height=0, fps=0, fourcc="", backend="dshow", _cap=None):
        self._cap = _cap or cv2.VideoCapture(index, self.BACKENDS[backend])
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open capture device {index} ({backend}); is another app (OBS, the card's software) using it?")
        if fourcc:  # set the format first: many cards only offer 1080p60 in MJPG (or YUY2/NV12 on USB3)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width and height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cv, self._frame, self._seq, self._seen, self.closed = threading.Condition(), None, 0, 0, False
        self._woken = False
        self._times = deque(maxlen=120)
        self.arrived = 0.0
        self.dropped = 0  # frames the card delivered that read() never returned (our loop was slower than the card)
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self):
        while not self.closed:
            ok, f = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            t = time.perf_counter()
            with self._cv:
                self._frame, self._seq, self.arrived = f, self._seq + 1, t
                self._times.append(t)
                self._cv.notify_all()

    def wake(self):
        """Ends a pending read() early (any thread): it returns (True, None) = nothing new, redraw the last frame. A new frame always wins."""
        with self._cv:
            self._woken = True
            self._cv.notify_all()

    def read(self):
        with self._cv:
            self._cv.wait_for(lambda: self._seq != self._seen or self._woken or self.closed, timeout=0.5)
            if self._frame is None:
                return False, None
            if self._woken and self._seq == self._seen:
                self._woken = False
                return True, None
            self._woken = False
            self._seen = self._seq
            return True, self._frame

    def info(self) -> dict:
        c = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return {"width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "driver_fps": self._cap.get(cv2.CAP_PROP_FPS), "fourcc": "".join(chr((c >> 8 * i) & 255) for i in range(4)).strip("\x00 ")}

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:  # observed delivery rate; driver-reported until enough frames
            t = list(self._times)
            return (len(t) - 1) / (t[-1] - t[0]) if len(t) >= 10 and t[-1] > t[0] else (self._cap.get(prop) or 60.0)
        return self._cap.get(prop)

    def release(self):
        self.closed = True
        with self._cv:
            self._cv.notify_all()
        self._thread.join(1)
        self._cap.release()


def open_source(device: str):
    """Source spec: capture card index ("0"), "cam:<index>[:WxH@fps][:FOURCC][:dshow|msmf]", "window:<title>", or a file/stream URL (OpenCV)."""
    if device.startswith("window:"):
        return WindowSource(device[len("window:"):])
    if device.startswith("cam:"):
        return CameraSource(**parse_cam(device))
    cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {device!r}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low: never queue stale frames
    return cap if device.isdigit() else _Paced(cap)
