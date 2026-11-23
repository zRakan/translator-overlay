"""Transparent, click-through, always-on-top subtitle window for Windows (ctypes).

The source window keeps showing its own video (no added video latency); a small per-pixel-alpha window with just the subtitle patch sits above it.
- click-through (WS_EX_TRANSPARENT), never takes focus (WS_EX_NOACTIVATE): input keeps going to the source window
- excluded from screen capture (WDA_EXCLUDEFROMCAPTURE, Windows 10 2004+): no feedback into OCR
- repositioned from the target window's on-screen bounds on every update"""
import ctypes
import ctypes.wintypes as wt
import threading

import numpy as np

from . import winutil
from .winutil import user32

gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = winutil.kernel32

WS_POPUP = 0x80000000
WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x8, 0x20, 0x80, 0x80000, 0x8000000
SW_HIDE, SW_SHOWNOACTIVATE = 0, 4
WDA_EXCLUDEFROMCAPTURE = 0x11
ULW_ALPHA, AC_SRC_OVER, AC_SRC_ALPHA, PM_REMOVE = 2, 0, 1, 1


class POINT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", wt.LONG), ("cy", wt.LONG)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte), ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD), ("biBitCount", wt.WORD),
                ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG),
                ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wt.UINT), ("lpfnWndProc", ctypes.c_void_p), ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wt.HINSTANCE), ("hIcon", wt.HANDLE), ("hCursor", wt.HANDLE), ("hbrBackground", wt.HANDLE),
                ("lpszMenuName", wt.LPCWSTR), ("lpszClassName", wt.LPCWSTR)]


_WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DefWindowProcW.restype = ctypes.c_ssize_t
user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wt.HWND, wt.HANDLE, wt.HINSTANCE, ctypes.c_void_p]
user32.CreateWindowExW.restype = wt.HWND
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.DestroyWindow.argtypes = [wt.HWND]
user32.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]
user32.GetWindowDisplayAffinity.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.SetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_ssize_t]
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.PeekMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
user32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
user32.GetDC.argtypes = [wt.HWND]
user32.GetDC.restype = wt.HDC
user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
user32.UpdateLayeredWindow.argtypes = [wt.HWND, wt.HDC, ctypes.POINTER(POINT), ctypes.POINTER(SIZE), wt.HDC, ctypes.POINTER(POINT),
                                       wt.COLORREF, ctypes.POINTER(BLENDFUNCTION), wt.DWORD]
gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
gdi32.CreateCompatibleDC.restype = wt.HDC
gdi32.CreateDIBSection.argtypes = [wt.HDC, ctypes.c_void_p, wt.UINT, ctypes.POINTER(ctypes.c_void_p), wt.HANDLE, wt.DWORD]
gdi32.CreateDIBSection.restype = wt.HBITMAP
gdi32.SelectObject.argtypes = [wt.HDC, wt.HANDLE]
gdi32.SelectObject.restype = wt.HANDLE
gdi32.DeleteObject.argtypes = [wt.HANDLE]
gdi32.DeleteDC.argtypes = [wt.HDC]
kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
kernel32.GetModuleHandleW.restype = wt.HINSTANCE
_proc_keepalive = _WNDPROC(lambda h, m, w, l: user32.DefWindowProcW(h, m, w, l))  # module-level so it is never garbage-collected
_class_lock, _class_done = threading.Lock(), False


def _register():
    global _class_done
    with _class_lock:
        if not _class_done:
            wc = WNDCLASSW()
            wc.lpfnWndProc = ctypes.cast(_proc_keepalive, ctypes.c_void_p)
            wc.hInstance = kernel32.GetModuleHandleW(None)
            wc.lpszClassName = "TranslatorOverlayWindow"
            user32.RegisterClassW(ctypes.byref(wc))
            _class_done = True


class OverlayWindow:
    """show(patch, fx, fy): patch = premultiplied BGRA uint8 array (see Overlay.patch); (fx, fy) = its position in
    the captured frame. With target_hwnd=None the position is taken as screen coordinates."""

    def __init__(self, target_hwnd: int | None = None, exclude_from_capture: bool = True):
        winutil.make_dpi_aware()
        self.target, self.excluded = target_hwnd, False
        self._want_exclude = exclude_from_capture
        self._lock, self._evt, self._ready, self._stop = threading.Lock(), threading.Event(), threading.Event(), False
        self._state = None          # None = hide, else (patch, fx, fy)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(5) or not self.hwnd:
            raise RuntimeError("could not create the overlay window")

    def show(self, patch, fx: int, fy: int):
        with self._lock:
            self._state = (patch, fx, fy)
        self._evt.set()

    def hide(self):
        with self._lock:
            self._state = None
        self._evt.set()

    def close(self):
        self._stop = True
        self._evt.set()
        self._thread.join(2)

    # everything below runs on the overlay's own thread (the thread that created a window must pump it)
    def _run(self):
        _register()
        ex = WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        self.hwnd = user32.CreateWindowExW(ex, "TranslatorOverlayWindow", "translator-overlay", WS_POPUP, 0, 0, 1, 1,
                                           None, None, kernel32.GetModuleHandleW(None), None)
        if self._want_exclude and self.hwnd:
            # Order matters: SetWindowDisplayAffinity fails (error 8) on a window that is already WS_EX_LAYERED,
            # but the setting survives adding the layered style afterwards.
            user32.SetWindowDisplayAffinity(self.hwnd, WDA_EXCLUDEFROMCAPTURE)
        user32.SetWindowLongPtrW(self.hwnd, -20, ex | WS_EX_LAYERED)  # GWL_EXSTYLE
        aff = wt.DWORD(0)
        user32.GetWindowDisplayAffinity(self.hwnd, ctypes.byref(aff))
        self.excluded = aff.value == WDA_EXCLUDEFROMCAPTURE
        self._hdc_screen = user32.GetDC(None)
        self._hdc_mem = gdi32.CreateCompatibleDC(self._hdc_screen)
        self._dib = self._bits = self._old = None
        self._size, self._visible = (0, 0), False
        self._blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        self._ready.set()
        msg = wt.MSG()
        while not self._stop:
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            if self._evt.wait(0.05):
                self._evt.clear()
                with self._lock:
                    state = self._state
                self._apply(state)
        self._free_dib()
        gdi32.DeleteDC(self._hdc_mem)
        user32.ReleaseDC(None, self._hdc_screen)
        user32.DestroyWindow(self.hwnd)

    def _free_dib(self):
        if self._dib:
            gdi32.SelectObject(self._hdc_mem, self._old)
            gdi32.DeleteObject(self._dib)
            self._dib = None

    def _apply(self, state):
        if state is None or (self.target and winutil.is_minimized(self.target)):
            if self._visible:
                user32.ShowWindow(self.hwnd, SW_HIDE)
                self._visible = False
            return
        patch, fx, fy = state
        h, w = patch.shape[:2]
        if self.target:
            left, top, _, _ = winutil.frame_bounds(self.target)
            fx, fy = left + fx, top + fy
        if (w, h) != self._size:  # one DIB per patch size, reused every frame
            self._free_dib()
            bmi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)  # negative height = top-down
            bits = ctypes.c_void_p()
            self._dib = gdi32.CreateDIBSection(self._hdc_mem, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
            self._old, self._bits, self._size = gdi32.SelectObject(self._hdc_mem, self._dib), bits, (w, h)
        ctypes.memmove(self._bits, np.ascontiguousarray(patch).ctypes.data, w * h * 4)
        user32.UpdateLayeredWindow(self.hwnd, self._hdc_screen, ctypes.byref(POINT(fx, fy)), ctypes.byref(SIZE(w, h)),
                                   self._hdc_mem, ctypes.byref(POINT(0, 0)), 0, ctypes.byref(self._blend), ULW_ALPHA)
        if not self._visible:
            user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
            self._visible = True
