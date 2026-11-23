"""Windows-only helpers (ctypes, no extra dependencies)."""
import ctypes
import ctypes.wintypes as wt

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")

user32.EnumWindows.argtypes = [ctypes.c_void_p, wt.LPARAM]
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.IsIconic.argtypes = [wt.HWND]
user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
kernel32.CloseHandle.argtypes = [wt.HANDLE]
dwmapi.DwmGetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]

_ENUM = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def make_dpi_aware():
    """Switch to physical-pixel coordinates (matches capture frames). Call before creating windows."""
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()


def windows():
    """Visible top-level windows with a title: [{title, hwnd, w, h, proc}]."""
    out = []

    @_ENUM
    def each(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                r = wt.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(r))
                pid = wt.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                proc, h = "", kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
                if h:
                    b, sz = ctypes.create_unicode_buffer(260), wt.DWORD(260)
                    if kernel32.QueryFullProcessImageNameW(h, 0, b, ctypes.byref(sz)):
                        proc = b.value.rsplit("\\", 1)[-1]
                    kernel32.CloseHandle(h)
                out.append({"title": buf.value, "hwnd": int(hwnd), "w": r.right - r.left, "h": r.bottom - r.top, "proc": proc})
        return True

    user32.EnumWindows(each, 0)
    return out


def find_window(sub: str):
    """hwnd of the largest visible window whose title contains `sub` (case-insensitive), else None."""
    hits = [w for w in windows() if sub.lower() in w["title"].lower() and w["w"] > 1 and w["h"] > 1]
    return max(hits, key=lambda w: w["w"] * w["h"])["hwnd"] if hits else None


def frame_bounds(hwnd):
    """(left, top, right, bottom) of the window as drawn, in screen pixels: the rectangle Windows Graphics Capture returns, so frame (0, 0) is at (left, top)."""
    r = wt.RECT()
    if dwmapi.DwmGetWindowAttribute(wt.HWND(hwnd), 9, ctypes.byref(r), ctypes.sizeof(r)) != 0:  # DWMWA_EXTENDED_FRAME_BOUNDS
        user32.GetWindowRect(wt.HWND(hwnd), ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def is_minimized(hwnd) -> bool:
    return bool(user32.IsIconic(wt.HWND(hwnd)))
