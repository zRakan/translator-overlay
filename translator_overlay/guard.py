"""Preflight and watchdog: refuse to start without RAM/VRAM headroom; exit cleanly when either runs low."""
import os
import subprocess
import sys
import threading
import time

MIN_FREE_RAM_MB = 1500  # WSL is OOM-killed below this
MIN_FREE_VRAM_MB = 400  # the desktop compositor shares the GPU
VRAM_BUDGET_MB = 2500  # VRAM taken at startup (OCR + translator combined)


def free_ram_mb() -> int:
    if sys.platform == "win32":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virt", "avail_virt", "avail_ext")]

        m = MemStatus()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail_phys // 2**20
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 1 << 30


def vram_mb() -> tuple[int, int] | None:
    """(free, total) for GPU 0 via nvidia-smi, or None without NVIDIA."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        free, total = (int(x) for x in out.stdout.split(",")[:2])
        return free, total
    except Exception:
        return None


def preflight():
    v = vram_mb()
    if v and v[0] < VRAM_BUDGET_MB:
        raise SystemExit(f"only {v[0]} MB VRAM free (need {VRAM_BUDGET_MB}); close GPU apps on Windows")
    if free_ram_mb() < MIN_FREE_RAM_MB * 2:
        raise SystemExit(f"only {free_ram_mb()} MB RAM free; need {MIN_FREE_RAM_MB * 2}")


def watchdog(log=print, interval=0.5):
    """Daemon thread: logs peaks, hard-exits the process if headroom is gone."""
    peak = {"ram_used": 0, "vram_used": 0}
    base = vram_mb()

    def loop():
        while True:
            ram, v = free_ram_mb(), vram_mb()
            if v and base:
                peak["vram_used"] = max(peak["vram_used"], base[0] - v[0])
            if ram < MIN_FREE_RAM_MB or (v and v[0] < MIN_FREE_VRAM_MB):
                log(f"[guard] low memory (RAM free {ram} MB, VRAM free {v[0] if v else '?'} MB), exiting")
                os._exit(3)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()
    return peak
