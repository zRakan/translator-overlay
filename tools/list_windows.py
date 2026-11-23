"""Windows only: list visible top-level windows (title, size, process) to pick a capture target, e.g. a remote-play window.  python tools/list_windows.py [filter]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from translator_overlay import winutil  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # window titles can hold any character; the default Windows console codec chokes
winutil.make_dpi_aware()
flt = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
for w in winutil.windows():
    if w["w"] > 1 and w["h"] > 1 and (flt in w["title"].lower() or flt in w["proc"].lower()):
        print(f"{w['w']:5d}x{w['h']:<5d} {w['proc']:24s} hwnd={w['hwnd']:<10d} {w['title']}")
