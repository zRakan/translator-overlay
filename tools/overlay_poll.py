"""Windows only: log when the subtitle overlay window becomes visible/hidden (polls the window list).
python tools/overlay_poll.py [seconds]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from translator_overlay import winutil  # noqa: E402

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20
t0, last, events = time.perf_counter(), None, []
while time.perf_counter() - t0 < secs:
    hit = [w for w in winutil.windows() if w["title"] == "translator-overlay"]
    state = (hit[0]["w"], hit[0]["h"]) if hit else None
    if (state is None) != (last is None):
        events.append((time.perf_counter() - t0, state))
    last = state
    time.sleep(0.05)
shows = [e for e in events if e[1]]
print(f"overlay appeared {len(shows)} times in {secs:.0f} s:")
for t, st in events:
    print(f"   {t:5.1f}s  {'SHOWN ' + str(st[0]) + 'x' + str(st[1]) if st else 'hidden'}")
