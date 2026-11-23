"""Open the capture-card audio passthrough for a few seconds and report the input level and dropouts.
python tools/audio_probe.py "<input device name part>" [seconds] [output name]   (names: python tools/list_audio.py)"""
import math
import sys
import time

from translator_overlay.audio import AudioPass

if len(sys.argv) < 2:
    sys.exit(__doc__)
name = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 6
out = sys.argv[3] if len(sys.argv) > 3 else ""
a = AudioPass(name, out).start()
print(f"{a.in_name!r} -> {a.out_name!r}", flush=True)
peak = -120.0
t0 = time.time()
while time.time() - t0 < secs:
    time.sleep(0.5)
    db = 20 * math.log10(max(a.level, 1e-6))
    peak = max(peak, db)
    print(f"  input level {db:6.1f} dB   dropouts {a.xruns}", flush=True)
a.stop()
print(f"peak {peak:.1f} dB (about -90 dB or lower = silence), dropouts {a.xruns}")
