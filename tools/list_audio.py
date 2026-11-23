"""List the audio devices Windows exposes (WASAPI), to pick the capture card's input and the output for [audio] in the config."""
import sys

import sounddevice as sd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
apis = sd.query_hostapis()
w = next(i for i, a in enumerate(apis) if "WASAPI" in a["name"])
print(f"default input: {sd.query_devices(apis[w]['default_input_device'])['name']!r}   default output: {sd.query_devices(apis[w]['default_output_device'])['name']!r}")
for i, d in enumerate(sd.query_devices()):
    if d["hostapi"] == w:
        kind = ("in " if d["max_input_channels"] else "") + ("out" if d["max_output_channels"] else "")
        print(f"{i:3d}  {kind:6s} {int(d['default_samplerate']):6d} Hz  {d['name']}")
