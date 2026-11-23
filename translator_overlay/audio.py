"""Capture-card audio passthrough: USB audio input -> speakers, with an adjustable delay for lip-sync."""
import numpy as np


class DelayLine:
    """Delays an audio stream by a fixed number of frames (0 = pass through unchanged)."""

    def __init__(self, delay_frames: int, channels: int = 2):
        self.n = max(0, int(delay_frames))
        self._buf = np.zeros((self.n, channels), np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return block
        joined = np.concatenate([self._buf, block])
        self._buf = joined[len(block):]
        return joined[: len(block)]


def find_device(sd, name: str, want_input: bool, hostapi: int):
    """Index of the first device on `hostapi` whose name contains `name` (case-insensitive); '' = the host API's default."""
    apis = sd.query_hostapis()
    if not name:
        return apis[hostapi]["default_input_device" if want_input else "default_output_device"]
    key = "max_input_channels" if want_input else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == hostapi and d[key] > 0 and name.lower() in d["name"].lower():
            return i
    raise RuntimeError(f"no {'input' if want_input else 'output'} audio device with {name!r} in its name (run tools/list_audio.py)")


class AudioPass:
    """sounddevice duplex stream on WASAPI (shared mode, low latency, automatic sample-rate conversion)."""

    def __init__(self, input_name: str = "", output_name: str = "", delay_ms: float = 0, volume: float = 1.0, rate: int = 48000):
        import sounddevice as sd  # optional dependency

        self.sd, self.muted, self.volume = sd, False, volume
        self.level = 0.0       # RMS of the last block (0..1)
        self.xruns = 0         # blocks the device reported as under/overflowed
        api = next((i for i, a in enumerate(sd.query_hostapis()) if "WASAPI" in a["name"]), 0)
        self.in_dev, self.out_dev = find_device(sd, input_name, True, api), find_device(sd, output_name, False, api)
        self.in_name, self.out_name = sd.query_devices(self.in_dev)["name"], sd.query_devices(self.out_dev)["name"]
        self.delay = DelayLine(round(delay_ms / 1000 * rate))
        extra = sd.WasapiSettings(auto_convert=True) if api and "WASAPI" in sd.query_hostapis(api)["name"] else None
        self._stream = sd.Stream(samplerate=rate, device=(self.in_dev, self.out_dev), channels=2, dtype="float32", latency="low",
                                 extra_settings=(extra, extra) if extra else None, callback=self._callback)

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.xruns += 1
        self.level = float(np.sqrt(np.mean(indata ** 2))) if frames else 0.0
        outdata[:] = 0 if self.muted else self.delay.process(indata) * self.volume

    def start(self):
        self._stream.start()
        return self

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted
