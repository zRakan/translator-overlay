"""Record the captured frames (and optionally the card's sound) to a file with NVENC, while the window keeps showing the game: --record FILE.
The frames go to ffmpeg as raw video at a constant 60 fps, one file frame per captured frame (no repeated or dropped frames: timing by arrival would repeat
one frame and drop the next whenever the capture loop is a few ms late)."""
import os
import queue
import re
import shutil
import subprocess
import threading
import time

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class Recorder:
    def __init__(self, path: str, fps: float = 60.0, cq: int = 17, audio_match: str = "", ffmpeg: str = ""):
        self.path, self.fps, self.cq, self.audio_match, self.ffmpeg = path, fps, cq, audio_match, ffmpeg or FFMPEG
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=90)   # 1.5 s of frames
        self._proc = self._aproc = self._t = None
        self.t0 = None                                   # when the first frame arrived
        self.frames = self.dropped = self.marks = 0
        self.size = None

    def _start(self, h: int, w: int):
        cmd = [self.ffmpeg, "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{w}x{h}", "-framerate", f"{self.fps:g}", "-i", "-",
               "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(self.cq), "-b:v", "0", "-maxrate", "80M", "-bufsize", "160M", "-pix_fmt", "yuv420p",
               self.path]
        self._log = open(self.path + ".log", "w", encoding="utf-8")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=self._log, stderr=self._log)
        self.size, self.t0 = (w, h), time.time()
        self._t = threading.Thread(target=self._writer, daemon=True)
        self._t.start()
        self._start_audio()

    def _start_audio(self):
        """The card's sound next to the video (<file>.mka), if a dshow audio device matches: for listening to the recording, the pipeline does not use it."""
        if not self.audio_match:
            return
        try:
            out = subprocess.run([self.ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], capture_output=True, text=True, timeout=15).stderr
            names = re.findall(r'"([^"]+)" \(audio\)', out)
            name = next((n for n in names if self.audio_match.lower() in n.lower()), None)
            if name:
                self._aproc = subprocess.Popen([self.ffmpeg, "-hide_banner", "-y", "-f", "dshow", "-audio_buffer_size", "50", "-i", f"audio={name}", "-c:a", "aac", "-b:a", "192k",
                                                os.path.splitext(self.path)[0] + ".mka"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self._aproc = None

    def _writer(self):
        while (f := self._q.get()) is not None:
            try:
                self._proc.stdin.write(f)
            except (OSError, ValueError):
                return

    def write(self, frame):
        """Hand over one captured frame (never blocks the video: a full queue drops the frame and counts it)."""
        if self._proc is None:
            self._start(*frame.shape[:2])
        try:
            self._q.put_nowait(frame.tobytes())
            self.frames += 1
        except queue.Full:
            self.dropped += 1

    def elapsed(self) -> float:
        return time.time() - self.t0 if self.t0 else 0.0

    def mark(self, text: str = "") -> float:
        el = self.elapsed()
        self.marks += 1
        with open(self.path + ".marks.txt", "a", encoding="utf-8") as f:
            f.write(f"{el:.1f}\t{text}\n")
        return el

    def close(self):
        if self._proc is None:
            return
        self._q.put(None)
        self._t.join(10)
        try:
            self._proc.stdin.close()
            self._proc.wait(60)
        except Exception:
            self._proc.kill()
        if self._aproc:
            try:
                self._aproc.stdin.write(b"q")
                self._aproc.stdin.flush()
                self._aproc.wait(20)
            except Exception:
                self._aproc.kill()
        self._log.close()
