import cv2
import numpy as np


def crop(frame: np.ndarray, region) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y = int(region.x * w), int(region.y * h)
    return frame[y : y + int(region.h * h), x : x + int(region.w * w)]


def _small(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (96, 24), interpolation=cv2.INTER_AREA).astype(np.int16)


class ChangeDetector:
    """Decides when to OCR the subtitle region.
    Default: fire once per new subtitle, after the region has been still for `stable_frames`.
    fast: also fire on the first changed frame (2-3 frames sooner); fires again once the region settles.
    poll_frames > 0: while the region keeps changing, fire every `poll_frames` frames regardless."""

    def __init__(self, threshold: float, stable_frames: int, fast: bool = False, poll_frames: int = 0):
        self.threshold, self.stable_frames, self.fast, self.poll_frames = threshold, stable_frames, fast, poll_frames
        self._prev = self._fired = None
        self._still = 0
        self._since = 0      # frames since the last fire
        self._early = False  # already fired early for the current burst of change

    def _differs(self, a, b) -> bool:
        return b is None or np.abs(a - b).mean() > self.threshold

    def update(self, region_img: np.ndarray) -> bool:
        return self.update_small(_small(region_img))

    def update_small(self, cur: np.ndarray) -> bool:
        moved = self._prev is not None and np.abs(cur - self._prev).mean() > self.threshold
        self._still = 0 if moved else self._still + 1
        self._since += 1
        self._prev = cur
        settled = self._still >= self.stable_frames
        if settled:
            self._early = False
        if not self._differs(cur, self._fired):
            return False
        if settled or (self.fast and moved and not self._early):
            self._early = not settled
            fire = True
        else:
            fire = bool(self.poll_frames) and self._since >= self.poll_frames
        if fire:
            self._fired, self._since = cur, 0
        return fire
