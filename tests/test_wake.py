import threading
import time

import numpy as np

from translator_overlay import capture, config, translate
from translator_overlay.overlay import Overlay
from translator_overlay.pipeline import run


class FakeVideo:
    """Stands in for cv2.VideoCapture: 10 fps, every read returns a frame."""

    def __init__(self):
        self.n = 0

    def get(self, prop):
        return 10.0

    def read(self):
        self.n += 1
        return True, np.full((4, 4, 3), self.n, np.uint8)

    def release(self):
        pass


def test_paced_wake_ends_the_wait_early_and_pacing_continues():
    src = capture._Paced(FakeVideo())
    assert src.read()[1][0, 0, 0] == 1  # the clock starts at the first read
    threading.Timer(0.02, src.wake).start()
    t = time.perf_counter()
    ok, frame = src.read()
    assert ok and frame is None and time.perf_counter() - t < 0.07  # woken well before the next frame (0.1 s)
    t = time.perf_counter()
    ok, frame = src.read()
    assert ok and frame[0, 0, 0] == 2 and 0.03 < time.perf_counter() - t < 0.2  # then back to the normal pace


def test_paced_wake_set_while_a_frame_is_due_is_absorbed():
    src = capture._Paced(FakeVideo())
    src.read()
    src.wake()
    time.sleep(0.12)  # the next frame is already due
    ok, frame = src.read()
    assert frame is not None and frame[0, 0, 0] == 2
    assert src._wake.is_set() is False  # the frame is drawn with the current overlay, so no extra redraw


class BlockingCard:
    """Camera stand-in whose frames arrive only when released."""

    def __init__(self):
        self.go, self.n = threading.Semaphore(0), 0

    def isOpened(self):
        return True

    def set(self, *a):
        pass

    def get(self, prop):
        return 60.0

    def read(self):
        if not self.go.acquire(timeout=0.05):
            return False, None
        self.n += 1
        return True, np.full((4, 4, 3), self.n, np.uint8)

    def release(self):
        pass


def test_camera_wake_returns_no_frame_and_a_new_frame_always_wins():
    card = BlockingCard()
    src = capture.CameraSource(_cap=card)
    try:
        card.go.release()
        ok, f1 = src.read()
        assert ok and f1[0, 0, 0] == 1
        threading.Timer(0.02, src.wake).start()
        t = time.perf_counter()
        ok, f = src.read()
        assert ok and f is None and time.perf_counter() - t < 0.3  # woken; no new frame yet
        card.go.release()
        time.sleep(0.2)
        src.wake()  # a frame is pending AND a wake is requested: the frame wins
        ok, f2 = src.read()
        assert ok and f2 is not None and f2[0, 0, 0] == 2
        assert src._woken is False
    finally:
        src.release()


def test_overlay_calls_on_change_for_subtitles_and_cards():
    ov = Overlay(config.OverlayCfg())
    calls = []
    ov.on_change = lambda: calls.append(1)
    ov.set("مرحبا")
    ov.set("مرحبا")  # same subtitle again: nothing changed on screen
    ov.set(None)
    ov.set(None)  # clearing an empty overlay: nothing changed either
    ov.set_map([])
    assert len(calls) == 3
    assert not ov.has_content()
    ov.set("نص")
    assert ov.has_content()


def test_run_redraws_the_last_frame_on_wake_without_feeding_the_detector():
    frames = [np.zeros((100, 400, 3), np.uint8)] * 5 + [np.full((100, 400, 3), 255, np.uint8)] * 8
    script = iter([(True, f) for f in frames[:7]] + [(True, None), (True, None)] + [(True, f) for f in frames[7:]])

    class Cap:
        def read(self):
            return next(script, (False, None))

    class FakeOcr:
        def read(self, im, **kw):
            return "hello" if im.mean() > 100 else ""

    shown, redrawn, out = [], [], []
    run(config.Config(), Cap(), FakeOcr(), translate.build(config.Translate()), emit=out.append,
        on_frame=shown.append, on_redraw=redrawn.append)
    assert len(shown) == len(frames) and len(redrawn) == 2  # real frames and redraws are separate
    assert redrawn[0] is shown[6] and redrawn[1] is shown[6]  # the redraw repeats the last real frame
    assert len(out) == 1 and "'hello' -> 'hello'" in out[0]  # the detector saw each real frame once: one subtitle, one translation


def test_run_ignores_a_wake_before_the_first_frame():
    script = iter([(True, None), (True, np.zeros((100, 400, 3), np.uint8))])

    class Cap:
        def read(self):
            return next(script, (False, None))

    redrawn = []
    run(config.Config(), Cap(), type("O", (), {"read": lambda s, im, **k: ""})(), translate.build(config.Translate()), emit=lambda *a: None, on_redraw=redrawn.append)
    assert redrawn == []
