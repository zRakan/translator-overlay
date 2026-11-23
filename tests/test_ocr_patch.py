import threading

import numpy as np
import pytest

U = pytest.importorskip("rapidocr.ch_ppocr_det.utils")  # optional dependency (extra "ocr")
from translator_overlay.ocr import _speed_up_det_preprocess  # noqa: E402


@pytest.fixture
def det():
    """DetPreProcess with the patch flag cleared; restored afterwards."""
    D = U.DetPreProcess
    saved, flag = D.__dict__["__call__"], D.__dict__.get("_fast_preprocess")
    if flag is not None:
        del D._fast_preprocess
    yield D
    D.__call__ = saved
    if flag is None:
        if "_fast_preprocess" in D.__dict__:
            del D._fast_preprocess
    else:
        D._fast_preprocess = flag


def reference(inst, img):
    """The detector's original chain: resize -> normalize -> permute -> batch of one."""
    out = (inst.resize(img).astype("float32") * inst.scale - inst.mean) / inst.std
    return np.expand_dims(out.transpose((2, 0, 1)), 0).astype(np.float32)


def test_fast_preprocess_is_bit_identical_to_the_original_chain(det):
    inst = det()
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, s, dtype=np.uint8) for s in ((7, 13, 3), (96, 1120, 3), (288, 1113, 3))]
    imgs.append(np.repeat(np.arange(256, dtype=np.uint8).reshape(256, 1, 1), 3, axis=2).repeat(64, axis=1))  # all byte values
    want = [reference(inst, i) for i in imgs]
    assert _speed_up_det_preprocess() is True
    for img, w in zip(imgs, want):
        got = inst(img)
        assert got.dtype == np.float32 and got.flags["C_CONTIGUOUS"] and np.array_equal(got, w)


def test_buffer_is_reused_per_shape_and_alternating_shapes_stay_correct(det):
    inst = det()
    rng = np.random.default_rng(1)
    a, b = (rng.integers(0, 256, (96, 1120, 3), dtype=np.uint8) for _ in range(2))
    c = rng.integers(0, 256, (64, 640, 3), dtype=np.uint8)
    wa, wb, wc = (reference(inst, i) for i in (a, b, c))
    assert _speed_up_det_preprocess() is True
    first = inst(a)
    assert np.array_equal(first, wa)
    assert np.array_equal(inst(c), wc) and np.array_equal(first, wa)  # another shape does not touch it
    assert inst(b) is first and np.array_equal(first, wb)  # same shape reuses the buffer


def test_threads_do_not_share_buffers(det):
    inst = det()
    img = np.random.default_rng(2).integers(0, 256, (96, 1120, 3), dtype=np.uint8)
    assert _speed_up_det_preprocess() is True
    main = inst(img)
    other = []
    t = threading.Thread(target=lambda: other.append(inst(img)))
    t.start(); t.join()
    assert other[0] is not main and np.array_equal(other[0], main)


def test_patch_is_idempotent(det):
    assert _speed_up_det_preprocess() is True
    first = det.__call__
    assert _speed_up_det_preprocess() is True and det.__call__ is first


def test_patch_is_refused_when_the_library_behaves_differently(det):
    different = lambda self, img: np.zeros((1, 3, 32, 32), np.float32)  # e.g. a future version that normalizes differently
    det.__call__ = different
    assert _speed_up_det_preprocess() is False
    assert det.__call__ is different and not det.__dict__.get("_fast_preprocess")
