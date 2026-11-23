import json
import sys
import time

import numpy as np
import pytest

from translator_overlay import config, translate
from translator_overlay.debug import Debug, quality_flags
from translator_overlay.pipeline import run

sys.path.insert(0, __file__.rsplit("tests", 1)[0] + "tools")
import diagnose  # noqa: E402


def rec(t=100.0, **kw):
    return {"outcome": "translated", "text": "Jason: hello", "src": "hello", "out": "مرحبا", "t_fire": t, "t_ocr0": t + 0.001, "t_ocr1": t + 0.008, "t_mt1": t + 0.060,
            "score": 0.97, "img": np.zeros((20, 50, 3), np.uint8), "history": [("a", "ب")], "retried_alone": False,
            "server": {"prompt_ms": 7.0, "predicted_ms": 41.0, "prompt_n": 20, "cache_n": 30, "predicted_n": 21}, "cache_hit": False, "frame": 12, "frame_age_ms": 0.4, **kw}


def lines(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")]


def test_a_translation_is_closed_when_it_reaches_the_screen_and_logged(tmp_path):
    d = Debug(config.DebugCfg(dir=str(tmp_path)), {"ocr": {"engine": "trt"}, "config": {"translate": {"api_key": "***"}}})
    d.trace(rec())
    assert not d.all  # translated, not yet on screen
    d.shown(100.064)
    d.close()
    got = lines(d.path)
    assert got[0]["event"] == "start" and got[0]["ocr"]["engine"] == "trt"
    r = [x for x in got if x.get("event") == "read"][0]
    assert r["outcome"] == "translated" and r["src"] == "hello" and r["text"] == "Jason: hello" and r["history"] == [["a", "ب"]]
    assert r["queue_ms"] == 1.0 and r["ocr_ms"] == 7.0 and r["translate_ms"] == 52.0 and r["handoff_ms"] == 4.0 and r["on_screen_ms"] == 64.0
    assert r["server_gen_ms"] == 41.0 and r["gen_tokens"] == 21 and r["flags"] == [] and r["ocr_score"] == 0.97
    assert (tmp_path / r["crop"]).exists() and (tmp_path / "LATEST").read_text().strip() == d.path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    assert got[-1]["event"] == "summary" and got[-1]["on_screen_ms"]["median"] == 64.0


def test_every_outcome_is_logged_and_only_useful_ones_get_a_crop(tmp_path):
    d = Debug(config.DebugCfg(dir=str(tmp_path)))
    base = {"t_fire": 1.0, "t_ocr0": 1.001, "t_ocr1": 1.010, "img": np.zeros((5, 5, 3), np.uint8)}
    d.trace({"outcome": "same_text", "text": "hi", "previous": "hi.", **base})
    d.trace({"outcome": "no_text", "text": "", **base})
    d.trace({"outcome": "low_score", "text": "h1", "score": 0.5, **base})
    d.trace({"outcome": "ocr_error", "error": "boom", **base})
    d.trace({"outcome": "translate_error", "text": "hi", "error": "refused", "score": 1.0, **base})
    d.close()
    got = {x["outcome"]: x for x in lines(d.path) if x.get("event") == "read"}
    assert set(got) == {"same_text", "no_text", "low_score", "ocr_error", "translate_error"}
    assert "crop" not in got["same_text"] and "crop" not in got["no_text"] and got["low_score"]["crop"] and got["translate_error"]["error"] == "refused"
    assert lines(d.path)[-1]["outcomes"]["low_score"] == 1


def test_crops_can_be_switched_off(tmp_path):
    d = Debug(config.DebugCfg(dir=str(tmp_path), crops=False))
    d.trace(rec()); d.shown(100.07); d.close()
    assert "crop" not in [x for x in lines(d.path) if x.get("event") == "read"][0] and not (tmp_path / "crops").exists()


def test_quality_flags():
    assert quality_flags("Hello there.", "مرحباً بك.") == []
    assert "no_arabic" in quality_flags("Hello there.", "Hola there")
    assert "copied_source" in quality_flags("Hello there.", "hello there.")
    assert "english_left" in quality_flags("Open the door.", "افتح the door")
    assert "much_longer_than_source" in quality_flags("No.", "لا " * 30)
    assert "much_shorter_than_source" in quality_flags("This is a rather long subtitle line.", "نعم")


def test_frame_gaps_stalls_and_periodic_capture_samples(tmp_path):
    d = Debug(config.DebugCfg(dir=str(tmp_path)))
    for i in range(40):
        d.frame(d.t0 + i / 60)
    d.frame(d.t0 + 40 / 60 + 0.2)  # a 200 ms hole in a 60 fps stream
    c = d.capture_stats()
    assert d.stalls == 1 and 55 < c["fps"] < 65 and c["gap_max_ms"] > 150
    d.frame(d.t0 + 6.0)  # past the sampling interval
    d.close()
    cap = [x for x in lines(d.path) if x.get("event") == "capture"]
    assert len(cap) == 1 and cap[0]["stalls"] == 2 and cap[0]["frames"] == 42  # the jump to 6 s is a stall too


def test_mark_saves_the_frame_and_points_at_the_last_subtitle(tmp_path):
    d = Debug(config.DebugCfg(dir=str(tmp_path)))
    d.trace(rec()); d.shown(100.07)
    assert d.mark(np.zeros((10, 10, 3), np.uint8)) == 1
    d.close()
    m = [x for x in lines(d.path) if x.get("event") == "mark"][0]
    assert m["last_subtitle"] == 1 and (tmp_path / m["frame_file"]).exists()


def test_the_log_is_only_enabled_by_the_flag_not_by_the_config():
    with pytest.raises(TypeError):  # unknown key: there is no [debug] enabled
        config.DebugCfg(enabled=True)
    assert not hasattr(config.Config().debug, "enabled")


def test_pipeline_reports_every_read_and_counters():
    frames = [np.zeros((100, 400, 3), np.uint8)] * 5 + [np.full((100, 400, 3), 255, np.uint8)] * 8 + [np.zeros((100, 400, 3), np.uint8)] * 6

    class Cap:
        def __init__(self):
            self.f = iter(frames)

        def read(self):
            time.sleep(0.03)  # the worker keeps up: a newer job replaces a pending one that was not picked up yet
            f = next(self.f, None)
            return (f is not None), f

    class FakeOcr:
        def read(self, im, **kw):
            return "hello" if im.mean() > 100 else ""

    got, stats = [], {}
    run(config.Config(), Cap(), FakeOcr(), translate.build(config.Translate()), emit=lambda m: None, on_trace=got.append, stats=stats)
    tr = [g for g in got if g["outcome"] == "translated"]
    assert len(tr) == 1 and tr[0]["src"] == "hello" and tr[0]["cache_hit"] is False and tr[0]["img"].shape[0] > 0 and tr[0]["history"] == []
    assert tr[0]["t_fire"] <= tr[0]["t_ocr0"] <= tr[0]["t_ocr1"] <= tr[0]["t_mt1"]
    assert any(g["outcome"] == "no_text" for g in got)  # the subtitle going away is a read too
    assert stats["fires"] >= 2 and stats["reads"] >= 2 and stats["errors"] == 0


def test_server_timings_reach_the_trace_and_a_cache_hit_is_flagged(monkeypatch):
    monkeypatch.setattr(translate, "_post_json", lambda *a, **k: {"choices": [{"text": " x "}], "timings": {"prompt_ms": 5.0, "predicted_ms": 30.0}})
    t = translate.Cached(translate.Chat("http://h/v1/chat/completions", "", "ar", style="plamo-pfml"), 8)
    t.translate("one")
    assert translate.last.hit is False and translate.last.timings["predicted_ms"] == 30.0
    t.translate("one")
    assert translate.last.hit is True and translate.last.timings is None


def test_diagnose_reads_a_log_and_names_the_problems(tmp_path):
    d = Debug(config.DebugCfg(dir=str(tmp_path)), {"ocr": {"engine": "trt"}})
    d.trace(rec(1.0)); d.shown(1.07)
    d.trace(rec(2.0, src="Open the door", out="افتح the door", text="Open the door")); d.shown(2.5)  # slow + English left
    d.trace(rec(2.8, src="Open the doors", out="افتح الأبواب", text="Open the doors")); d.shown(2.87)  # near-identical line right after
    d.trace(rec(9.0, src="hello", out="أهلاً"))  # the same English, other Arabic
    d.shown(9.07)
    d.trace({"outcome": "low_score", "text": "h3llo", "score": 0.4, "t_fire": 10.0, "t_ocr0": 10.001, "t_ocr1": 10.01, "img": None})
    d.close()
    path, rows = diagnose.load(str(tmp_path))
    a = diagnose.analyse(rows)
    assert path.endswith(".jsonl") and a["session"]["finished_cleanly"] and a["outcomes"]["translated"] == 4
    assert a["stages"]["ocr_ms"]["median"] == 7.0 and a["slowest"][0]["src"] == "Open the door"
    assert a["flags"] == {"english_left": 1} and len(a["partial_or_flicker"]) == 1 and len(a["inconsistent"]) == 1 and len(a["low_score_reads"]) == 1
    assert any("english_left" in p for p in a["problems"])
