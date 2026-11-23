import time

import numpy as np

from translator_overlay import config, translate
from translator_overlay.change import ChangeDetector
from translator_overlay.pipeline import run


def img(v):
    return np.full((40, 200, 3), v, np.uint8)


def test_detector_fires_once_per_stable_change():
    d = ChangeDetector(threshold=6, stable_frames=3)
    fires = [d.update(img(v)) for v in [0, 0, 0, 0, 0, 200, 0, 200, 200, 200, 200, 200]]
    # first subtitle (blank->stable) fires once, flicker doesn't, final stable change fires once
    assert fires.count(True) == 2 and fires[-1] is False


def test_cache_hits_inner_once_and_evicts():
    calls = []

    class Inner:
        def translate(self, t):
            calls.append(t)
            return t.upper()

    c = translate.Cached(Inner(), 2)
    assert [c.translate(x) for x in ["a", "a", "b", "c", "a"]] == ["A", "A", "B", "C", "A"]
    assert calls == ["a", "b", "c", "a"]  # "a" evicted by c, refetched


def test_unknown_backend_fails_loudly():
    cfg = config.Translate(backend="nope")
    try:
        translate.build(cfg)
    except ValueError:
        return
    raise AssertionError


def test_pipeline_end_to_end_with_fakes():
    frames = [np.zeros((100, 400, 3), np.uint8)] * 5 + [np.full((100, 400, 3), 255, np.uint8)] * 8

    class Cap:
        def __init__(self):
            self.f = iter(frames)

        def read(self):
            f = next(self.f, None)
            return (f is not None), f

    class FakeOcr:
        def read(self, im, **kw):
            return "hello" if im.mean() > 100 else ""

    out = []
    run(config.Config(), Cap(), FakeOcr(), translate.build(config.Translate()), emit=out.append)
    assert len(out) == 1 and "'hello' -> 'hello'" in out[0]


def test_noisy_rereads_of_same_subtitle_translate_once():
    reads = iter(["Put it back on the shelf for her, tell her to wait", "Put it back on the shelf for her,S tell her to wait",
                  "Put it back on the shelf for her, tell er to wait", "Where are you going, Captain?"])
    seen = []

    class Ocr:
        last = ""

        def read(self, im, **kw):
            self.last = next(reads, self.last)  # after the script ends, keep returning the last read
            return self.last

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.005)  # let the worker keep up (the queue drops stale detections by design)
            # alternate blank/white so the detector fires 4 times
            return self.n <= 40, np.full((100, 400, 3), 255 if (self.n // 5) % 2 else 0, np.uint8)

    run(config.Config(), Cap(), Ocr(), translate.build(config.Translate()), emit=seen.append)
    assert len(seen) == 2 and "Captain" in seen[1]


def test_empty_translation_is_not_shown():
    shown, logs = [], []

    class Ocr:
        def read(self, im, **kw):
            return "Okay, a rifle..."

    class Empty:
        def translate(self, t):
            return ""

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            return self.n < 12, np.full((100, 400, 3), 0 if self.n < 5 else 255, np.uint8)

    run(config.Config(), Cap(), Ocr(), Empty(), emit=logs.append, on_subtitle=lambda *a: shown.append(a))
    assert all(a[0] is None for a in shown) and any("no usable" in l for l in logs)


def test_strip_speaker_only_removes_name_labels():
    s = translate.strip_speaker
    assert s("Parrot: Bang! Bang!") == "Bang! Bang!"
    assert s("Jason: Hey, I'm here.") == "Hey, I'm here."
    assert s("Okay: this is not a label? no, it is") == "this is not a label? no, it is"  # still a Name: prefix
    assert s("Where are you going, Captain?") == "Where are you going, Captain?"
    assert s("It is 10:30 now") == "It is 10:30 now"  # lowercase start / digits: untouched
    assert s("Parrot:") == "Parrot:"  # nothing after the label: keep


def test_chat_backend_sends_hymt_prompt_and_parses_reply():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["auth"] = self.headers.get("Authorization")
            out = json.dumps({"choices": [{"message": {"content": "  مرحباً \n"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = config.Translate(backend="chat", chat_url=f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", api_key="k")
    assert translate.build(cfg).translate("Hello!") == "مرحباً"
    prompt = seen["body"]["messages"][0]["content"]
    assert "into Arabic" in prompt and prompt.endswith("Hello!") and seen["auth"] == "Bearer k"
    srv.shutdown()


def test_fast_detector_fires_on_first_frame_and_again_after_a_fade():
    def run_seq(fast, seq):
        d = ChangeDetector(threshold=6, stable_frames=3, fast=fast)
        return [i for i, v in enumerate(seq) if d.update(img(v))]

    instant = [0] * 5 + [200] * 8                       # whole subtitle in one frame
    assert run_seq(False, instant) == [2, 8]            # baseline: blank settles at 2; subtitle (onset at 5) only fires after 3 still frames, at 8
    assert run_seq(True, instant) == [2, 5]             # fast: subtitle fires on its first frame (5) -> 3 frames sooner
    fade = [0] * 5 + [60, 120, 200] + [200] * 6         # fades in over 3 frames
    assert run_seq(True, fade) == [2, 5, 10]           # early read mid-fade (5), then again once settled on the final text (10)


def test_plamo_pfml_prompt_matches_the_documented_format():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["body"] = self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"text": " مرحباً "}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = config.Translate(backend="chat", chat_style="plamo-pfml", chat_url=f"http://127.0.0.1:{srv.server_port}/v1/chat/completions")
    assert translate.build(cfg).translate("Hello!") == "مرحباً"
    assert seen["path"] == "/v1/completions"
    assert seen["body"]["prompt"] == ("<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en"
                                      "<|plamo:msg|>Hello!<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ar<|plamo:msg|>")
    assert seen["body"]["stop"] == ["<|plamo:tag|>", "<|plamo:bos|>"] and seen["body"]["temperature"] == 0
    srv.shutdown()


def test_poll_reads_a_subtitle_that_appears_over_constant_motion():
    def frame(i):  # background flips every frame (never still); a bright "subtitle" block appears from frame 15
        f = np.full((40, 200, 3), (i % 2) * 120, np.uint8)
        if i >= 15:
            f[10:30, 20:180] = 255
        return f

    def fires(**kw):
        d = ChangeDetector(threshold=6, stable_frames=3, **kw)
        return [i for i in range(40) if d.update(frame(i))]

    assert [i for i in fires(fast=True) if i >= 15] == []                    # never settles -> the text is never read
    polled = [i for i in fires(fast=True, poll_frames=5) if i >= 15]
    assert polled and polled[0] <= 20                                        # read within poll_frames of appearing


def test_low_confidence_read_is_ignored_not_treated_as_no_subtitle():
    reads = iter([("Hello there, friend.", 0.99), ("Hel o thre frnd", 0.5), ("", None), ("No.", 0.98)])
    shown = []

    class Ocr:
        last_score = None

        def read(self, im, **kw):
            t, self.last_score = next(reads)
            return t

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)  # let the worker keep up
            return self.n <= 80, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    cfg = config.Config()
    cfg.ocr.min_score = 0.9
    run(cfg, Cap(), Ocr(), translate.build(config.Translate()), on_subtitle=lambda *a: shown.append(a[0]))
    assert shown[:2] == ["Hello there, friend.", None]   # the shaky read in between neither replaced nor cleared it
    assert "No." in shown                               # 2-letter subtitle survives the letter filter


def test_parse_cam_spec_and_camera_source_keeps_only_the_newest_frame():
    from translator_overlay.capture import CameraSource, parse_cam

    assert parse_cam("cam:0") == {"index": 0, "width": 0, "height": 0, "fps": 0, "fourcc": "", "backend": "dshow"}
    assert parse_cam("cam:2:1920x1080@60:mjpg:msmf") == {"index": 2, "width": 1920, "height": 1080, "fps": 60, "fourcc": "MJPG", "backend": "msmf"}
    assert parse_cam("cam:1:NV12:1280x720")["fps"] == 0

    class FakeCap:
        def __init__(self):
            self.n, self.props = 0, {}

        def isOpened(self):
            return True

        def set(self, k, v):
            self.props[k] = v

        def get(self, k):
            return self.props.get(k, 0)

        def read(self):
            self.n += 1
            time.sleep(0.002)
            return True, np.full((4, 4, 3), self.n % 250, np.uint8)

        def release(self):
            pass

    cam = CameraSource(width=640, height=360, fps=60, fourcc="MJPG", _cap=FakeCap())
    cam.read()
    time.sleep(0.15)                       # a slow consumer: ~75 frames were grabbed meanwhile
    ok, f = cam.read()
    assert ok and cam._seq - cam._seen == 0 and cam._seq > 20   # we got the newest frame, not the oldest queued one
    assert f[0, 0, 0] == cam._frame[0, 0, 0]
    assert cam._cap.props[3] == 640 and cam._cap.props[4] == 360 and cam._cap.props[38] == 1  # width, height, BUFFERSIZE
    cam.release()


def test_conversation_context_only_while_lines_belong_to_one_conversation():
    from translator_overlay.pipeline import ConversationContext

    c = ConversationContext(lines=2, gap_s=5, when="always")
    c.visible(0.0)
    assert c.history() == []                                  # first line ever: nothing before it
    c.add("A", "a")
    c.visible(1.0)                                            # next subtitle 1 s later: same conversation
    assert c.history() == [("A", "a")]
    c.add("B", "b")
    c.visible(2.0)
    c.add("C", "c")
    assert c.history() == [("B", "b"), ("C", "c")]            # only the last `lines`
    c.visible(4.0)                                            # the same subtitle still on screen keeps the conversation alive
    c.visible(8.5)                                            # 4.5 s after it was last visible: still one conversation
    assert len(c.history()) == 2
    c.visible(20.0)                                           # 11.5 s of silence: a new scene / unrelated text
    assert c.history() == []
    assert ConversationContext(lines=0, gap_s=5).history() == []  # context off


def test_plamo_pfml_history_prompt_follows_the_model_card_layout():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"text": "x"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = config.Translate(backend="chat", chat_style="plamo-pfml", chat_url=f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", cache_size=0)
    tr = translate.build(cfg)
    tr.translate("is to share a problem with me.", history=[("'Cause I would guess your only reason", "AR1"), ("for calling me", "AR2<|plamo:tag|>")])
    I, O = "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>", "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ar<|plamo:msg|>"
    assert seen["body"]["prompt"] == ("<|plamo:tag|>dataset<|plamo:msg|>translation"
                                      + I + "'Cause I would guess your only reason" + O + "AR1"
                                      + I + "for calling me" + O + "AR2tag|>"      # reserved tokens stripped from history text
                                      + I + "is to share a problem with me." + O)
    tr.translate("no history here")
    assert seen["body"]["prompt"] == "<|plamo:tag|>dataset<|plamo:msg|>translation" + I + "no history here" + O
    srv.shutdown()


def test_pipeline_passes_history_only_when_context_is_on():
    def run_with(context):
        calls, reads = [], iter(["Hello there, my friend.", "How are you doing today?", "I am doing very well."])

        class Tr:
            def translate(self, text, history=None):
                calls.append((text, history))
                return "ar:" + text

        class Ocr:
            def read(self, im, **kw):
                return next(reads, "")

        class Cap:
            n = 0

            def read(self):
                self.n += 1
                time.sleep(0.01)
                return self.n <= 90, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

        cfg = config.Config()
        cfg.translate.context, cfg.translate.context_lines, cfg.translate.context_when = context, 2, "always"
        run(cfg, Cap(), Ocr(), Tr())
        return calls

    on = run_with(True)
    assert [c[1] for c in on[:3]] == [None, [("Hello there, my friend.", "ar:Hello there, my friend.")],
                                      [("Hello there, my friend.", "ar:Hello there, my friend."), ("How are you doing today?", "ar:How are you doing today?")]]
    assert all(c[1] is None for c in run_with(False))   # off: translators are called exactly as before


def test_cache_does_not_reuse_a_translation_across_different_context():
    class Inner:
        def translate(self, t, history=None):
            return f"{t}|{history[0][0] if history else '-'}"

    c = translate.Cached(Inner(), 10)
    assert c.translate("x") == "x|-"
    assert c.translate("x", [("prev1", "p1")]) == "x|prev1"
    assert c.translate("x", [("prev2", "p2")]) == "x|prev2"     # different context -> not served from the cache
    assert c.translate("x", [("prev1", "p1")]) == "x|prev1"     # same context -> cache hit


def test_continuation_mode_gives_history_only_to_lines_that_continue_a_sentence():
    from translator_overlay.pipeline import ConversationContext

    def hist(prev_lines, current):
        c = ConversationContext(lines=3, gap_s=5)  # default mode = continuation
        for i, p in enumerate(prev_lines):
            c.visible(float(i))
            c.add(p, "ar:" + p)
        c.visible(float(len(prev_lines)))
        return [s for s, _ in c.history(current)]

    assert hist(["I know all about that."], "Guys like us gotta stick together.") == []                      # complete + new sentence: alone
    assert hist(["'Cause I would guess your only reason for calling me"], "is to share a problem with me.") == ["'Cause I would guess your only reason for calling me"]
    assert hist(["Put it back on the shelf for her,"], "Tell her to wait outside.") == ["Put it back on the shelf for her,"]  # unfinished (comma)
    assert hist(["He said hello."], "and then he left.") == ["He said hello."]                                 # lowercase start = continuation
    assert hist(["Fine.", "As long as it's under a thousand", "or some such"], "from a three letter agency.") == ["As long as it's under a thousand", "or some such"]
    assert hist(["It is done.", "Really?"], "Yes.") == []


def test_a_failing_timing_callback_does_not_stop_later_subtitles():
    shown, reads = [], iter(["Contacts", "Where are you going?", "Wait for me, please."])

    class Tr:
        def translate(self, text, history=None):
            return "ar:" + text

    class Ocr:
        def read(self, im, **kw):
            return next(reads, "")

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return self.n <= 90, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    def bad_timing(*a):
        raise TypeError("boom")

    cfg = config.Config()
    run(cfg, Cap(), Ocr(), Tr(), on_subtitle=lambda t, b, h: shown.append(t), on_timing=bad_timing)
    assert [x for x in shown if x] == ["ar:Contacts", "ar:Where are you going?", "ar:Wait for me, please."]


def test_a_copied_line_is_retried_alone_and_never_becomes_context():
    calls, reads = [], iter(["WhatUp", "Contacts", "Messages"])

    class Tr:  # copies the input whenever it is shown a copied example (what the real model did), translates otherwise
        def translate(self, text, history=None):
            calls.append((text, history))
            return text if history else "ar:" + text

    class Ocr:
        def read(self, im, **kw):
            return next(reads, "")

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return self.n <= 90, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    shown = []
    cfg = config.Config()
    cfg.translate.context, cfg.translate.context_lines, cfg.translate.context_when = True, 2, "always"
    run(cfg, Cap(), Ocr(), Tr(), on_subtitle=lambda t, b, h: shown.append(t))
    assert [x for x in shown if x] == ["ar:WhatUp", "ar:Contacts", "ar:Messages"]  # no English copies reach the screen


def test_region_clamp_and_save_keep_the_rest_of_the_config(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('# top comment\n[source]\ndevice = "cam:0"\n\n[region]\nx = 0.20   # keep\ny = 0.855\nw = 0.60\nh = 0.14\n\n[detect]\nfast = true\n', encoding="utf-8")
    r = config.Region(x=0.95, y=-0.2, w=0.30, h=0.001)
    config.clamp_region(r)
    assert (r.x, r.y, r.w, r.h) == (0.70, 0.0, 0.30, 0.02)
    config.save_region(str(f), r)
    t = f.read_text(encoding="utf-8")
    assert "# top comment" in t and 'device = "cam:0"' in t and "fast = true" in t
    got = config.load(str(f)).region
    assert (got.x, got.y, got.w, got.h) == (0.70, 0.0, 0.30, 0.02)
    g = tmp_path / "d.toml"
    g.write_text("[detect]\nfast = true\n", encoding="utf-8")  # no [region] yet: it gets appended
    config.save_region(str(g), config.Region(0.1, 0.2, 0.3, 0.4))
    assert config.load(str(g)).region.h == 0.4


def test_audio_delay_line_delays_by_exactly_the_requested_frames():
    from translator_overlay.audio import DelayLine

    d = DelayLine(5, channels=1)
    out = [d.process(np.arange(i * 4, i * 4 + 4, dtype=np.float32).reshape(-1, 1)).ravel().tolist() for i in range(3)]
    flat = sum(out, [])
    assert flat == [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6]          # silence for 5 frames, then the input, in order
    assert DelayLine(0).process(np.ones((4, 2), np.float32)).shape == (4, 2)  # no delay: unchanged


def test_app_waits_for_a_capture_card_instead_of_exiting(monkeypatch):
    from translator_overlay import __main__ as app

    calls = []

    def flaky_open(device):
        calls.append(device)
        if len(calls) < 3:
            raise RuntimeError("cannot open capture device 0 (msmf)")
        return "CAP"

    monkeypatch.setattr(app, "open_source", flaky_open)
    monkeypatch.setattr(app.time, "sleep", lambda s: None)
    assert app.open_when_ready("cam:0:1920x1080@60:MJPG:msmf", show=False) == "CAP" and len(calls) == 3

    def missing(device):
        raise RuntimeError("cannot open video source 'x.mp4'")

    monkeypatch.setattr(app, "open_source", missing)
    import pytest

    with pytest.raises(RuntimeError):              # a missing file is a config error: fail at once, do not wait forever
        app.open_when_ready("x.mp4", show=False)


def test_local_translator_puts_the_target_tag_in_front_of_the_source():
    from translator_overlay.translate import Local

    seen = []

    class Tok:
        def __call__(self, text, **kw):
            seen.append(text)
            raise StopIteration  # enough: we only look at what reaches the tokenizer

    loc = Local.__new__(Local)
    loc.tok, loc.prefix, loc.beams = Tok(), ">>ara<< ", 1
    try:
        loc._gen("Where are you going?")
    except StopIteration:
        pass
    assert seen == [">>ara<< Where are you going?"]
    plain = Local.__new__(Local)
    plain.prefix = ""
    assert plain.prefix + "Hi" == "Hi"


def test_a_failing_log_sink_does_not_stop_translation():
    import time

    first = [np.zeros((100, 400, 3), np.uint8)] * 5 + [np.full((100, 400, 3), 255, np.uint8)] * 8
    second = [np.zeros((100, 400, 3), np.uint8)] * 5 + [np.full((100, 400, 3), 200, np.uint8)] * 8
    shown = []

    class Cap:
        def __init__(self):
            self.f = iter(first)
            self.phase2 = False

        def read(self):
            f = next(self.f, None)
            if f is None and not self.phase2:  # feed the second subtitle only once the first was shown (the job queue drops stale jobs)
                self.phase2, deadline = True, time.time() + 5
                while not shown and time.time() < deadline:
                    time.sleep(0.001)
                self.f = iter(second)
                f = next(self.f, None)
            return (f is not None), f

    class FakeOcr:
        def read(self, im, **kw):
            return "first line here" if im.mean() > 230 else "second line there" if im.mean() > 100 else ""

    def broken(msg):
        raise UnicodeEncodeError("charmap", "", 0, 1, "cannot encode")

    import threading

    t = threading.Thread(target=run, daemon=True, args=(config.Config(), Cap(), FakeOcr(), translate.build(config.Translate())),
                         kwargs=dict(emit=broken, on_subtitle=lambda *a: a[0] and shown.append(a[0])))
    t.start()
    t.join(20)  # without the guard the worker dies and shutdown would wait on the full queue forever
    assert not t.is_alive()
    assert shown == ["first line here", "second line there"]  # the second subtitle is still translated after the first log line failed
