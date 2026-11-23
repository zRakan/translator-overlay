import time
import numpy as np

from translator_overlay import config, translate
from translator_overlay.overlay import Overlay
from translator_overlay.pipeline import run


def test_draw_bar_and_clear():
    ov = Overlay(config.OverlayCfg())
    f = np.full((720, 1280, 3), 40, np.uint8)
    assert (ov.draw(f.copy()) == f).all()  # nothing set -> untouched
    ov.set("سنغادر عند الفجر")
    out = ov.draw(f.copy())
    assert (out != f).any() and (out[:300] == f[:300]).all()  # only the bottom bar changes
    ov.set(None)
    assert (ov.draw(f.copy()) == f).all()


def test_replace_covers_original_box():
    ov = Overlay(config.OverlayCfg(mode="replace"))
    f = np.full((720, 1280, 3), 255, np.uint8)
    ov.set("نص", (300, 600, 700, 650))
    out = ov.draw(f)
    assert (out[592:600, 292:310] < 140).all()  # backdrop dimmed well below the white frame


def test_ocr_never_sees_overlay():
    seen = []

    class Cap:
        def __init__(self):
            self.n = 0

        def read(self):
            self.n += 1
            return self.n < 14, np.full((100, 400, 3), 0 if self.n < 5 else 255, np.uint8)

    class SpyOcr:
        def read(self, im, **kw):
            seen.append(im.copy())
            return "hi" if im.mean() > 100 else ""

    ov = Overlay(config.OverlayCfg(mode="replace"))
    ov.set("x", (0, 0, 400, 100))  # box touches every frame edge: also guards the clamp
    run(config.Config(), Cap(), SpyOcr(), translate.build(config.Translate()),
        on_subtitle=ov.set, on_frame=lambda f: ov.draw(f))
    assert any(i.mean() > 100 for i in seen)  # white subtitle reached OCR despite the black overlay


def test_junk_ocr_text_is_ignored():
    got = []

    class Junk:
        def read(self, im, **kw):
            return "\u4e0a /"  # CJK glyph + slash: no Latin letters -> not a subtitle

    class Cap:
        n = 0

        def read(self, *_):
            self.n += 1
            return self.n < 12, np.full((100, 400, 3), 0 if self.n < 5 else 255, np.uint8)

    run(config.Config(), Cap(), Junk(), translate.build(config.Translate()), emit=got.append)
    assert got == []


def test_text_size_is_stable_across_jittery_line_heights():
    ov = Overlay(config.OverlayCfg())
    for h in (26, 27, 26, 38, 26):  # 38 = an OCR box inflated by a descender
        ov.set("نص", (0, 0, 10, 10), h)
    assert ov.current[2] == 26


def _composite(frame, patch, x, y):
    """What the screen shows when a premultiplied BGRA patch sits over the frame."""
    out = frame.copy().astype(np.int32)
    h, w = patch.shape[:2]
    a = patch[..., 3:4].astype(np.int32)
    out[y : y + h, x : x + w] = patch[..., :3].astype(np.int32) + out[y : y + h, x : x + w] * (255 - a) // 255
    return out


def test_patch_matches_draw_for_blur_and_solid_modes_in_both_layouts():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (360, 640, 3)).astype(np.uint8)
    for mode in ("bar", "replace"):
        for blur in (True, False):
            ov = Overlay(config.OverlayCfg(mode=mode, blur=blur))
            ov.set("\u0623\u0647\u0644\u0627\u064b \u0628\u0643", (200, 300, 440, 330), 20)
            drawn = ov.draw(frame.copy()).astype(np.int32)
            patch, x, y = ov.patch(frame)
            assert patch.dtype == np.uint8 and patch.shape[2] == 4 and (x, y) == (x, y)
            assert np.abs(_composite(frame, patch, x, y) - drawn).max() <= 2, (mode, blur)
            assert (ov.patch(frame)[0] == patch).all()  # patch() must not modify the frame it reads
    ov = Overlay(config.OverlayCfg())
    assert ov.patch(frame) is None  # no subtitle -> nothing to show


def test_text_drawn_at_screen_scale_lands_where_the_scaled_video_puts_it():
    base = np.zeros((360, 640, 3), np.uint8)
    big = np.zeros((720, 1280, 3), np.uint8)
    ov = Overlay(config.OverlayCfg(mode="replace", blur=False))
    ov.set("\u0623\u0647\u0644\u0627", (200, 280, 440, 320), 22)          # OCR box + line height measured in the 640x360 frame
    ov.draw(base, 1.0)
    ov.draw(big, 2.0)                                                         # same subtitle on the 2x frame

    def box(img):
        ys, xs = np.where(img.max(axis=2) > 0)
        return xs.min(), ys.min(), xs.max(), ys.max()

    for a, b in zip(box(base), box(big)):
        assert abs(b - 2 * a) <= 6                                            # the patch is ~2x the size at ~2x the position


def test_speaker_label_is_split_before_translation_and_drawn_in_its_colour():
    from translator_overlay.translate import split_speaker

    assert split_speaker("Michael: Get in the car.") == ("Michael", "Get in the car.")
    assert split_speaker("Get in the car: now") == (None, "Get in the car: now")
    assert split_speaker("OFFICER_2: Stop right there!")[0] == "OFFICER_2"
    ov = Overlay(config.OverlayCfg(blur=False))
    plain, tagged = Overlay(config.OverlayCfg(blur=False)), ov
    frame = np.zeros((300, 900, 3), np.uint8)
    box = (100, 100, 800, 160)
    plain.set("اركب السيارة، الآن", box, 34)
    tagged.set("اركب السيارة، الآن", box, 34, "Michael")
    a, b = plain.draw(frame.copy()), tagged.draw(frame.copy())
    col = ov.speaker_color("Michael")
    assert col == ov.speaker_color("MICHAEL") and col != (255, 255, 255)
    px = b.reshape(-1, 3)[:, ::-1]  # BGR -> RGB
    assert (np.abs(px.astype(int) - np.array(col)).sum(1) < 40).sum() > 30  # pixels in the speaker colour exist
    ap = a.reshape(-1, 3)[:, ::-1]
    assert (np.abs(ap.astype(int) - np.array(col)).sum(1) < 40).sum() == 0   # and not without a speaker
    ov2 = Overlay(config.OverlayCfg(speaker=False))
    ov2.set("نص", box, 30, "Michael")
    assert ov2.current[3] is None


def test_pipeline_hides_the_speaker_from_the_model_and_hands_it_to_the_overlay():
    seen, shown = [], []

    class Ocr:
        last_box, last_line_h, last_score = (0, 0, 10, 10), 20, 0.99

        def read(self, im, **kw):
            return "Michael: Get in the car."

    class Tr:
        def translate(self, text, history=None):
            seen.append(text)
            return "اركب السيارة."

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return self.n <= 25, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    cfg = config.Config()
    cfg.translate.strip_speaker = True
    run(cfg, Cap(), Ocr(), Tr(), on_subtitle=lambda *a: shown.append(a))
    assert seen and all(s == "Get in the car." for s in seen)
    assert any(len(a) == 4 and a[3] == "Michael" for a in shown)


def test_speaker_names_in_arabic_from_the_list():
    from translator_overlay.speakers import arabic_name

    assert arabic_name("Lucia") == "لوسيا" and arabic_name("LUCIA") == "لوسيا"
    assert arabic_name("The Man") == "الرجل" and arabic_name("Guard 2") == "الحارس 2"
    assert arabic_name("Luqia") == "لوسيا"          # an OCR slip of a listed name
    assert arabic_name("Zorvath") is None and arabic_name("Tip") is None
    seen = []

    class Ocr:
        last_box, last_line_h, last_score = (0, 0, 10, 10), 20, 0.99

        def read(self, im, **kw):
            return "Lucia: Get in the car."

    class Tr:
        def translate(self, text, history=None):
            return "اركب السيارة."

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return self.n <= 25, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    cfg = config.Config()
    cfg.translate.strip_speaker = True
    run(cfg, Cap(), Ocr(), Tr(), on_subtitle=lambda *a: seen.append(a))
    assert any(len(a) == 4 and a[3] == "لوسيا" for a in seen)
    cfg.overlay.speaker_arabic = False
    seen.clear()
    run(cfg, Cap(), Ocr(), Tr(), on_subtitle=lambda *a: seen.append(a))
    assert any(len(a) == 4 and a[3] == "Lucia" for a in seen)


def test_two_speakers_at_once_are_split_translated_separately_and_drawn_on_two_lines():
    from translator_overlay.speakers import arabic_name
    from translator_overlay.translate import split_segments

    k = lambda n: arabic_name(n) is not None
    assert split_segments("I don't have the time, so... Lucia: Do you have a minute?", k) == [(None, "I don't have the time, so..."), ("Lucia", "Do you have a minute?")]
    assert split_segments("Jason: Yeah, no. Lucia: Oh my God, stop..", k) == [("Jason", "Yeah, no."), ("Lucia", "Oh my God, stop..")]
    assert split_segments("Lucia: Note: this is fine", k) == [("Lucia", "Note: this is fine")]   # a colon inside a sentence is not a speaker
    seen, shown = [], []

    class Ocr:
        last_box, last_line_h, last_score = (0, 0, 300, 60), 20, 0.99

        def read(self, im, **kw):
            return "Jason: Yeah, no. Lucia: Oh my God, stop.."

    class Tr:
        def translate(self, text, history=None):
            seen.append(text)
            return {"Yeah, no.": "أجل، لا.", "Oh my God, stop..": "يا إلهي، توقف.."}[text]

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return self.n <= 25, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    cfg = config.Config()
    cfg.translate.strip_speaker = True
    run(cfg, Cap(), Ocr(), Tr(), on_subtitle=lambda *a: shown.append(a))
    assert seen[:2] == ["Yeah, no.", "Oh my God, stop.."]                   # no label reaches the model
    assert any(a[0] == [("جيسون", "أجل، لا."), ("لوسيا", "يا إلهي، توقف..")] for a in shown)
    ov = Overlay(config.OverlayCfg(blur=False))
    one, two = Overlay(config.OverlayCfg(blur=False)), ov
    frame = np.zeros((300, 900, 3), np.uint8)
    one.set("أجل، لا.", (100, 100, 800, 140), 34, "جيسون")
    two.set([("جيسون", "أجل، لا."), ("لوسيا", "يا إلهي، توقف..")], (100, 100, 800, 180), 34)
    for o, cols in ((one, 1), (two, 2)):
        px = o.draw(frame.copy()).reshape(-1, 3)[:, ::-1].astype(int)
        found = [c for c in {o.speaker_color("جيسون"), o.speaker_color("لوسيا")} if (np.abs(px - np.array(c)).sum(1) < 40).sum() > 20]
        assert len(found) == cols
    assert two._place(300, 900)[0].shape[0] > one._place(300, 900)[0].shape[0]       # two lines are taller than one


def test_subtitle_taller_than_the_region_is_read_again_with_the_strip_above():
    cfg = config.Config()
    cfg.region.y = 0.5          # something above the region to add
    cfg.translate.strip_speaker = True
    calls, shown = [], []

    class Ocr:
        extend_up = True
        last_box = last_line_h = last_score = None

        def read(self, im, x_range=None, skip_caps=False, max_gap=None):
            calls.append((im.shape[0], max_gap))
            self.last_line_h, self.last_score = 20, 0.99
            if max_gap:   # the taller picture: the whole caption, the first line starts 30 px down
                self.last_box = (0, 30, 300, 120)
                return "Jason: Yeah, I don't really have one right now. Lucia: Do you have a minute?"
            self.last_box = (0, 0, 300, 40)   # touches the top of the region: the rest of the caption is above it
            return "Lucia: Do you have a minute?"

    class Tr:
        def translate(self, text, history=None):
            return "ترجمة"

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return self.n <= 25, np.full((200, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    stats = {"fires": 0, "reads": 0, "same_text": 0, "low_score": 0, "errors": 0}
    run(cfg, Cap(), Ocr(), Tr(), on_subtitle=lambda *a: shown.append(a), stats=stats)
    first, again = calls[0], calls[1]
    assert again[1] == 1.8 and again[0] > first[0]       # a second, taller read of the same moment
    assert stats.get("extended", 0) >= 1
    assert len(shown[0][0]) == 2 and shown[0][1][1] == 30 + 100 - (again[0] - first[0])   # box moved up by the strip height (region top = y 100)


def test_an_unexpected_error_in_the_worker_costs_one_job_not_all_later_subtitles():
    shown, boom = [], [True]

    class Ocr:
        last_box, last_line_h = (0, 0, 10, 10), 20

        @property
        def last_score(self):
            if boom[0]:
                boom[0] = False
                raise RuntimeError("unexpected")   # outside every try block of the worker
            return 0.99

        def read(self, im, **kw):
            return "Michael: Get in the car."

    class Tr:
        def translate(self, text, history=None):
            return "اركب السيارة."

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return self.n <= 60, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    logs, stats = [], {"fires": 0, "reads": 0, "same_text": 0, "low_score": 0, "errors": 0}
    run(config.Config(), Cap(), Ocr(), Tr(), emit=logs.append, on_subtitle=lambda *a: shown.append(a), stats=stats)
    assert stats["worker_restarts"] == 1 and any("worker error" in l for l in logs)
    assert shown and shown[0][0]                         # translation went on after the failure


def test_context_is_kept_per_speaker_and_a_translated_line_is_reused():
    from translator_overlay.pipeline import ConversationContext

    c = ConversationContext(lines=2, gap_s=5, when="always")
    c.visible(0.0)
    c.add("Hi Jason.", "مرحبا", "lucia")
    c.add("Yo.", "يو", "jason")
    assert c.history("x", "lucia") == [("Hi Jason.", "مرحبا")] and c.history("x", "jason") == [("Yo.", "يو")]   # not one mixed thread
    assert c.recall("hi  JASON.", "lucia") == "مرحبا" and c.recall("Hi Jason.", "jason") is None
    assert c.recall("Hi Jason,", "lucia") == "مرحبا" and c.recall("Hi Jason...", "lucia") == "مرحبا"   # the end mark of a cut-off line varies
    c.visible(20.0)                                       # a new conversation forgets both
    assert c.recall("Hi Jason.", "lucia") is None and c.history("x", "lucia") == []


def test_a_line_already_translated_is_not_translated_again_when_the_caption_around_it_changes():
    calls, shown, reads = [], [], iter(["Jason: Yeah, no.", "Jason: Yeah, no. Lucia: Hi there!", "Jason: Yeah, no. Lucia: Hi there!"] + ["Lucia: Hi there!"] * 30)

    class Ocr:
        last_box, last_line_h, last_score = (0, 0, 10, 10), 20, 0.99

        def read(self, im, **kw):
            return next(reads)

    class Tr:
        def translate(self, text, history=None):
            calls.append((text, history))
            return f"ar({text})"

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.02)
            return self.n <= 70, np.full((100, 400, 3), 255 if (self.n // 10) % 2 else 0, np.uint8)

    cfg = config.Config()
    cfg.translate.strip_speaker = True
    cfg.translate.context = True
    cfg.translate.context_when = "always"
    run(cfg, Cap(), Ocr(), Tr(), on_subtitle=lambda *a: shown.append(a))
    texts = [t for t, _ in calls]
    assert texts.count("Yeah, no.") == 1                                   # translated once, reused in the two-speaker caption
    assert all(h in (None, [], ()) or "Yeah, no." not in [x for x, _ in h] for t, h in calls if t == "Hi there!")  # Lucia's thread has no Jason lines
    assert any(isinstance(a[0], list) and [x[1] for x in a[0]] == ["ar(Yeah, no.)", "ar(Hi there!)"] for a in shown)


def test_after_a_tall_caption_the_next_reads_go_straight_to_the_taller_picture():
    cfg = config.Config()
    cfg.region.y = 0.5
    calls = []

    class Ocr:
        extend_up = True
        last_box = last_line_h = last_score = None

        def read(self, im, x_range=None, skip_caps=False, max_gap=None):
            calls.append(bool(max_gap))
            self.last_line_h, self.last_score = 20, 0.99
            self.last_box = (0, 30, 300, 120) if max_gap else (0, 0, 300, 40)
            return f"Jason: line {len(calls)} of a tall caption here."

    class Tr:
        def translate(self, text, history=None):
            return "ترجمة"

    class Cap:
        n = 0

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return self.n <= 90, np.full((200, 400, 3), 255 if (self.n // 6) % 2 else 0, np.uint8)

    stats = {"fires": 0, "reads": 0, "same_text": 0, "low_score": 0, "errors": 0}
    run(cfg, Cap(), Ocr(), Tr(), stats=stats)
    assert calls[:2] == [False, True]                      # first the normal read, then the taller one
    assert all(calls[2:]) and len(calls) > 4               # afterwards only the taller one
    assert stats["extended"] == 1 and stats["tall_direct"] >= 2


def test_a_sentence_ending_in_a_capitalised_word_does_not_swallow_the_next_speaker_label():
    from translator_overlay.translate import split_segments, split_speaker

    t = "Brian: Hey, you've reached the harbour office of Pier Nine and Marina. Ms. Sharleen: I guess it went well."
    assert split_segments(t) == [("Brian", "Hey, you've reached the harbour office of Pier Nine and Marina."), ("Ms. Sharleen", "I guess it went well.")]
    assert split_speaker("Ms. Sharleen: Hello, friend.") == ("Ms. Sharleen", "Hello, friend.")
    assert split_segments("Jay Norris: Hi. Officer Smith: Stop!")[1][0] == "Officer Smith"


def test_a_colon_read_as_a_letter_still_gives_the_speaker_and_a_mark_saves_the_shown_picture(tmp_path):
    from translator_overlay.debug import Debug
    from translator_overlay.speakers import arabic_name, is_listed
    from translator_overlay.translate import split_segments

    k = lambda n: arabic_name(n) is not None
    t = 'Ms. Sharleene Or maybe only wave at anyone who says "cheese". Brian: Hey, you\'ve reached'
    assert split_segments(t, k, is_listed) == [("Ms. Sharleen", 'Or maybe only wave at anyone who says "cheese".'), ("Brian", "Hey, you've reached")]
    assert split_segments("Brians call me back later.", k, is_listed) == [("Brian", "call me back later.")]
    assert split_segments("Mike runs. Mikes friend said hi", k, is_listed) == [(None, "Mike runs. Mikes friend said hi")]   # a possessive typo mid-text is not a label
    assert split_segments("Brian's boat is here.", k, is_listed) == [(None, "Brian's boat is here.")]
    d = Debug(config.DebugCfg(dir=str(tmp_path / "dbg"), crops=False), {})
    d.mark(np.zeros((20, 30, 3), np.uint8), shown=np.full((20, 30, 3), 7, np.uint8), overlay="(overlay state)")
    d.close()
    import json
    rows = [json.loads(l) for l in open(d.path, encoding="utf-8")]
    m = next(r for r in rows if r["event"] == "mark")
    assert m["overlay"] == "(overlay state)" and m["shown_file"].endswith("-shown.png") and (tmp_path / "dbg" / m["shown_file"]).exists()


def test_recorder_streams_frames_to_ffmpeg_and_marks_the_time(tmp_path, monkeypatch):
    import sys

    from translator_overlay import recorder

    stub = tmp_path / "fake_ffmpeg.py"      # reads the raw video from stdin, writes the byte count to the last argument
    stub.write_text("import sys\nn = 0\nwhile (b := sys.stdin.buffer.read(65536)):\n    n += len(b)\nopen(sys.argv[-1], 'w').write(str(n))\n")
    monkeypatch.setattr(recorder, "FFMPEG", str(stub))
    monkeypatch.setattr(recorder.subprocess, "Popen", (lambda orig: lambda cmd, **kw: orig([sys.executable] + cmd[:1] + cmd[1:], **kw))(recorder.subprocess.Popen))
    r = recorder.Recorder(str(tmp_path / "out" / "rec.mkv"))
    for i in range(10):
        r.write(np.full((6, 8, 3), i, np.uint8))
    el = r.mark("a conversation")
    r.close()
    assert r.frames == 10 and r.dropped == 0 and el >= 0
    assert (tmp_path / "out" / "rec.mkv").read_text() == str(10 * 6 * 8 * 3)
    assert (tmp_path / "out" / "rec.mkv.marks.txt").read_text().strip().endswith("a conversation")


def test_record_section_and_auto_recording_path(tmp_path):
    cfg = config.load(None)
    assert cfg.record.dir == "recordings" and cfg.record.ffmpeg == ""
    toml = tmp_path / "c.toml"
    toml.write_text('[record]\ndir = "out"\nffmpeg = "/opt/ffmpeg"\n', encoding="utf-8")
    c = config.load(str(toml))
    assert c.record.dir == "out" and c.record.ffmpeg == "/opt/ffmpeg"
    import pytest

    toml.write_text("[record]\nfolder = 'x'\n", encoding="utf-8")
    with pytest.raises(TypeError):
        config.load(str(toml))                  # unknown keys still fail loudly
