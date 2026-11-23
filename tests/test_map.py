import json
import time

import numpy as np

from translator_overlay import config
from translator_overlay.mapstate import MapDetector, PhraseBook, fix_case, group_blocks
from translator_overlay.pipeline import run

PINK_BGR = (186, 142, 255)  # median colour of the selected tab word


def frame(pink: bool, w=640, h=360):
    f = np.full((h, w, 3), (60, 40, 30), np.uint8)
    x0, y0, x1, y1 = config.MapCfg().tab_box
    f[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = PINK_BGR if pink else (255, 255, 255)
    return f


def test_tab_detector_needs_pink_and_has_hysteresis():
    d = MapDetector(config.MapCfg(check_every=1))
    seq = [d.update(frame(p)) for p in [False, True, True, True, False, True, False, False, False]]
    assert seq == [False, False, True, True, True, True, True, True, False]  # on after 2 pink, off after 3 non-pink


def test_white_tab_and_pink_elsewhere_do_not_trigger():
    d = MapDetector(config.MapCfg(check_every=1))
    f = frame(False)
    f[200:300, 100:300] = PINK_BGR  # a pink blip somewhere else on the map
    assert not any(d.update(f) for _ in range(6))


def L(t, x0, y0, x1, y1):
    return (t, (x0, y0, x1, y1))


CARD = [L("SAMPLE SHOP NAME", 932, 315, 1292, 353), L("Small store selling clothing, footwear,", 929, 363, 1420, 398),
        L("and accessories at a bargain.", 929, 401, 1429, 437), L("685 m", 600, 400, 660, 430),
        L("STYLE", 930, 448, 995, 479), L("THRIFT", 1353, 449, 1425, 480)]


def test_card_blocks_title_paragraph_and_tags():
    b = group_blocks(CARD)
    assert [x.text for x in b] == ["SAMPLE SHOP NAME", "Small store selling clothing, footwear, and accessories at a bargain.", "STYLE", "THRIFT"]
    assert b[1].box == (929, 363, 1429, 437)
    assert fix_case(b[0].text) == "Sample Shop Name" and fix_case(b[1].text) == b[1].text


def test_text_elsewhere_is_its_own_label_and_does_not_join_the_card():
    far = CARD + [L("Some street name", 100, 900, 400, 930)]
    blocks = group_blocks(far)
    card = [x for x in blocks if x.anchor != "center"]
    label = [x for x in blocks if x.text == "Some street name"]
    assert "Some street name" not in [x.text for x in card]          # never merged into the card's blocks
    assert len(label) == 1 and label[0].anchor == "center"           # but translated, centred on where it was
    assert group_blocks([L("685 m", 1, 1, 50, 20)]) == []            # a distance label is not a place name


def test_zoomed_out_map_translates_every_label_not_just_the_biggest():
    labels = [L("Northfield", 600, 140, 790, 190), L("East Harbor County", 960, 215, 1330, 260), L("Village", 1010, 305, 1130, 350),
              L("Pine Bay", 300, 660, 450, 700), L("Cape Bay", 1030, 750, 1180, 790), L("1.33 km", 800, 450, 880, 480)]
    b = group_blocks(labels)
    assert sorted(x.text for x in b) == ["Cape Bay", "East Harbor County", "Northfield", "Pine Bay", "Village"]
    assert all(x.anchor == "center" for x in b)


def test_phrasebook_persists_and_matches_fuzzily(tmp_path):
    p = str(tmp_path / "b.json")
    b = PhraseBook(p)
    b.put("Small store selling clothing, footwear, and accessories at a bargain.", "متجر ملابس قديمة")
    b2 = PhraseBook(p)
    assert b2.get("small store selling clothing,  footwear, and accessories at a bargain.") == "متجر ملابس قديمة"
    assert b2.get("Small store selling clothing, footwear and accessories at a bargain") == "متجر ملابس قديمة"  # OCR dropped punctuation
    assert b2.get("Totally different text here") is None
    assert json.load(open(p, encoding="utf-8"))  # plain, hand-editable JSON


def test_pipeline_translates_the_card_only_while_the_map_is_open(tmp_path):
    seq = [False] * 25 + [True] * 70 + [False] * 25
    events, reads, subs, maps, line_calls = [], [], [], [], []

    class Cap:
        n = 0

        def read(self):
            time.sleep(0.01)
            self.n += 1
            return self.n <= len(seq), frame(seq[min(self.n, len(seq)) - 1])

    class Ocr:
        def read(self, im, **kw):
            reads.append(time.perf_counter())
            return ""

        def lines(self, im):
            line_calls.append(1)  # crop coordinates of the card (area starts at 6% / 13% of the frame)
            return [(t, b, 0.97) for t, b in CARD]

    class Tr:
        calls = 0

        def translate(self, text, history=None):
            Tr.calls += 1
            return "ترجمة " + text

    cfg = config.Config()
    cfg.map = config.MapCfg(enabled=True, check_every=1, poll_ms=20, cache=str(tmp_path / "m.json"), exclude=[])
    run(cfg, Cap(), Ocr(), Tr(), emit=events.append, on_subtitle=lambda *a: subs.append(a), on_map=lambda it: maps.append(it),
        on_state=lambda on: events.append(("state", on)))
    assert [e for e in events if isinstance(e, tuple)] == [("state", True), ("state", False)]
    shown = [m for m in maps if m]
    assert shown and len(shown[0]) == 4 and all(it[0].startswith("ترجمة") and len(it[3]) == 3 for it in shown[0])
    assert maps[-1] == []                              # the card is removed when the map closes
    assert Tr.calls == 4                               # 4 blocks translated once; every later poll came from the saved translations
    assert json.load(open(tmp_path / "m.json", encoding="utf-8"))
    assert any(x == (None, None, None) for x in subs)   # the subtitle overlay is cleared when the map opens
    assert len(line_calls) <= 3                        # the picture never changed, so the card was read once, not on every poll


def test_card_redraw_ignores_pixel_jitter_but_not_movement():
    from translator_overlay.pipeline import _same_card

    a = [("س", (100, 100, 200, 130), 20)]
    assert _same_card([("س", (102, 99, 203, 131), 20)], a)
    assert not _same_card([("س", (160, 100, 260, 130), 20)], a)  # the card moved with the cursor
    assert not _same_card([("ص", (100, 100, 200, 130), 20)], a)
    assert not _same_card(a, None)


def test_a_tag_wider_than_its_box_stays_on_one_line_and_grows_both_ways():
    from translator_overlay.overlay import Overlay

    ov = Overlay(config.OverlayCfg())
    box = (1000, 400, 1070, 430)
    ov.set_map([("سلعة خردة مستعملة", box, 30)])
    f = np.full((1080, 1920, 3), 90, np.uint8)
    bmp, x, y, _ = ov._place_item("سلعة خردة مستعملة", box, 30, 1080, 1920, 1.0, "left", None)
    assert bmp.shape[0] < 60 and x < box[0] and x + bmp.shape[1] > box[2]  # one line, centred on the original box


def test_card_blocks_do_not_overlap_each_other_and_use_the_card_colour():
    from translator_overlay.overlay import Overlay

    ov = Overlay(config.OverlayCfg())
    panel = (60, 40, 36)
    items = [("عنوان", (100, 100, 400, 140), 36, panel),
             ("متجر عتيق يبيع الملابس والأحذية والإكسسوارات بأسعار مخفضة ومناسبة للجميع.", (100, 150, 600, 224), 36, panel),
             ("الأسلوب", (100, 235, 170, 266), 31, panel)]
    ov.set_map(items)
    f = np.full((400, 800, 3), 200, np.uint8)
    ov.draw(f)
    bmp, x, y, rect = ov._place_item(items[1][0], items[1][1], 36, 400, 800, 1.0, "left")
    assert y + bmp.shape[0] <= items[2][1][1] + 2          # the wrapped description ends where the tag row begins
    assert tuple(f[145, 98]) == panel                      # no erased strokes available: the English area falls back to the card's own colour


def test_erase_text_removes_only_the_english_strokes():
    import cv2

    from translator_overlay.mapstate import erase_text

    img = np.zeros((120, 400, 3), np.uint8)
    img[:] = np.linspace(40, 70, 400, dtype=np.uint8)[None, :, None]       # a card with a gradient (translucent over the map)
    img[100:102, :] = (160, 160, 160)                                      # a divider line below the text
    cv2.putText(img, "Small store", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (235, 235, 235), 2, cv2.LINE_AA)
    box = (18, 30, 300, 72)
    patch, (px, py), mask, ink = erase_text(img, [box], box)
    assert ink is not None and min(ink) > 180                              # the English was near-white
    assert patch.max() < 120                                               # no bright text left
    ref = np.zeros_like(patch); ref[:] = np.linspace(40, 70, 400, dtype=np.uint8)[px : px + patch.shape[1]][None, :, None]
    assert np.abs(patch.astype(int) - ref.astype(int)).mean() < 4          # what is left is the gradient card, edge to edge
    assert (img[100:102] == 160).all()                                     # the divider outside the box is untouched


def test_erased_patch_is_pasted_under_the_arabic_with_no_flat_rectangle():
    from translator_overlay.overlay import Overlay

    ov = Overlay(config.OverlayCfg())
    card = np.zeros((200, 600, 3), np.uint8); card[:] = np.linspace(40, 90, 600, dtype=np.uint8)[None, :, None]
    box = (100, 80, 400, 120)
    patch = card[74:126, 94:406].copy()
    ov.set_map([("نص", box, 30, (10, 10, 10), (patch, (94, 74)))])
    f = card.copy(); f[85:115, 110:390] = 250                              # stand-in for the English text
    ov.draw(f)
    assert tuple(f[78, 300]) == tuple(card[78, 300])                       # the margin shows the card's own pixels, not a flat colour


LOCKUP = [("ALEX'S STORAGE", (984, 221, 1197, 264)), ("$ No goods to sell", (940, 281, 1187, 317)),
          ("Alex will pay for any antiques, jewelry", (929, 326, 1422, 364)), ("and coins dropped off at his", (932, 367, 1385, 400)),
          ("storage.", (928, 404, 1028, 443)), ("50 m", (802, 418, 857, 449)), ("ANTIQUES", (971, 454, 1084, 483)), ("VERY LOW", (1328, 454, 1423, 483)),
          ("JEWELRY", (970, 499, 1063, 533)), ("VERY LOW", (1328, 499, 1425, 533)), ("COINS", (971, 548, 1112, 581)), ("VERY LOW", (1328, 547, 1426, 581))]


def test_a_card_with_a_status_line_a_long_paragraph_and_icon_rows():
    """Card with a status line, a long paragraph and icon rows (OCR boxes from a real read)."""
    b = group_blocks(LOCKUP)
    assert [x.text for x in b] == ["ALEX'S STORAGE", "No goods to sell",
                                   "Alex will pay for any antiques, jewelry and coins dropped off at his storage.",
                                   "ANTIQUES", "VERY LOW", "JEWELRY", "VERY LOW", "COINS", "VERY LOW"]
    assert b[1].box[0] > 940 + 15 and b[1].box[2] == 1187  # moved right so the icon is not erased


def test_card_presence_probe_notices_the_card_going_away_but_not_the_map_moving():
    import cv2

    from translator_overlay.mapstate import card_present, erase_text, make_probe

    card = np.full((200, 500, 3), 55, np.uint8)
    cv2.putText(card, "Small store", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (235, 235, 235), 2, cv2.LINE_AA)
    box = (18, 70, 300, 112)
    probe = make_probe(card, erase_text(card, [box], box))
    assert probe is not None
    assert card_present(card, [probe])
    moved = card.copy(); moved[:, :, 0] = np.clip(moved[:, :, 0].astype(int) + 25, 0, 255).astype(np.uint8)  # background shifts under the card
    assert card_present(moved, [probe])
    gone = np.full((200, 500, 3), 40, np.uint8)  # the card is gone, plain map underneath
    assert not card_present(gone, [probe])


def test_arabic_is_taken_away_as_soon_as_the_english_card_disappears():
    import cv2

    def scene(card_on):
        f = np.full((1080, 1920, 3), (60, 40, 30), np.uint8)
        f[int(.031 * 1080):int(.060 * 1080), int(.470 * 1920):int(.500 * 1920)] = PINK_BGR
        if card_on:  # white text at the CARD coordinates (the search area starts at x 115, y 140)
            for t, (x0, y0, x1, y1) in CARD:
                if "685" not in t:  # thin vertical strokes like glyphs: text covers a minority of its box
                    for xx in range(x0 + 115, x1 + 115, 7):
                        cv2.rectangle(f, (xx, y0 + 140 + 4), (xx + 2, y1 + 140 - 4), (240, 240, 240), -1)
        return f

    on, off = scene(True), scene(False)
    seq = [True] * 70 + [False] * 40
    maps, stamps = [], []

    class Cap:
        n = 0

        def read(self):
            time.sleep(0.01)
            self.n += 1
            return self.n <= len(seq), on if seq[min(self.n, len(seq)) - 1] else off

    class Ocr:
        def read(self, im, **kw):
            return ""

        def lines(self, im):  # like a real OCR: nothing to read once the English card is gone
            return [(t, b, 0.97) for t, b in CARD] if (im > 200).any() else []

    class Tr:
        def translate(self, text, history=None):
            return "ترجمة " + text

    cfg = config.Config()
    cfg.map = config.MapCfg(enabled=True, check_every=1, poll_ms=20, cache="", exclude=[])
    t0 = time.perf_counter()
    run(cfg, Cap(), Ocr(), Tr(), on_map=lambda it: (maps.append(it), stamps.append(time.perf_counter() - t0)))
    shown = [i for i, m in enumerate(maps) if m]
    assert shown, "the card was never shown"
    assert maps[-1] == []                             # taken away after the English card disappeared (map still open: tab stays pink)
    assert stamps[-1] - stamps[shown[0]] < 1.5        # and nothing waited for slow OCR confirmation to do it


def test_icon_led_status_line_keeps_its_icon_using_character_boxes():
    """Icon-led status line: the leading '+' is a bonus mark; the icon next to it is not text."""
    def chars(t, x0, y0, x1, y1):
        step = (x1 - x0) / len(t.replace(" ", ""))
        return [(c, (int(x0 + i * step), y0, int(x0 + (i + 1) * step), y1)) for i, c in enumerate(t.replace(" ", ""))]

    status = [("+", (937, 353, 949, 385))] + chars("Sam will appreciate this", 979, 353, 1280, 385)
    lines = [("FIX THE BOAT", (930, 298, 1108, 336), chars("FIX THE BOAT", 938, 298, 1100, 336)),
             ("+Sam will appreciate this", (930, 353, 1280, 385), status),
             ("The old boat has seen better days. Get", (930, 398, 1414, 434), None),
             ("it fixed up and a new coat of paint at", (930, 439, 1400, 472), None),
             ("Pat's repair shop before the weekend.", (929, 475, 1428, 515), None)]
    b = group_blocks(lines)
    assert [x.text for x in b] == ["FIX THE BOAT", "Sam will appreciate this",
                                   "The old boat has seen better days. Get it fixed up and a new coat of paint at Pat's repair shop before the weekend."]
    assert b[1].box[0] >= 975 and b[1].icon            # starts at the first letter, past the icon; anchored (grows right only)
    assert not b[0].icon and not b[2].icon


def test_trailing_icon_inside_the_detection_box_is_not_part_of_the_block():
    """Trailing icon: a tag's detection box (1306-1425) also covers a badge after the word (text ends ~1385)."""
    def chars(t, x0, y0, x1, y1):
        step = (x1 - x0) / len(t)
        return [(c, (int(x0 + i * step), y0, int(x0 + (i + 1) * step), y1)) for i, c in enumerate(t)]

    lines = [("VAPID", (932, 334, 994, 366), chars("VAPID", 936, 334, 990, 366)),
             ("JASON'S '70 GANADO", (930, 363, 1128, 395), chars("JASON'S'70GANADO", 934, 363, 1124, 395)),
             ("SPECIALIST", (927, 421, 1043, 458), chars("SPECIALIST", 937, 421, 1035, 458)),
             ("MUSCLE", (1306, 419, 1425, 459), chars("MUSCLE", 1316, 419, 1385, 459))]
    b = group_blocks(lines)
    assert [x.text for x in b] == ["VAPID", "JASON'S '70 GANADO", "SPECIALIST", "MUSCLE"]  # a brand label and a name: two short lines, two blocks
    muscle = b[3]
    assert muscle.box[2] <= 1385 + 4                  # ends at the last letter: the badge icon (1390-1422) is outside the block
    assert not muscle.icon


def test_paragraph_stays_together_when_detection_boxes_jitter_by_a_few_pixels():
    """Two reads of the same card differed in one box height (33 vs 28 px), which split the paragraph."""
    for second in ((932, 367, 1387, 400), (933, 370, 1385, 398)):
        lines = [("Alex will pay for any antiques, jewelry", (929, 326, 1422, 366)),
                 ("and coins dropped off at his", second), ("storage.", (928, 404, 1029, 443))]
        b = group_blocks(lines)
        assert len(b) == 1 and b[0].text == "Alex will pay for any antiques, jewelry and coins dropped off at his storage."


HOUSE = [("SAM'S HOUSE", (929, 164, 1133, 207)), ("Repair the house", (970, 224, 1190, 257)),
         ("Small house on loan from Alex in return", (930, 268, 1431, 302)), ("for helping with his business.", (932, 309, 1290, 342)),
         ("Fix Steps", (981, 353, 1107, 391)), ("Fix Leaky Roof", (981, 401, 1170, 435)), ("1.33 km", (806, 434, 895, 468)),
         ("Fix Railing", (979, 448, 1115, 489)), ("Fix Wall", (982, 499, 1085, 533)), ("FEATURES", (930, 551, 1029, 581)),
         ("STASH", (929, 597, 999, 632)), ("$2,480", (1315, 597, 1422, 632))]


def test_a_card_with_a_checklist_keeps_every_item_separate_and_ignores_the_blip_label():
    """Card with a checklist: title, icon line, wrapped paragraph, four checklist items, features, stash (real OCR boxes)."""
    b = group_blocks(HOUSE)
    assert [x.text for x in b] == ["SAM'S HOUSE", "Repair the house", "Small house on loan from Alex in return for helping with his business.",
                                   "Fix Steps", "Fix Leaky Roof", "Fix Railing", "Fix Wall", "FEATURES", "STASH"]
    assert all(x.box[0] > 900 for x in b)  # nothing from the label outside the card (it starts at x=806)


def test_text_size_is_the_median_of_its_size_class_not_one_jittery_box():
    b = {x.text: x for x in group_blocks(HOUSE)}
    items = [b[k].line_h for k in ("Fix Steps", "Fix Leaky Roof", "Fix Railing", "Fix Wall")]  # boxes are 38, 34, 41, 34 px tall
    assert max(items) - min(items) == 0                # one size for the whole checklist


def test_anchors_follow_the_layout_not_a_guess():
    b = {x.text: x for x in group_blocks(HOUSE + [("VERY LOW", (1328, 600, 1425, 632))])}
    assert b["Fix Steps"].anchor == "left" and b["STASH"].anchor == "left"
    assert b["VERY LOW"].anchor == "right"            # flush against the card's right margin
    assert b["Small house on loan from Alex in return for helping with his business."].anchor == "left"


def test_right_anchored_block_keeps_its_right_edge():
    from translator_overlay.overlay import Overlay

    ov = Overlay(config.OverlayCfg())
    box = (1328, 600, 1425, 632)
    bmp, x, y, _ = ov._place_item("منخفض جدا", box, 30, 1080, 1920, 1.0, "left", "right")
    assert abs((x + bmp.shape[1]) - (box[2] + 6)) <= 1


def test_hotkey_switch_turns_the_map_state_off_and_on_at_runtime():
    states, seen, frames_n = [], [], [0]
    switch = [False]

    class Cap:
        n = 0

        def read(self):
            time.sleep(0.005)
            self.n += 1
            return self.n <= 120, frame(True)           # the MAP tab is pink the whole time

    class Ocr:
        def read(self, im, **kw):
            return ""

        def lines(self, im):
            return []

    class Tr:
        def translate(self, text, history=None):
            return text

    def on_frame(f):
        frames_n[0] += 1
        if frames_n[0] == 60:
            switch[0] = True  # the user presses m
        seen.append((frames_n[0], switch[0]))

    cfg = config.Config()
    cfg.map = config.MapCfg(enabled=True, check_every=1, cache="", exclude=[])
    run(cfg, Cap(), Ocr(), Tr(), on_state=lambda on: states.append((on, frames_n[0])), on_frame=on_frame, map_switch=switch)
    assert len(states) == 1 and states[0][0] is True and states[0][1] >= 60   # nothing while off; the state turns on only after the switch is flipped
    switch[0] = False
