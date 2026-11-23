"""Map state: detect that the MAP tab is open (pink pixels, no OCR), find the text blocks of the hovered card or the map labels,
and keep translations on disk."""
import difflib
import json
import os
import re
import statistics

import cv2
import numpy as np


class MapDetector:
    """True while the MAP tab word is pink: counts pink pixels in a small box (~20 us), re-tested every `check_every` frames, with hysteresis against single odd frames."""

    def __init__(self, cfg):
        self.cfg, self.state, self._n, self._hits, self._misses = cfg, False, 0, 0, 0

    def pink_fraction(self, frame: np.ndarray) -> float:
        c = self.cfg
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = c.tab_box
        patch = frame[int(y0 * h) : max(int(y0 * h) + 1, int(y1 * h)), int(x0 * w) : max(int(x0 * w) + 1, int(x1 * w))]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        m = (hsv[..., 0] >= c.hue[0]) & (hsv[..., 0] <= c.hue[1]) & (hsv[..., 1] >= c.sat_min) & (hsv[..., 2] >= c.val_min)
        return float(m.mean())

    def reset(self):
        self.state, self._n, self._hits, self._misses = False, 0, 0, 0

    def update(self, frame: np.ndarray) -> bool:
        self._n += 1
        if self._n % self.cfg.check_every:
            return self.state
        if self.pink_fraction(frame) >= self.cfg.min_frac:
            self._hits, self._misses = self._hits + 1, 0
            if self._hits >= self.cfg.on_checks:
                self.state = True
        else:
            self._misses, self._hits = self._misses + 1, 0
            if self._misses >= self.cfg.off_checks:
                self.state = False
        return self.state


def _caps(t: str) -> bool:
    return bool(re.search("[A-Z]", t)) and not re.search("[a-z]", t)


def fix_case(text: str) -> str:
    """ALL-CAPS UI text is read as an abbreviation or name and copied back by the translator; sentence-style capitals translate."""
    return " ".join(w.capitalize() for w in text.split()) if _caps(text) else text


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def _lum(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)


def make_probe(img: np.ndarray, e):
    """From an erase_text() result and its source frame: the signature of 'the English text is still on screen'.
    (x, y, stroke mask, reference contrast), or None when there is nothing measurable."""
    if e is None:
        return None
    patch, (x, y), mask = e[:3]
    g = _lum(img[y : y + mask.shape[0], x : x + mask.shape[1]])
    m = mask > 0
    if g.shape != m.shape or m.sum() < 20:
        return None
    contrast = abs(float(g[m].mean()) - float(np.median(g)))
    return (x, y, m, contrast) if contrast >= 15 else None


def card_present(frame: np.ndarray, probes, frac: float = 0.5) -> bool:
    """True if at least half of the blocks still show their strokes: the live pixels under each stroke mask keep at least half of their original contrast.
    Sub-millisecond per frame; background motion under a stationary card does not trigger it, the card disappearing does."""
    if not probes:
        return True
    ok = 0
    for x, y, m, ref in probes:
        g = _lum(frame[y : y + m.shape[0], x : x + m.shape[1]])
        if g.shape == m.shape and abs(float(g[m].mean()) - float(np.median(g))) >= frac * ref:
            ok += 1
    return ok >= max(1, int(np.ceil(len(probes) * frac)))


def panel_color(img: np.ndarray, box, m: int = 8):
    """BGR colour behind a block of text: the median of the pixels in and around the box (text covers well under half of them)."""
    h, w = img.shape[:2]
    roi = img[max(0, box[1] - m) : min(h, box[3] + m), max(0, box[0] - m) : min(w, box[2] + m)].reshape(-1, 3)
    if not len(roi):
        return (40, 40, 40)
    return tuple(int(v) for v in np.median(roi, 0))


def erase_text(img: np.ndarray, line_boxes, box, m: int = 6):
    """The region around a block with only the English strokes removed (inpainted), so the translucent panel, dividers and icons stay intact.
    Text = pixels that differ clearly from the box median (light-on-dark or dark-on-light).
    Returns (patch, (x, y), mask, ink) in `img` coordinates (mask = text pixels, ink = their BGR colour), or None. Once per card."""
    h, w = img.shape[:2]
    mx = max(m, int(0.15 * (box[3] - box[1])) + 1)
    x0, y0, x1, y1 = max(0, box[0] - mx), max(0, box[1] - m), min(w, box[2] + mx), min(h, box[3] + m)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = img[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.int16)
    mask = np.zeros(gray.shape, np.uint8)
    for b in line_boxes:
        pad = max(3, int(0.15 * (b[3] - b[1])))  # large display text has glyphs reaching past the character boxes
        bx0, by0, bx1, by1 = max(0, b[0] - pad - x0), max(0, b[1] - 3 - y0), min(x1 - x0, b[2] + pad - x0), min(y1 - y0, b[3] + 3 - y0)
        if bx1 > bx0 and by1 > by0:
            sub = gray[by0:by1, bx0:bx1]
            mask[by0:by1, bx0:bx1] = (np.abs(sub - np.median(sub)) > 40).astype(np.uint8) * 255
    ink = None
    ys, xs = np.nonzero(mask)
    if len(ys) >= 15:  # the text's own colour: the strokes furthest from the background
        d = np.abs(gray[ys, xs] - np.median(gray))
        keep = d >= np.percentile(d, 60)
        ink = tuple(int(v) for v in np.median(roi[ys[keep], xs[keep]], 0))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)  # covers antialiasing / shadow around the strokes
    return cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA), (x0, y0), mask, ink


class Block:
    def __init__(self, text, box, line_h, caps, lines=None, icon=False):
        self.text, self.box, self.line_h, self.caps, self.lines, self.icon = text, box, line_h, caps, lines or [box], icon
        self.anchor = "left"  # edge that stays put when the Arabic is wider than the English: "left" grows right, "right" grows left, None both

    def __repr__(self):
        return f"Block({self.text!r}, {self.box}, h={self.line_h:.0f})"


def group_blocks(lines, min_letters=2):
    """lines: [(text, (x0,y0,x1,y1))] -> the text blocks of the map screen. The largest group of nearby lines is the card (>= 3 lines);
    other groups are scattered labels. A block is a paragraph (consecutive lines of similar height, same left edge and capitalisation, merged into one sentence)
    or a single word on a row with a wide gap (tags)."""
    L, solo, edge = [], set(), {}  # edge[box] = (icon at its left end, icon at its right end)
    for ln in lines:
        t, b, chars = ln[0].strip(), ln[1], (ln[2] if len(ln) > 2 else None)
        if len(re.findall("[A-Za-z]", t)) < min_letters:
            continue
        if chars:
            # Icons and bullets sit in the line's box but are not text: start the block at the first letter/digit and end it at the last one,
            # so the icon is neither erased nor covered.
            ok = [i for i, (c, _) in enumerate(chars) if c.isalnum()]
            if ok:
                first, last = chars[ok[0]][1], chars[ok[-1]][1]
                lead = bool(ok[0]) or first[0] - b[0] > 0.6 * (b[3] - b[1])
                end = ok[-1]  # keep punctuation after the last letter, not a stray symbol
                while end + 1 < len(chars) and chars[end + 1][0] in ".,!?:;'\")%…":
                    end += 1
                # The detection box can swallow an icon at either end. The block is only what the characters cover,
                # so the icon is neither erased nor covered.
                nb = (max(b[0], first[0] - 3), b[1], min(b[2], chars[end][1][2] + 3), b[3])
                edge[nb] = (lead, b[2] - nb[2] > 8)  # the detection box runs past the last letter: an icon sits there
                if lead:
                    solo.add(nb)  # an icon-led line is a status line of its own, never part of the paragraph below
                b = nb
                t = re.sub(r"^[^A-Za-z0-9]+", "", t) if lead else t
        else:
            m_ = re.match(r"^([^A-Za-z0-9\s]{1,2})\s+(\S.*)$", t)
            if m_:  # no character boxes (tests): estimate the text start from the string
                b = (b[0] + round((b[2] - b[0]) * (len(t) - len(m_.group(2))) / len(t)), b[1], b[2], b[3])
                t = m_.group(2)
                solo.add(b)
        L.append((t, b))
    if not L:
        return []
    h = statistics.median(b[3] - b[1] for _, b in L)
    n = len(L)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = L[i][1], L[j][1]
            vgap = max(b[1] - a[3], a[1] - b[3])
            hgap = max(b[0] - a[2], a[0] - b[2])
            same_row = vgap < 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0.5 * min(a[3] - a[1], b[3] - b[1])
            stacked = vgap <= 1.2 * h and (abs(a[0] - b[0]) <= 2 * h or min(a[2], b[2]) - max(a[0], b[0]) > 0.2 * min(a[2] - a[0], b[2] - b[0]))
            if (same_row and hgap <= 30 * h) or stacked:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(L[i])
    def build(card, labels):

        # Left text margin = the left edge shared by several lines. Text well left of it is outside the card (e.g. a marker's distance label).
        x0s = sorted(b[0] for _, b in card)
        margin = next((x for x in x0s if sum(abs(y - x) <= 12 for y in x0s) >= 3), x0s[0])
        if len(card) >= 4:
            card = [(t, b) for t, b in card if b[0] >= margin - 25]
        card.sort(key=lambda tb: (tb[1][1] + tb[1][3]) / 2)
        rows = []
        for t, b in card:  # same row = vertical centres within 60% of a line
            cy, lh = (b[1] + b[3]) / 2, b[3] - b[1]
            if rows and abs(cy - rows[-1]["cy"]) < 0.6 * max(lh, rows[-1]["lh"]):
                rows[-1]["items"].append((t, b))
            else:
                rows.append({"cy": cy, "lh": lh, "items": [(t, b)]})
        # Wrapped text fills the column: a line continues into the next only if it reached (nearly) the right margin;
        # a short line ends its item however tightly the next one is stacked.
        right = max((r["items"][0][1][2] for r in rows if len(r["items"]) == 1), default=0)
        blocks, para = [], None

        def flush():
            nonlocal para
            if para:
                tx = " ".join(t for t, _ in para["lines"])
                bx = (min(b[0] for _, b in para["lines"]), min(b[1] for _, b in para["lines"]),
                      max(b[2] for _, b in para["lines"]), max(b[3] for _, b in para["lines"]))
                blocks.append(Block(tx, bx, statistics.median(b[3] - b[1] for _, b in para["lines"]), _caps(tx), [b for _, b in para["lines"]]))
            para = None

        for r in rows:
            items = sorted(r["items"], key=lambda tb: tb[1][0])
            gaps = [items[k + 1][1][0] - items[k][1][2] for k in range(len(items) - 1)]
            if gaps and min(gaps) > 3 * h:  # row with a wide gap: every word is its own block
                flush()
                for t, b in items:
                    blocks.append(Block(t, b, b[3] - b[1], _caps(t)))
                continue
            if len(items) == 1 and items[0][1] in solo:  # an icon-led line never merges into the paragraph around it
                flush()
                blocks.append(Block(items[0][0], items[0][1], items[0][1][3] - items[0][1][1], _caps(items[0][0]), icon=True))
                continue
            t = " ".join(t for t, _ in items)
            b = (min(x[1][0] for x in items), min(x[1][1] for x in items), max(x[1][2] for x in items), max(x[1][3] for x in items))
            lh = b[3] - b[1]
            if para:
                last = para["lines"][-1][1]
                # box height depends on ascenders/descenders and jitters a few px between reads: only a clearly different size breaks a paragraph
                same = (_caps(t) == para["caps"] and 0.6 <= lh / max(1, last[3] - last[1]) <= 1.6
                        and b[1] - last[3] <= 0.9 * lh and abs(b[0] - last[0]) <= 0.4 * lh
                        and last[2] - margin >= 0.65 * (right - margin))
                if not same:
                    flush()
            if para:
                para["lines"].append((t, b))
            else:
                para = {"lines": [(t, b)], "caps": _caps(t)}
        flush()
        span = max(1, right - margin)
        for blk in blocks:
            if len(blk.lines) > 1:
                continue  # a paragraph stays left-anchored
            lead, tail = edge.get(blk.box, (False, False))
            if lead and tail:
                blk.anchor = None
            elif tail:
                blk.anchor = "right"
            elif lead:
                blk.anchor = "left"
            elif blk.box[2] >= right - 0.12 * span and blk.box[0] > margin + 0.35 * span:
                blk.anchor = "right"  # flush against the right margin: keep that edge
        # Text size per run: one box height jitters ~20% with the letters in it, so a block's size is the median over its run
        # (consecutive blocks with the same left edge and capitalisation).
        # A block standing alone keeps its own size.
        run = []

        def settle():
            med = statistics.median(x.line_h for x in run) if run else 0
            for x in run:
                x.line_h = med

        for blk in blocks:
            if run and abs(blk.box[0] - run[0].box[0]) <= 10 and blk.caps == run[0].caps and 0.7 <= blk.line_h / run[-1].line_h <= 1.4:
                run.append(blk)
            else:
                settle()
                run = [blk]
        settle()
        if labels:  # scattered labels: centre the Arabic on the English instead of anchoring it
            for blk in blocks:
                blk.anchor = "center"
        return blocks

    letters = lambda g: sum(len(re.findall("[A-Za-z]", t)) for t, _ in g)
    ranked = sorted(groups.values(), key=letters, reverse=True)
    card_mode = len(ranked[0]) >= 3  # a card has a title and at least two more lines; otherwise everything is a scattered label
    out = []
    for gi, g in enumerate(ranked):
        labels = not (card_mode and gi == 0)
        if labels and (len(g) == 1 and (letters(g) < 3 or re.fullmatch(r"[\d.,\s]*(km|mi|m|ft|yd)\.?", g[0][0].strip().lower()))):
            continue  # a marker's distance label ("1.3 km", "50 m"), not a place name
        out.extend(build(g, labels))
    return out


class PhraseBook:
    """Saved translations (JSON: English -> Arabic). Exact match after normalising, then fuzzy (OCR reads the same text slightly differently).
    The file can be edited to pin wording."""

    def __init__(self, path: str = ""):
        self.path, self.data = path, {}
        if path and os.path.exists(path):
            try:
                self.data = {norm(k): v for k, v in json.load(open(path, encoding="utf-8")).items()}
            except (OSError, ValueError):
                self.data = {}

    def get(self, text: str):
        k = norm(text)
        if k in self.data:
            return self.data[k]
        best, hit = 0.93, None
        for kk, v in self.data.items():
            if abs(len(kk) - len(k)) <= 0.15 * len(k) + 2:
                sm = difflib.SequenceMatcher(None, k, kk)
                if sm.real_quick_ratio() >= best and sm.quick_ratio() >= best and sm.ratio() >= best:
                    best, hit = sm.ratio(), v
        return hit

    def put(self, text: str, ar: str):
        self.data[norm(text)] = ar
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
