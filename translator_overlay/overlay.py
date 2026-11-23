import os
import re
import statistics
import zlib
from collections import deque

import cv2

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont, features

FONTS = [os.path.join(os.path.dirname(__file__), "fonts", "IBMPlexSansArabic-SemiBold.ttf"),
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/tahomabd.ttf"]


RAQM = features.check("raqm")  # PIL's own shaper: joins letters + reorders RTL correctly


def shape(text: str) -> str:
    """Without raqm PIL draws isolated LTR glyphs, so join and reorder by hand. With raqm this must not run (double shaping garbles the text)."""
    return text if RAQM else get_display(arabic_reshaper.reshape(text))


class Overlay:
    """Draws the translation onto frames. mode: "bar" (fixed caption) | "replace" (cover the original text box). Text bitmaps are cached per (text, width): one blend per frame."""

    def __init__(self, cfg):
        self.cfg = cfg
        path = cfg.font or next((f for f in FONTS if os.path.exists(f)), None)
        if not path:
            raise RuntimeError("no Arabic-capable font found; set [overlay] font = '/path/to.ttf'")
        self.font_path = path
        self.current = None  # (arabic_text, box|None, line_h, speaker|None), set by the worker thread, read per frame
        self.map_align, self.map_size_scale = "left", 1.0  # set from [map]
        self._erased = {}
        self.map_items = []  # map state: [(arabic_text, box, line_h, ...)], one per block of the card
        self._cache = {}
        self._heights = deque(maxlen=15)  # text size is constant; single OCR boxes jitter
        self.on_change = None  # called from any thread when the subtitle or the card changed: lets the display redraw at once

    def set(self, text, box=None, line_h=None, speaker=None):
        if line_h:
            self._heights.append(line_h)
            line_h = statistics.median(self._heights)
        if isinstance(text, list):  # several speakers at once: [(speaker, text), ...]
            text = tuple((sp if self.cfg.speaker else None, t) for sp, t in text)
        new, old = (text, box, line_h, speaker if self.cfg.speaker else None) if text else None, self.current
        self.current = new
        if new != old:  # a repeated clear (no subtitle, still none) changes nothing on screen
            self.on_change and self.on_change()

    def set_map(self, items):
        self.map_items = list(items or [])
        self._erased = {}  # (item index, scale) -> the erased card patch scaled to the screen
        self.on_change and self.on_change()

    def has_content(self) -> bool:
        return self.current is not None or bool(self.map_items)

    def speaker_color(self, name):
        pal = self.cfg.speaker_palette
        return tuple(pal[zlib.crc32(name.lower().encode()) % len(pal)])

    def _bitmap(self, text, max_w, size, min_w=0, min_h=0, opaque=False, align="center", plain=False, pad=None, color=None, speaker=None):
        """text: a string (optionally with one `speaker`), or several (speaker|None, text) segments = several people talking at once, each starting its own line."""
        segs = tuple(tuple(x) for x in text) if isinstance(text, (list, tuple)) else ((speaker, text),)
        key = (segs, max_w, size, min_w, min_h, opaque, align, plain, pad, color)
        if key not in self._cache:
            font, pad = ImageFont.truetype(self.font_path, size), self.cfg.padding if pad is None else pad
            sp = font.getlength(" ")

            def label_of(name):  # (name, shaped label, is Arabic, width): drawn in its own colour at the start (right end) of the segment's first line
                lab = f"{name}:"
                ar = bool(re.search("[\u0600-\u06ff]", lab))  # an Arabic name flows right to left, an English one is drawn left to right
                ls = shape(lab) if ar else lab
                return name, ls, ar, font.getlength(ls)

            def width(ws, lab):  # width of a line of logical words, plus the label when it is the segment's first line
                w = font.getlength(shape(" ".join(ws))) if ws else 0
                return w + (lab[3] + (sp if ws else 0) if lab else 0)

            lines = []  # (words, label | None)
            for name, body in segs:
                lab, line = (label_of(name) if name else None), []
                first = True
                for w in body.split():  # wrap the logical text, shape each finished line afterwards
                    if line and width(line + [w], lab if first else None) > max_w - 2 * pad:
                        lines.append((line, lab if first else None))
                        first, line = False, [w]
                    else:
                        line = line + [w]
                lines.append((line, lab if first else None))
            shaped = [shape(" ".join(l)) for l, _ in lines]
            lh = int(size * 1.35)
            wpx = max(int(max(width(l, lab) for l, lab in lines)) + 2 * pad, min_w)
            hpx = max(lh * len(shaped) + pad, min_h)
            img = Image.new("RGBA", (wpx, hpx), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            if not self.cfg.blur and not plain:  # blur mode: the backdrop is drawn in draw(), which needs the live frame
                d.rectangle([0, 0, *img.size], fill=(*self.cfg.bg, 255 if opaque else int(255 * self.cfg.bg_opacity)))
            top = (hpx - lh * len(shaped)) // 2
            for i, (l, (words, lab)) in enumerate(zip(shaped, lines)):
                lw = width(words, lab)
                lx = pad if align == "left" else wpx - pad - lw if align == "right" else (wpx - lw) / 2
                d.text((lx, top + i * lh), l, font=font, fill=(*(color or self.cfg.color), 255),
                       **({"direction": "rtl", "language": "ar"} if RAQM else {}))
                if lab:  # at the right end, followed (to its left) by the Arabic
                    d.text((lx + lw - lab[3], top + i * lh), lab[1], font=font, fill=(*self.speaker_color(lab[0]), 255),
                           **(({"direction": "rtl", "language": "ar"} if lab[2] else {"direction": "ltr", "language": "en"}) if RAQM else {}))
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = np.asarray(img)
        return self._cache[key]

    def _place(self, h: int, w: int, scale: float = 1.0):
        """(RGBA text layer, x, y) for the current subtitle in an h x w frame, or None. `scale` = frame size relative to the frame the OCR box / line height were measured in
        (video is scaled to the screen first, then text is drawn at screen resolution)."""
        cur = self.current
        if not cur:
            return None
        text, box, line_h, speaker = cur
        if scale != 1.0:
            box = tuple(int(v * scale) for v in box) if box else box
            line_h = line_h * scale if line_h else line_h
        size = max(12, int(line_h * self.cfg.size_scale)) if (self.cfg.match_size and line_h) else max(14, int(h * self.cfg.size))
        if self.cfg.mode == "replace" and box:
            m = 10  # margin covers the original's anti-aliased edges
            x0, y0, x1, y1 = max(0, box[0] - m), max(0, box[1] - m), min(w, box[2] + m), min(h, box[3] + m)
            bmp = self._bitmap(text, max(x1 - x0, int(w * 0.5)), size, x1 - x0, y1 - y0, opaque=True, speaker=speaker)
            x, y = max(0, min(w - bmp.shape[1], (x0 + x1 - bmp.shape[1]) // 2)), y0
        else:
            bmp = self._bitmap(text, int(w * 0.8), size, speaker=speaker)
            x = (w - bmp.shape[1]) // 2
            y = int(h * (self.cfg.bar_y)) - bmp.shape[0] if self.cfg.bar_y > 0.5 else int(h * self.cfg.bar_y)
        return bmp, x, max(0, min(h - bmp.shape[0], y))

    def _place_item(self, text, box, line_h, h, w, scale, align, anchor="left", ink=None):
        """Like _place for one map-card block: the Arabic goes where the English was (same box, `align` inside it).
        Returns (text-only bitmap, x, y, (x0, y0, x1, y1) = area to paint first). The font starts at the English size and shrinks until
        wrapped text fits the English box height."""
        if scale != 1.0:
            box, line_h = tuple(int(v * scale) for v in box), line_h * scale
        m = 6
        x0, y0, x1, y1 = max(0, box[0] - m), max(0, box[1] - m), min(w, box[2] + m), min(h, box[3] + m)
        one_line = (box[3] - box[1]) <= 1.6 * line_h  # a title / tag / list item: never wrap it
        size = max(12, int(line_h * self.map_size_scale * (1.25 if anchor == "center" else 1.0)))  # map labels are large display text
        eff = "right" if (anchor == "right" and one_line) else "center" if anchor == "center" else align  # right-anchored: at the block's right edge; labels: centred
        for _ in range(8):
            bmp = self._bitmap(text, int(w * 0.6) if one_line else x1 - x0, size, x1 - x0, y1 - y0, align=eff, plain=True, pad=3, color=(ink[2], ink[1], ink[0]) if ink else None)
            if one_line or bmp.shape[0] <= (y1 - y0) + 8 or size <= 12:
                break
            size = max(12, int(size * 0.92))
        over = bmp.shape[1] - (x1 - x0)
        if anchor == "center":
            x = (x0 + x1 - bmp.shape[1]) // 2  # a map label: centred on the English one
        elif one_line and anchor == "right":
            x = x1 - bmp.shape[1]  # keep the English block's right edge: an icon or the card's margin lies beyond it
        elif over > 0 and one_line and anchor is None:
            x = x0 - over // 2     # nothing known on either side: grow both ways
        else:
            x = x0 if align == "left" else x1 - bmp.shape[1] if align == "right" else (x0 + x1 - bmp.shape[1]) // 2
        x, y = max(0, min(w - bmp.shape[1], x)), max(0, min(h - bmp.shape[0], y0))
        rect = (x0, y0, x1, y1)  # only where the English was; Arabic that overflows the box lands on empty card
        return bmp, x, y, rect

    def _blur(self, roi: np.ndarray) -> np.ndarray:
        """Blurred, dimmed copy of the pixels behind the subtitle: erode (removes thin bright strokes = the original text), blur at 1/4 resolution, dim."""
        bh, bw = roi.shape[:2]
        sm = cv2.resize(roi, (max(1, bw // 4), max(1, bh // 4)), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(cv2.erode(sm, np.ones((3, 3), np.uint8)), (0, 0), self.cfg.blur_sigma / 4)
        return (cv2.resize(sm, (bw, bh), interpolation=cv2.INTER_LINEAR) * (1 - self.cfg.bg_opacity)).astype(np.uint8)

    def _blend(self, frame, bmp, x, y, bg=None):
        bh, bw_ = bmp.shape[:2]
        roi = frame[y : y + bh, x : x + bw_]
        if bg is None and self.cfg.blur:
            roi[:] = self._blur(roi)
        a = bmp[..., 3:4].astype(np.uint16)
        roi[:] = ((bmp[..., 2::-1].astype(np.uint16) * a + roi.astype(np.uint16) * (255 - a)) // 255).astype(np.uint8)

    def draw(self, frame: np.ndarray, scale: float = 1.0, map_align: str | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        cards = []
        for i, (text, box, line_h, *rest) in enumerate(self.map_items):  # map state: every block of the card, each over its own English text
            cards.append((*self._place_item(text, box, line_h or 20, h, w, scale, map_align or self.map_align, rest[2] if len(rest) > 2 else "left", rest[3] if len(rest) > 3 else None), rest[0] if rest else None,
                          rest[1] if len(rest) > 1 else None, i))
        for _, _, _, r, bg, er, i in cards:  # pass 1: remove the English (card colour under any overflow, then the erased pixels)
            if bg is not None and er is None:  # fallback: with the strokes erased nothing needs hiding, and a flat fill shows as a patch
                frame[r[1] : r[3], r[0] : r[2]] = bg
            if er is not None:
                key = (i, scale)
                if key not in self._erased:
                    pa, (px, py), *mk = er
                    sc = (lambda a, interp: a if scale == 1.0 else cv2.resize(a, (max(1, round(a.shape[1] * scale)), max(1, round(a.shape[0] * scale))), interpolation=interp))
                    self._erased[key] = (sc(pa, cv2.INTER_CUBIC), (round(px * scale), round(py * scale)),
                                         (sc(mk[0], cv2.INTER_LINEAR) > 127) if mk else None)
                pa, (px, py), mask = self._erased[key]
                ph, pw = min(pa.shape[0], h - py), min(pa.shape[1], w - px)
                if ph > 0 and pw > 0 and px >= 0 and py >= 0:
                    roi = frame[py : py + ph, px : px + pw]
                    if mask is None:
                        roi[:] = pa[:ph, :pw]
                    else:  # replace only the English strokes; the rest stays live
                        roi[mask[:ph, :pw]] = pa[:ph, :pw][mask[:ph, :pw]]
        for bmp, x, y, _, bg, _, _ in cards:  # pass 2: draw all Arabic, so no block's backdrop clips another block's text
            self._blend(frame, bmp, x, y, bg)
        placed = self._place(h, w, scale)
        if placed:
            self._blend(frame, *placed)
        return frame

    def patch(self, frame: np.ndarray, scale: float = 1.0):
        """(premultiplied BGRA patch, x, y) = the pixels draw() would paint, for a separate click-through window. Reads `frame` without modifying it; None when there is no subtitle."""
        placed = self._place(*frame.shape[:2], scale)
        if not placed:
            return None
        bmp, x, y = placed
        bh, bw_ = bmp.shape[:2]
        a = bmp[..., 3:4].astype(np.uint16)
        rgb = bmp[..., 2::-1].astype(np.uint16)
        if self.cfg.blur:  # blurred backdrop is opaque, so the whole patch is opaque
            back = self._blur(frame[y : y + bh, x : x + bw_]).astype(np.uint16)
            bgr = ((rgb * a + back * (255 - a)) // 255).astype(np.uint8)
            alpha = np.full((bh, bw_, 1), 255, np.uint8)
        else:  # translucent solid box: premultiply colour by alpha (UpdateLayeredWindow)
            bgr = (rgb * a // 255).astype(np.uint8)
            alpha = bmp[..., 3:4]
        return np.ascontiguousarray(np.concatenate([bgr, alpha], axis=2)), x, y
