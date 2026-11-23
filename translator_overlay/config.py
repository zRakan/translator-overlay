import tomllib
from dataclasses import dataclass, field, fields


@dataclass
class Source:
    device: str = "0"


@dataclass
class Region:
    x: float = 0.15
    y: float = 0.78
    w: float = 0.70
    h: float = 0.15


@dataclass
class Detect:
    threshold: float = 1.5
    stable_frames: int = 3
    fast: bool = False        # OCR on the first changed frame (then again once settled): ~2-3 frames sooner
    skip_caps: bool = False   # drop ALL-CAPS lines (signs/posters); off for all-caps subtitle games
    poll_ms: int = 0          # >0: while the background keeps moving, OCR at least this often anyway (else subtitles over camera motion are never read)
    center_tol: float = 0.0   # >0: keep only text centred within 0.5±tol of frame width (0 = off)


@dataclass
class Translate:
    backend: str = "echo"
    source: str = "en"
    target: str = "ar"
    url: str = "http://127.0.0.1:5000/translate"
    model: str = "Helsinki-NLP/opus-mt-en-ar"
    cache_size: int = 2048
    chat_url: str = "http://127.0.0.1:8080/v1/chat/completions"  # chat-completions endpoint
    chat_style: str = "hymt"  # chat prompt style: hymt (instruction prompt) | plamo (chat_template_kwargs) | plamo-pfml (completions prompt)
    api_model: str = ""        # chat backend: model name for servers that require one
    api_key: str = ""          # chat backend: bearer token for cloud APIs
    temperature: float = 0.7  # chat backend sampling temperature
    target_tag: str = ""  # local backend: multilingual models need a target-language tag before the source, e.g. ">>ara<<"
    beams: int = 1             # local backend: 1 = greedy (fastest); 3-4 = better quality, slower
    context: bool = False  # pass the previous line(s) while they belong to one conversation (sentences split across subtitles; plamo-pfml style only)
    context_lines: int = 2       # how many previous lines to include
    context_when: str = "continuation"  # continuation = only for a line that continues an unfinished sentence; always = every line of a conversation
    context_gap_s: float = 5.0   # a previous line counts as the same conversation if it was on screen this recently
    strip_speaker: bool = False  # drop 'Name:' labels before translating (they get translated as content)


@dataclass
class OverlayCfg:
    mode: str = "bar"          # bar | replace
    font: str = ""             # path to an Arabic-capable .ttf; empty = auto-detect
    size: float = 0.05         # fallback text size (fraction of frame height) when the game's isn't known
    match_size: bool = True  # size the Arabic from the OCR'd text line height
    size_scale: float = 0.82   # Arabic font px per px of game line height
    color: tuple = (255, 255, 255)
    bg: tuple = (0, 0, 0)
    bg_opacity: float = 0.55   # blur mode: how much the blurred backdrop is darkened; solid mode: alpha
    blur: bool = True          # blurred frame backdrop instead of a solid box
    blur_sigma: float = 12.0   # backdrop blur radius in px (full resolution)
    padding: int = 14
    bar_y: float = 0.95        # bar mode: bottom edge of the bar as a fraction of frame height
    speaker: bool = True       # put the 'NAME:' label (removed before translating, needs [translate] strip_speaker) back in front of the Arabic, in a colour
    speaker_arabic: bool = True  # write known names in Arabic (data/speaker_names_ar.tsv); unknown names stay in English
    speaker_palette: list = field(default_factory=lambda: [(255, 214, 102), (120, 200, 255), (140, 230, 140), (255, 150, 150),
                                                           (200, 160, 255), (255, 170, 80), (100, 230, 220), (255, 130, 210)])  # each name gets a stable colour from here (RGB)


@dataclass
class DisplayCfg:
    scale: str = "sharp"     # --show: how the video is scaled to the screen: off (let the window stretch it) | cubic | sharp (cubic + light sharpening)
    fps: float = 0         # --show: cap the window at this many frames/s (0 = show every captured frame). 30 for a 30 fps game on a 60 fps card
    sharpen: float = 0.5     # sharp mode: unsharp-mask strength (0.3 subtle ... 1.0 strong)


@dataclass
class MapCfg:
    """Map state: while the MAP tab is open, translate the hovered card and the map labels instead of the subtitles."""
    enabled: bool = False
    # MAP tab word box (frame fractions; 1080p: x 911-952, y 38-60 + 0.4% padding). Pink when selected, white otherwise:
    # map open = enough pink pixels in the box (no OCR).
    tab_box: list = field(default_factory=lambda: [0.470, 0.031, 0.500, 0.060])
    hue: list = field(default_factory=lambda: [145, 179])  # OpenCV hue range of the pink (median 168)
    sat_min: int = 70  # median 109; white text ~4
    val_min: int = 150
    min_frac: float = 0.08  # fraction of the box that must be pink (~0.15)
    check_every: int = 3                                    # test the tab every N frames
    on_checks: int = 2                                      # consecutive positive checks before the state turns ON
    off_checks: int = 3                                     # consecutive negative checks before it turns OFF
    poll_ms: int = 250                                      # while ON: look for the blip card this often
    area: list = field(default_factory=lambda: [0.06, 0.13, 0.94, 0.84])  # x, y, w, h searched for the card
    exclude: list = field(default_factory=lambda: [[0.0, 0.0, 0.10, 1.0], [0.78, 0.62, 0.22, 0.38], [0.80, 0.0, 0.20, 0.20]])  # x,y,w,h of fixed UI (left menu, button hints, money)
    min_score: float = 0.85                                 # OCR confidence for card lines
    size_scale: float = 0.8                                 # Arabic font px per px of the English line height on the card
    align: str = "left"                                     # where the Arabic sits in the original text's box: left | right | center
    debug: bool = False                                     # save the card crop + every OCR line (with score) to map_debug/ on each new card
    cache: str = "map_translations.json"                    # saved translations; edit this file to pin your own wording


@dataclass
class AudioCfg:
    """Capture-card sound -> speakers."""
    enabled: bool = False
    input: str = ""  # substring of the capture device name; "" = system default input
    output: str = ""  # substring of the output device name; "" = system default output
    delay_ms: float = 0  # delay the sound to line it up with the picture
    volume: float = 1.0


@dataclass
class OcrCfg:
    rec: str = "v5-en"  # v5-en (English-only, default) | v6 (multilingual)
    engine: str = "auto"      # auto (TensorRT -> CUDA -> CPU) | trt | cuda | cpu
    cache_dir: str = ""       # TensorRT engine cache; empty = ~/.cache/translator-overlay/trt
    min_score: float = 0.0    # ignore reads whose weakest line scores below this (0 = off; ~0.9 drops garbled reads during motion)


@dataclass
class RecordCfg:
    """--record: where recordings go and which ffmpeg writes them."""
    dir: str = "recordings"  # --record auto writes <dir>/rec-<time>.mkv
    ffmpeg: str = ""         # path to ffmpeg; "" = the one on the PATH


@dataclass
class DebugCfg:
    """Settings of the --debug log (the log itself is only written when the --debug flag is given)."""
    dir: str = "debug"       # trace-<time>.jsonl, crops/ and marks/ go here
    crops: bool = True       # save the OCR'd subtitle crop of every translation (to see what the OCR actually saw)


@dataclass
class Config:
    source: Source = field(default_factory=Source)
    region: Region = field(default_factory=Region)
    detect: Detect = field(default_factory=Detect)
    translate: Translate = field(default_factory=Translate)
    overlay: OverlayCfg = field(default_factory=OverlayCfg)
    ocr: OcrCfg = field(default_factory=OcrCfg)
    display: DisplayCfg = field(default_factory=DisplayCfg)
    map: MapCfg = field(default_factory=MapCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    debug: DebugCfg = field(default_factory=DebugCfg)
    record: RecordCfg = field(default_factory=RecordCfg)


def load(path: str | None) -> Config:
    cfg = Config()
    if not path:
        return cfg
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for sec in fields(cfg):
        # unknown keys raise TypeError, so config typos fail loudly
        setattr(cfg, sec.name, type(getattr(cfg, sec.name))(**raw.get(sec.name, {})))
    cfg.overlay.color, cfg.overlay.bg = tuple(cfg.overlay.color), tuple(cfg.overlay.bg)
    cfg.overlay.speaker_palette = [tuple(c) for c in cfg.overlay.speaker_palette]
    return cfg


def clamp_region(r: "Region") -> None:
    """Keep the subtitle region inside the frame and at least 2% x 2% (in place)."""
    r.w, r.h = min(1.0, max(0.02, r.w)), min(1.0, max(0.02, r.h))
    r.x, r.y = min(1.0 - r.w, max(0.0, r.x)), min(1.0 - r.h, max(0.0, r.y))


def save_region(path: str, r: "Region") -> None:
    """Write x/y/w/h into the [region] table of the TOML file at `path`, keeping every other line and comment."""
    import re

    text = open(path, encoding="utf-8").read()
    vals = {"x": r.x, "y": r.y, "w": r.w, "h": r.h}
    m = re.search(r"(?ms)^\[region\]\s*$(.*?)(?=^\[|\Z)", text)
    if not m:
        text = text.rstrip("\n") + "\n\n[region]\n" + "".join(f"{k} = {v:.4f}\n" for k, v in vals.items())
    else:
        body = m.group(1)
        for k, v in vals.items():
            body, n = re.subn(rf"(?m)^{k}\s*=\s*[-0-9.eE+]+", f"{k} = {v:.4f}", body)
            if not n:
                body = body.rstrip("\n") + f"\n{k} = {v:.4f}\n"
        text = text[: m.start(1)] + body + text[m.end(1) :]
    open(path, "w", encoding="utf-8", newline="\n").write(text)
