# Advanced guide

Everything beyond the basics in the [README](../README.md): how the pipeline works, conversations and speaker names, recording, the debug log, tools,
other ways to install and run, and limitations.

## Performance

Measured on an RTX 5070 (1080p, native Windows, RapidOCR (TRT), 0.8B translation model, two recorded games played in real time at 30 and 60 fps):
OCR about 7 ms per subtitle, translation about 50 ms, 65-72 ms median and 100-115 ms 90th percentile from the first readable frame to the
translation being handed to the display (`tools/latency_trace.py`); the display then shows it about 5 ms later (`tools/render_bench.py`).
Capture-card and monitor latency are not included.

RapidOCR (TRT) puts the Arabic on screen 2-4 frames earlier than ONNX / CUDA OCR on the same recording (33-67 ms at 60 fps); the translation is identical.

## Pipeline

```
capture ─> change detector ─> OCR ─> filters ─> translator ─> overlay renderer ─> window | click-through overlay | mp4
```

| Stage | Details |
|---|---|
| Capture | UVC capture card (Media Foundation / DirectShow), single-window capture (Windows Graphics Capture), video file. A reader thread keeps only the newest frame. |
| Change detector | Compares a 96x24 grey copy of the subtitle region. `fast` mode fires on the first changed frame; `poll_ms` handles subtitles over moving backgrounds. |
| OCR | ONNX Runtime with TensorRT, CUDA or CPU. Filters: centred text, ALL-CAPS skip, minimum letters and confidence, fuzzy de-duplication, speaker-label stripping, several speakers on screen at once (split at each `NAME:`, translated and drawn per speaker), a region that is too short for a tall caption is extended upward on demand (the `NAME:` is kept out of the translation and drawn back in front of the Arabic in a per-speaker colour). |
| Translator | Pluggable: `local` (in-process seq2seq), `llm` / `plamo_hf` (in-process causal LM), `chat` (any chat/completions HTTP server), `http` (LibreTranslate-style API), `echo`. LRU cache. Optional previous-line context for sentences split across subtitles. |
| Renderer | Arabic shaped with Pillow's raqm engine, size derived from the OCR line height, blurred backdrop. Drawn at screen resolution after the video is scaled. |
| Output | Own window (`--show`), click-through always-on-top window above a captured window (`--overlay`), or an mp4 (`--out`). |

A second state handles a game's map screen: a pink-pixel check on the tab word (~6 µs, no OCR) switches subtitle translation to the hovered
card and map labels. Layout is inferred from OCR boxes; only the English strokes are erased, so icons and panel pixels stay intact.

## Conversations and speaker names

- **Context:** with `context = true` the previous lines of the same conversation are passed to the model. `context_when = "continuation"` (default) only for
  a line that continues an unfinished sentence, `"always"` for every line (`context_lines`, `context_gap_s`). History is kept per speaker. One or two
  lines are enough: more history costs a little time and adds failure modes (a repeated or empty translation).
- **Speaker labels:** with `[translate] strip_speaker = true` a leading `NAME:` never reaches the model. The overlay draws it again in front of the Arabic in a
  per-speaker colour (`[overlay] speaker`, `speaker_palette`); a listed name is written in Arabic (`speaker_arabic`,
  `translator_overlay/data/speaker_names_ar.tsv`, tab-separated, edit it to add names), an unlisted one stays in English.
- **Several speakers at once:** the text is split at each `NAME:`, each speaker is translated separately and gets its own line and colour; a line that was
  already translated in the conversation is reused. A caption taller than `[region]` is re-read once with the strip above it added.

## Recording and offline runs (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows_run.ps1 start -Record     # show the card (with sound) and record it, no OCR / translation
# k = mark the time, q = stop; the file is <[record] dir>\rec-<time>.mkv (default: recordings\)
python -m translator_overlay -c config.capture.toml --source rec.mkv --out out.mp4 --debug   # then run the pipeline on the recording
python tools/slowmo_compare.py ...                                                  # slow-motion comparison of two renders (see the script)
```

`[record]` (folder, ffmpeg path) and `[audio]` belong in your `config.local.toml`, like the capture device (see the README).

## Install (pip)

Python 3.11+. An NVIDIA GPU is recommended; everything also runs on CPU.

```sh
pip install -e ".[dev]"                 # core + tests
pip install -e ".[ocr]"                 # GPU OCR (RapidOCR, ONNX Runtime GPU, TensorRT)
pip install -e ".[windows]"             # window capture, capture-card tools, audio passthrough
pip install -e ".[local]"               # in-process translators (torch, transformers)
```

`--show` needs OpenCV's GUI build: replace `opencv-python-headless` with `opencv-python`. Models are not included.

## Configuration

All options are documented in [`config.example.toml`](../config.example.toml): `[source]`, `[region]`, `[detect]`, `[translate]`, `[overlay]`, `[ocr]`,
`[display]`, `[map]`, `[audio]`, `[debug]`, `[record]`. Unknown keys raise an error. Ready-made profiles: `config.capture.toml` (capture card, 60 fps),
`config.capture30.toml` (30 fps mode), `config.xbox.toml` (window capture), `config.plamo.toml` (chat backend), `config.nllb.toml` and `config.test.toml` (other translators / a test setup).

## Another machine

- **Get the project:** `git clone`, or an offline bundle: `python tools/make_bundle.py` (source and configs, small zip) or `--offline` (also the model and the translation-server binaries, about 1 GB).
- **Setup:** `tools\setup_windows.ps1` fetches only what is missing; `-SkipModel` skips the model (for example when it came with an offline bundle).
- **First start:** about 30-60 s slower, because TensorRT builds the OCR engines for this GPU once (cached in `%USERPROFILE%\.cache\translator-overlay\trt`).
- **Saved map translations:** `map_translations.json` is created at run time; copy it to keep them, delete an entry to re-translate it.

## Window capture (no capture card)

Captures one window (for example a remote-play window) and draws the subtitles in a click-through window above it.

1. Keep the source window windowed or maximised (exclusive fullscreen hides overlays).
2. Start the translation server: `powershell -ExecutionPolicy Bypass -File tools\windows_server.ps1 start`.
3. `.venv\Scripts\python.exe tools\list_windows.py <filter>` and note part of the window title.
4. `.venv\Scripts\python.exe -m translator_overlay -c config.xbox.toml --source "window:<title part>" --overlay` (Ctrl+C to stop), then `tools\windows_server.ps1 stop`.

The overlay never takes focus and is excluded from screen capture (check with `tools\overlay_selfcheck.py`). Windows 10 draws a thin border around the captured
window; Windows 11 can hide it. On the tested capture card, Media Foundation MJPG 1920x1080@60 delivered a steady 60 fps while DirectShow gave only 9-21 fps.

## Window controls

Besides the keys in the README: `r` shows the subtitle region, then drag a box, `WASD` moves it, `-` `=` change the height, `[` `]` the width, `Enter` saves it to
the config; `p` shows a performance panel (OCR / translation / total latency, display time, fps); `k` (only with `--debug`) marks a problem frame in the debug log.

## Debug log (`--debug`)

```sh
python -m translator_overlay -c config.capture.toml --show --debug
```

Nothing is logged without this flag, and nothing is added to the window. The log is meant to be read afterwards, to find out what happened during a play session:

| File | Content |
|---|---|
| `debug/trace-<time>.jsonl` | one JSON object per line: `start` (the whole config, OCR engine, capture source), `read` (every OCR read and its outcome: `translated`, `same_text`, `low_score`, `no_text`, `ocr_error`, `translate_error`, `no_translation`, `map_card`; for a translation: the English, the Arabic, the context given to the model, queue / OCR / translation / server prompt + generation / display times, **ms from detector fire to on screen**, quality flags), `capture` (every 5 s: fps, frame gaps, stalls, frames skipped), `mark`, `summary` (at exit) |
| `debug/crops/*.png` | the crop the OCR read, for translations, dropped low-score reads and errors |
| `debug/marks/*.png` | `k` in the window: the clean frame at the moment you saw a problem (the trace notes which subtitle was the latest) |
| `debug/LATEST` | name of the newest trace |

`python tools/diagnose.py` (no argument = the newest log) prints the diagnosis: stage times, capture health, slowest subtitles, flagged
translations, dropped / partial reads, inconsistent translations, your marks, and a verdict; `--json` for the raw analysis.
Not measured (not possible in software): the capture card's own delay, USB transfer and the monitor's scan-out. `[debug]` in the config sets `dir` and `crops`.

## Tools

| Tool | Purpose |
|---|---|
| `tools/map_overhead_bench.py` | cost of the map state on the subtitle path |
| `tools/ocr_profile.py` | per-stage OCR timing on subtitle crops |
| `tools/render_bench.py` | cost of rendering the Arabic and scaling/showing the picture, per component |
| `tools/diagnose.py` | diagnosis of a `--debug` log: stage times, capture health, slowest and flagged translations |
| `tools/slowmo_compare.py` | slow-motion video comparing two rendered outputs (for example ONNX vs RapidOCR (TRT)) |
| `tools/latency_trace.py` | per-stage latency of the real pipeline, from the first readable frame to the display hand-off |
| `tools/list_cameras.py`, `camera_probe.py` | find a capture card and its best video mode |
| `tools/list_windows.py`, `capture_probe.py`, `capture_frame.py` | window capture checks |
| `tools/overlay_demo.py`, `overlay_selfcheck.py` | verify the overlay is click-through and excluded from capture |
| `tools/list_audio.py`, `audio_probe.py` | audio devices and passthrough check |
| `tools/windows_run.ps1`, `windows_server.ps1` | start/stop the translation server and the app in the background |

## Tests

```sh
pytest
```

The tests run on CPU without models, GPU, or a display.

## Layout

```
translator_overlay/   package: capture, change, ocr, translate, pipeline, overlay, speakers, recorder, debug, mapstate, winoverlay, audio, guard
                      (fonts/: IBM Plex Sans Arabic under the SIL Open Font License; data/: the speaker-name list)
tests/                unit and pipeline tests
tools/                probes, benchmarks, Windows launchers
docs/                 this guide and the demo GIFs (docs/media/)
config*.toml          example and profile configs (repository root)
CHANGELOG.md          everything that changed, from the first prototype (repository root)
```

## Limitations

- OCR reads the original capture pixels; picture quality limits accuracy on small or low-contrast text.
- Map-screen detection uses fixed positions and a tab colour: it needs a per-game profile (`[map]`).
- Speaker colours come from a fixed palette (two names can share one); the Arabic name list covers the speakers of one game.
- Pillow without raqm (the Windows wheels) draws Arabic through a fallback shaper: diacritics such as tanween are dropped.
