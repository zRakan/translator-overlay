# Changelog

Everything from the first prototype to the current state, in the order it happened (all work was done between 2026-11-21 and 2026-11-23; there are no
tagged releases yet). Figures are from benchmarks and live capture-card sessions on an RTX 5070.

## 2026-11-21: first working pipeline

- **Pipeline:** capture (video file, capture card, window) -> change detector -> OCR -> filters -> translator -> Arabic overlay renderer -> window / mp4.
  Pluggable translators (`local`, `llm`, `plamo_hf`, `chat`, `http`, `echo`) with an LRU cache; TOML config with strict keys (a typo fails loudly).
- **Safety:** `guard.py` (memory preflight and watchdog) after the first run exhausted RAM/VRAM and crashed WSL.
- **OCR:** RapidOCR on ONNX Runtime; centred-text, ALL-CAPS, minimum-letters and confidence filters; fuzzy de-duplication; `Name:` stripping.
  CUDA 13 libraries preloaded (OCR had silently fallen back to CPU).
- **Rendering:** Arabic shaped by Pillow's raqm engine only (a second shaping pass garbled it); IBM Plex Sans Arabic; size matched to the game's own text
  height; blurred, darkened backdrop instead of a black bar; "replace" mode covers the original text box.
- **Translation model selection:** NLLB, Hy-MT2 (BF16, 1.25-bit on an unmerged llama.cpp branch), PLaMo 0.8B / 2B, then Helsinki-NLP Opus models,
  compared on 60 extracted subtitle lines (chrF++ and latency). **PLaMo-3 Translate 0.8B Q4_K_M on the GPU** (llama.cpp CUDA, the model's own prompt
  format through the completions endpoint, temperature 0) became the default: 45.9 chrF++ at 45 ms median.
- **Official prebuilt llama.cpp** replaces source builds (CPU and CUDA).
- **Detector:** `fast` mode (fires on the first changed frame, 2-3 frames sooner) plus a poll timer for subtitles over moving backgrounds; two-letter
  lines ("No.", "Hi.") accepted.
- **OCR speed:** TensorRT engine through ONNX Runtime, English-only recogniser (PP-OCRv5), v6 detector; about 7 ms per subtitle.
- **Previous-line context:** optional, only while lines belong to one conversation (`context`, `context_lines`, `context_gap_s`); by default only for a
  line that continues an unfinished sentence (`context_when = "continuation"`). Lines the model copied are never fed back (retry without history).
- **Capture card:** Media Foundation MJPG 1080p60 (DirectShow gave 9-21 fps; hardware transforms disabled to open in seconds); the app waits and retries
  when the card is not ready. Fullscreen window with bicubic + light sharpening, no frame-rate cap by default (`[display] fps`), `p` performance panel,
  `r` live subtitle-region editor (drag, WASD, Enter saves to the config).
- **Map / blip-card state:** a pink-pixel check on the tab word (about 6 microseconds, no OCR) switches translation to the hovered card; layout inferred
  from OCR line and character boxes; only the English strokes are erased; labels on the zoomed-out map; saved translations; `m` toggles it.
- **Audio passthrough** from the capture card (`[audio]`, `a` mutes).
- **Windows:** native setup script with pinned, checksum-verified downloads; start/stop scripts; window capture (Windows Graphics Capture); click-through
  overlay excluded from capture.
- Fixed: a callback error killed the OCR worker thread (translation stopped after one subtitle); small English text not translated; model answering in
  English; flicker and stale Arabic on map cards.

## 2026-11-22: measurement, OCR tuning, tooling

- **Benchmarks:** Helsinki-NLP Opus models (best: opus-mt-tc-big-en-ar, 41.9 chrF++; PLaMo stays the default).
- **OCR:** profiled per stage on Windows and Linux; patched the detector preprocessing (in-place float32) to remove a slow copy.
- **Latency:** `poll_ms` 200 -> 100; persistent HTTP connection to the translation server; a finished translation redraws immediately instead of waiting
  for the next frame (`wake()`); `tools/latency_trace.py` measures first readable frame -> display hand-off per stage (median 65-72 ms, p90 100-115 ms).
- **`--debug` log:** per-read JSONL trace (outcome, stage times, server prompt/generation time and tokens, quality flags, context used), OCR crops,
  capture health every 5 s, `k` marks a problem frame, summary at exit; nothing is logged without the flag. `tools/diagnose.py` turns it into a report.

## 2026-11-23: conversations, speakers, recording

- **Speaker labels:** `NAME:` is removed before translation and drawn back in front of the Arabic in a stable per-speaker colour; known names are written
  in Arabic (`translator_overlay/data/speaker_names_ar.tsv`, 707 names, unknown names stay in English); a one-letter OCR slip in a name snaps to the nearest
  listed name; a colon misread as a letter still counts as a label; a sentence ending in a capitalised word does not swallow the next label.
- **Several speakers at once:** the OCR string is split at each `NAME:`; each speaker is translated and drawn on its own line with its own coloured name;
  context is kept per speaker; a line already translated in the conversation is reused instead of translated again.
- **Captions taller than the region:** when the text touches the top of the region the pipeline re-reads once with the strip above added, and then reads the
  taller picture directly until the caption is gone.
- **Robustness:** an unexpected error while handling one OCR job drops that job only (`worker_restarts` in the log) instead of ending translation.
  translation; `k` marks the time; `tools/windows_run.ps1 start -Record`. The recording is then processed offline with the full pipeline.
