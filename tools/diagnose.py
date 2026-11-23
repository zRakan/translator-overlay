"""Turn a --debug log into a diagnosis: performance (where the time goes, capture health) and translation quality (what to look at).
  python tools/diagnose.py [debug/trace-<time>.jsonl | debug-dir] [--top 8] [--json]
No argument = the newest log in ./debug (debug/LATEST). Read-only."""
import argparse
import collections
import json
import os
import statistics as st
import sys


def load(arg):
    path = arg or "debug"
    if os.path.isdir(path):
        latest = os.path.join(path, "LATEST")
        name = open(latest, encoding="utf-8").read().strip() if os.path.exists(latest) else max((f for f in os.listdir(path) if f.startswith("trace-")), default="")
        if not name:
            sys.exit(f"no trace-*.jsonl in {path}")
        path = os.path.join(path, name)
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # a line cut off by a crash
    return path, rows


def pct(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else None


def stat(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return {"n": len(v), "median": round(st.median(v), 1), "p90": round(pct(v, .9), 1), "max": round(max(v), 1)} if v else None


def analyse(rows):
    start = next((r for r in rows if r.get("event") == "start"), {})
    reads = [r for r in rows if r.get("event") == "read"]
    tr = [r for r in reads if r["outcome"] == "translated"]
    live = [r for r in tr if not r.get("cache_hit")]
    caps = [r for r in rows if r.get("event") == "capture"]
    marks = [r for r in rows if r.get("event") == "mark"]
    summ = next((r for r in reversed(rows) if r.get("event") == "summary"), None)
    out = {"session": {"time": start.get("time"), "platform": start.get("platform"), "ocr": start.get("ocr"), "source": start.get("source"),
                       "fps_reported": start.get("fps_reported"), "config_file": start.get("config_file"), "finished_cleanly": summ is not None},
           "outcomes": dict(collections.Counter(r["outcome"] for r in reads)), "stages": {}, "capture": {}, "flags": {}, "problems": []}
    for k in ("queue_ms", "ocr_ms", "translate_ms", "server_prompt_ms", "server_gen_ms", "handoff_ms", "on_screen_ms"):
        s = stat(live, k)
        if s:
            out["stages"][k] = s
    if live:
        out["stages"]["gen_tokens"] = stat(live, "gen_tokens")
        out["stages"]["prompt_tokens_new"] = stat(live, "prompt_tokens")
    if caps:
        out["capture"] = {"fps_median": round(st.median(c["fps"] for c in caps if "fps" in c), 1), "gap_max_ms": max(c.get("gap_max_ms", 0) for c in caps),
                          "gap_p95_ms_worst": max(c.get("gap_p95_ms", 0) for c in caps), "stalls": caps[-1].get("stalls"), "skipped_by_loop": caps[-1].get("skipped_by_loop"),
                          "frames": caps[-1].get("frames")}
    # quality
    flags = collections.Counter(f for r in tr for f in r.get("flags", []))
    out["flags"] = dict(flags)
    slow = sorted(live, key=lambda r: -(r.get("on_screen_ms") or r.get("worker_ms") or 0))
    out["slowest"] = [{"n": r["n"], "t": r["t"], "src": r["src"], "on_screen_ms": r.get("on_screen_ms"), "queue": r["queue_ms"], "ocr": r["ocr_ms"], "translate": r["translate_ms"],
                       "server_prompt": r.get("server_prompt_ms"), "server_gen": r.get("server_gen_ms"), "gen_tokens": r.get("gen_tokens")} for r in slow[:8]]
    out["flagged"] = [{"n": r["n"], "t": r["t"], "src": r["src"], "out": r["out"], "flags": r["flags"], "ocr_score": r.get("ocr_score"), "crop": r.get("crop")} for r in tr if r.get("flags")]
    out["low_score_reads"] = [{"t": r["t"], "text": r.get("text"), "score": r.get("ocr_score"), "crop": r.get("crop")} for r in reads if r["outcome"] == "low_score"][:20]
    out["errors"] = [{"t": r["t"], "outcome": r["outcome"], "error": r.get("error"), "text": r.get("text")} for r in reads if r["outcome"] in ("ocr_error", "translate_error", "no_translation")]
    # partial / flickering reads: a translation whose source is very close to the previous one a moment earlier
    import difflib

    flick = []
    for a, b in zip(tr, tr[1:]):
        if b["t"] - a["t"] < 3.0 and difflib.SequenceMatcher(None, a["src"].lower(), b["src"].lower()).ratio() >= 0.6:
            flick.append({"t": b["t"], "first": a["src"], "then": b["src"], "out_first": a["out"], "out_then": b["out"]})
    out["partial_or_flicker"] = flick[:20]
    # the same English translated differently
    by = collections.defaultdict(set)
    for r in tr:
        by[r["src"].strip().lower()].add(r["out"])
    out["inconsistent"] = [{"src": k, "outs": sorted(v)} for k, v in by.items() if len(v) > 1][:20]
    last_cap = caps[-1] if caps else {}
    out["multi_speaker"] = sum(1 for r in tr if r.get("segments"))
    out["extended_reads"] = last_cap.get("extended", 0)
    out["worker_restarts"] = last_cap.get("worker_restarts", 0)
    out["tall_direct_reads"] = last_cap.get("tall_direct", 0)
    out["reused_lines"] = sum(r.get("reused_lines") or 0 for r in tr)
    out["marks"] = []
    for m in marks:
        near = [r for r in tr if r["t"] <= m["t"]][-3:]
        out["marks"].append({"n": m["n"], "t": m["t"], "frame": m.get("frame_file"), "shown": m.get("shown_file"), "overlay": m.get("overlay"), "recent": [{"n": r["n"], "src": r["src"], "out": r["out"], "on_screen_ms": r.get("on_screen_ms")} for r in near]})
    # verdicts
    st_ = out["stages"]
    P = out["problems"]
    if "on_screen_ms" in st_ and st_["on_screen_ms"]["p90"] > 150:
        P.append(f"p90 detector->screen {st_['on_screen_ms']['p90']} ms: slow tail (see 'slowest')")
    if out["capture"].get("stalls"):
        P.append(f"{out['capture']['stalls']} capture stalls (gap max {out['capture']['gap_max_ms']} ms): the card / USB / our loop paused")
    if out["capture"].get("skipped_by_loop"):
        P.append(f"{out['capture']['skipped_by_loop']} frames skipped by our loop (it was slower than the card at times)")
    if out["outcomes"].get("same_text", 0) > 15 * max(1, len(tr)):
        P.append(f"{out['outcomes']['same_text']} re-reads of the same text for {len(tr)} translations: far more OCR than needed (poll_ms too low, or the picture keeps changing)")
    if flick:
        P.append(f"{len(flick)} translations of a near-identical line within 3 s: partial reads (animating text) cost extra translation calls")
    for f, n in flags.items():
        P.append(f"{n} translations flagged '{f}'")
    if out["worker_restarts"]:
        P.append(f"{out['worker_restarts']} unexpected worker errors (a job was dropped each time; see the worker_error rows)")
    if out["errors"]:
        P.append(f"{len(out['errors'])} errors / empty translations")
    return out


def show(path, a):
    print(f"log: {path}")
    s = a["session"]
    print(f"session: {s['time']}  {s['platform']}\n  ocr {s['ocr']}  source {s['source']}  reported fps {s['fps_reported']}  config {s['config_file']}"
          + ("" if s["finished_cleanly"] else "\n  NOTE: no summary line - the run did not end cleanly (crash or killed)"))
    print(f"outcomes: {a['outcomes']}")
    if a["capture"]:
        c = a["capture"]
        print(f"capture: {c['fps_median']} fps median, worst p95 gap {c['gap_p95_ms_worst']} ms, max gap {c['gap_max_ms']} ms, stalls {c['stalls']}, skipped {c['skipped_by_loop']}, frames {c['frames']}")
    print(f"several speakers at once: {a['multi_speaker']} subtitles; taller re-reads (caption above the region): {a['extended_reads']} first + {a['tall_direct_reads']} direct; lines reused instead of re-translated: {a['reused_lines']}; worker errors: {a['worker_restarts']}")
    print("stages for translations not served from the cache (ms):")
    for k, v in a["stages"].items():
        if v:
            print(f"  {k:<18} median {v['median']:7.1f}  p90 {v['p90']:7.1f}  max {v['max']:7.1f}   (n={v['n']})")
    print(f"slowest {min(a.get('top', 5), len(a['slowest']))}:")
    for r in a["slowest"][:a.get("top", 5)]:
        print(f"  #{r['n']} t={r['t']}s  {r['on_screen_ms']} ms  (queue {r['queue']:.0f} ocr {r['ocr']:.0f} translate {r['translate']:.0f} [server {r['server_prompt']}+{r['server_gen']}, {r['gen_tokens']} tok])  {r['src'][:50]!r}")
    for title, key, fmt in (("flagged translations", "flagged", lambda r: f"#{r['n']} t={r['t']}s {r['flags']} {r['src'][:44]!r} -> {r['out'][:40]!r}"),
                            ("low-score reads (dropped)", "low_score_reads", lambda r: f"t={r['t']}s score {r['score']} {r['text'][:50]!r}"),
                            ("partial / flickering reads", "partial_or_flicker", lambda r: f"t={r['t']}s {r['first'][:34]!r} then {r['then'][:34]!r}"),
                            ("same English, different Arabic", "inconsistent", lambda r: f"{r['src'][:40]!r}: {len(r['outs'])} variants"),
                            ("errors", "errors", lambda r: f"t={r['t']}s {r['outcome']} {r.get('error') or ''} {str(r.get('text') or '')[:40]!r}"),
                            ("marks (k)", "marks", lambda r: f"#{r['n']} t={r['t']}s frame {r['frame']} shown {r['shown']}  recent: {[(x['src'][:30], x['out'][:20]) for x in r['recent']]}")):
        if a[key]:
            print(f"{title} ({len(a[key])}):")
            for r in a[key][:a.get("top", 5) * 2]:
                print("  " + fmt(r))
    print("verdict:" + ("" if a["problems"] else " nothing stands out"))
    for p in a["problems"]:
        print("  - " + p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="print the analysis as JSON")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    path, rows = load(a.log)
    res = analyse(rows)
    res["top"] = a.top
    print(json.dumps({k: v for k, v in res.items() if k != "top"}, ensure_ascii=False, indent=1)) if a.json else show(path, res)
