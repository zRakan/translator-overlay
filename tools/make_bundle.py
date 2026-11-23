"""Pack the project into one zip to move to another machine.
python tools/make_bundle.py             source + configs + saved map translations (small; run tools/setup_windows.ps1 there)
python tools/make_bundle.py --offline   also the model and translation-server binaries (~1 GB; only the pip packages are downloaded there)
Never included: .venv (absolute paths), the TensorRT engine cache (per GPU), logs, test videos."""
import argparse
import os
import zipfile

SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", "map_debug", "bench", "translator_overlay.egg-info"}
SKIP_SUFFIX = (".mp4", ".log", ".pid", ".pyc", ".tmp")
HEAVY = ("models", "third_party")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="include models/ and third_party/")
    ap.add_argument("-o", "--out", default="translator-overlay-bundle.zip")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n = size = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(root):
            rel = os.path.relpath(d, root)
            top = rel.split(os.sep)[0]
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not (rel == "." and x in HEAVY and not a.offline)]
            for f in files:
                if f.endswith(SKIP_SUFFIX) or f == a.out:
                    continue
                p = os.path.join(d, f)
                z.write(p, os.path.join("translator-overlay", os.path.relpath(p, root)))
                n += 1
                size += os.path.getsize(p)
    print(f"{a.out}: {n} files, {size / 1e6:.1f} MB uncompressed, {os.path.getsize(a.out) / 1e6:.1f} MB zipped")


if __name__ == "__main__":
    main()
