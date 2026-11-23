import http.client
import json
import re
import threading
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Protocol


# 'NAME:' at the start: up to 3 words, each starting with a capital letter / digit ("Jay Norris", "Mr. Hunter", "OFFICER_2"); "Get in the car: now" is not a label
# A speaker label: up to 3 capitalised words. A word may end in "." only if it is a title ("Ms. Sharleen"), so the last word of the sentence before a label
# ("... and Marina. Ms. Sharleen: ...") is not taken for the start of the name.
_WORD = r"(?:(?:Mr|Mrs|Ms|Dr|Jr|Sr|St|Mt|Lt|Sgt|Capt|Det|Off|Prof)\.|[A-Z0-9][A-Za-z0-9_'\-]*)"
SPEAKER_RE = re.compile(rf"^({_WORD}(?: {_WORD}){{0,2}}):\s+(?=\S)")


def split_speaker(text: str):
    """'Parrot: Bang! Bang!' -> ('Parrot', 'Bang! Bang!'); no label -> (None, text)."""
    m = SPEAKER_RE.match(text)
    return (m.group(1).strip(), text[m.end():]) if m else (None, text)


_MID_RE = re.compile(rf"(?<![\w'.\-])({_WORD}(?: {_WORD}){{0,2}}): (?=\S)")


# The colon after a name is sometimes read as a letter ("Ms. Sharleene Or maybe...", "Brians talk to me"): a listed name followed by one such character
_LOOSE_RE = re.compile(rf"(?<![\w'.\-])({_WORD}(?: {_WORD}){{0,2}}?)([esilESIL;,.!1]) (?=\S)")


def split_segments(text: str, known=lambda name: False, listed=lambda name: False):
    """Several speakers on screen at once come out of the OCR as one string: 'so... Lucia: You don't...' -> [(None, 'so...'), ('Lucia', "You don't...")].
    A label inside the text counts if the line before it ends a sentence or the name is a known one ("Note: ..." in the middle of a sentence does not).
    `listed(name)`: exactly one of the game's speakers; such a name followed by one stray letter instead of the colon is still a label."""
    cuts = []
    for m in _MID_RE.finditer(text):
        before = text[:m.start()].rstrip()
        if m.start() == 0 or (before and (before[-1] in ".?!…\"”’)" or known(m.group(1)))):
            cuts.append((m.start(), m.end(), m.group(1)))
    for m in _LOOSE_RE.finditer(text):
        before = text[:m.start()].rstrip()
        if listed(m.group(1)) and (m.start() == 0 or (before and before[-1] in ".?!…\"”’)" and text[m.end():m.end() + 1].isupper())) and not any(a <= m.start() < b for a, b, _ in cuts):
            cuts.append((m.start(), m.end(), m.group(1)))
    cuts.sort()
    if not cuts:
        return [(None, text)]
    segs = [(None, text[:cuts[0][0]].strip())] if cuts[0][0] > 0 else []
    for k, (a, b, name) in enumerate(cuts):
        end = cuts[k + 1][0] if k + 1 < len(cuts) else len(text)
        segs.append((name.strip(), text[b:end].strip()))
    return [(n, t) for n, t in segs if t]


def strip_speaker(text: str) -> str:
    """'Parrot: Bang! Bang!' -> 'Bang! Bang!'. Speaker labels would otherwise be translated as content."""
    return split_speaker(text)[1]


class Translator(Protocol):
    def translate(self, text: str, history=None) -> str: ...


class Echo:
    """Passthrough, for wiring/benchmarks without a model."""

    def translate(self, text: str, history=None) -> str:
        return text


class Http:
    """LibreTranslate-style HTTP endpoint (self-hosted translation servers)."""

    def __init__(self, url: str, source: str, target: str):
        self.url, self.source, self.target = url, source, target

    def translate(self, text: str, history=None) -> str:
        body = json.dumps({"q": text, "source": self.source, "target": self.target}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)["translatedText"]


LANG = {"en": "English", "ar": "Arabic", "fr": "French", "es": "Spanish", "de": "German", "ja": "Japanese",
        "ko": "Korean", "zh": "Chinese", "ru": "Russian", "tr": "Turkish", "pt": "Portuguese", "it": "Italian"}


class _Connection:
    """One persistent HTTP/1.1 connection. A new connection per request costs ~13 ms on Windows loopback (0.5 ms when reused).
    Reconnects when the server closed it while idle; a request is retried once only if it failed on a reused connection."""

    def __init__(self):
        self._conn, self._key, self._lock = None, None, threading.Lock()

    def _drop(self):
        if self._conn is not None:
            self._conn.close()
        self._conn = None

    def post_json(self, url: str, body: dict, headers: dict, timeout: float = 10) -> dict:
        u, data = urllib.parse.urlsplit(url), json.dumps(body).encode()
        key, path = (u.scheme, u.netloc), (u.path or "/") + (f"?{u.query}" if u.query else "")
        with self._lock:
            for attempt in (0, 1):
                reused = self._conn is not None and self._key == key
                if not reused:
                    self._drop()
                    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
                    self._conn, self._key = cls(u.netloc, timeout=timeout), key
                self._conn.timeout = timeout
                try:
                    self._conn.request("POST", path, data, headers)
                    r = self._conn.getresponse()
                    payload = r.read()
                except (ConnectionError, http.client.HTTPException):  # stale connection (server idle timeout)
                    self._drop()
                    if reused and attempt == 0:
                        continue
                    raise
                except Exception:  # timeout or anything else: never reuse a connection in an unknown state
                    self._drop()
                    raise
                if r.will_close:
                    self._drop()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status} from {u.netloc}: {payload[:200]!r}")
                return json.loads(payload)


_connection = _Connection()
last = threading.local()  # per thread: .timings = the server's own timings for the last request, .hit = the last Cached lookup was a hit (debug view)


def _post_json(url: str, body: dict, headers: dict, timeout: float = 10) -> dict:
    return _connection.post_json(url, body, headers, timeout)


class Chat:
    """Chat-completions HTTP backend (any server exposing /v1/chat/completions, or a cloud API via `api_key`).
    One request per subtitle, no streaming or batching."""

    PROMPT = ("Translate the following text into {target}. Note that you should **only output the "
              "translated result without any additional explanation**:\n\n{text}")

    def __init__(self, url, model, target, api_key="", temperature=0.7, style="hymt", source="en"):
        self.url, self.model, self.target, self.api_key, self.temperature = url, model, LANG.get(target, target), api_key, temperature
        self.style, self.codes = style, (source, target)

    def translate(self, text: str, history=None) -> str:
        if self.style == "plamo-pfml":  # completions prompt with a hand-built PFML3 template
            src, dst = self.codes
            tag = lambda t: t.replace("<|plamo:", "")  # reserved tokens must never come from OCR/translations
            # previous (source, translation) pairs in order, then the current source
            past = "".join(f"<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>{src}<|plamo:msg|>{tag(a)}"
                           f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{dst}<|plamo:msg|>{tag(b)}" for a, b in (history or []))
            prompt = (f"<|plamo:tag|>dataset<|plamo:msg|>translation{past}<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>{src}"
                      f"<|plamo:msg|>{tag(text)}<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{dst}<|plamo:msg|>")  # the server adds BOS
            body = {"prompt": prompt, "temperature": 0, "cache_prompt": True, "stop": ["<|plamo:tag|>", "<|plamo:bos|>"],
                    "max_tokens": min(256, 8 * len(text) + 32)}
            url = self.url.replace("/chat/completions", "/completions")
            hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})}
            r = _post_json(url, body, hdr)
            last.timings = r.get("timings") if isinstance(r, dict) else None
            return r["choices"][0]["text"].strip()
        if self.style == "plamo":  # the server's chat template builds the prompt from these options
            msgs = [m for a, b in (history or []) for m in ({"role": "user", "content": a}, {"role": "assistant", "content": b})]
            body = {"messages": msgs + [{"role": "user", "content": text}], "temperature": 0, "cache_prompt": True,
                    "chat_template_kwargs": {"source_lang": self.codes[0], "target_lang": self.codes[1]},
                    "stop": ["<|plamo:tag|>", "<|plamo:bos|>"], "max_tokens": min(256, 8 * len(text) + 32)}
        else:  # instruction prompt + sampling defaults
            body = {"messages": [{"role": "user", "content": self.PROMPT.format(target=self.target, text=text)}],
                    "temperature": self.temperature, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05, "cache_prompt": True,  # prompt cache reuses the fixed instruction prefix
                    "max_tokens": min(256, 8 * len(text) + 32)}  # cap: a subtitle never needs 4096 tokens
        if self.model:
            body["model"] = self.model
        hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})}
        return _post_json(self.url, body, hdr)["choices"][0]["message"]["content"].strip()


class LLM:
    """In-process causal LM on GPU (transformers, chat template). Greedy, with repetition penalty. One line at a time."""

    def __init__(self, path, target, vram_gb=5.0, rep_penalty=1.05):
        import torch
        self.rep = rep_penalty
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.target = torch, LANG.get(target, target)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":  # hard VRAM cap (the desktop shares the GPU)
            torch.cuda.set_per_process_memory_fraction(min(0.95, vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        for _ in range(3):  # warm-up
            self.translate("Warm up the engine.")

    def translate(self, text: str, history=None) -> str:
        msg = [{"role": "user", "content": Chat.PROMPT.format(target=self.target, text=text)}]
        ids = self.tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.m.device)
        with self.torch.inference_mode():
            out = self.m.generate(**ids, do_sample=False, repetition_penalty=self.rep, max_new_tokens=min(256, 8 * len(text) + 32))
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class PlamoHF:
    """bf16 causal LM via transformers. The repository ships its own modeling/tokenizer code, so this uses trust_remote_code
    and only accepts a local, audited directory (never a hub id)."""

    def __init__(self, path, source, target, vram_gb=4.0):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.path.isdir(path):
            raise ValueError("plamo_hf needs a local, audited model directory (trust_remote_code runs repo code)")
        self.torch, self.src, self.dst = torch, source, target
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            torch.cuda.set_per_process_memory_fraction(min(0.95, vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        # the model ends a translation by emitting one of these tags instead of EOS
        self.stop = [i for i in self.tok.convert_tokens_to_ids(["<|plamo:tag|>", "<|plamo:bos|>"]) if isinstance(i, int)] + [self.tok.eos_token_id]
        for _ in range(3):  # warm-up
            self.translate("Warm up the engine.")

    def translate(self, text: str, history=None) -> str:
        ids = self.tok.apply_chat_template([{"role": "user", "content": text}], source_lang=self.src, target_lang=self.dst,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.m.device)
        with self.torch.inference_mode():
            out = self.m.generate(**ids, do_sample=False, max_new_tokens=min(256, 8 * len(text) + 32), eos_token_id=self.stop)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class Local:
    """In-process seq2seq on GPU (Marian by default; other seq2seq models work). fp16, greedy: subtitles are short, so latency beats beam-search quality.
    One line at a time."""

    NLLB = {"en": "eng_Latn", "ar": "arb_Arab"}

    def __init__(self, model: str, source: str, target: str, beams: int = 1, target_tag: str = ""):
        self.beams = beams
        self.prefix = f"{target_tag} " if target_tag else ""  # multilingual Marian models pick the output language from this tag
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        if torch.cuda.is_available():
            # hard VRAM cap (the desktop shares the GPU): ~3 GB for a 600M seq2seq in fp16, ~1.4 GB for Marian
            torch.cuda.set_per_process_memory_fraction(0.25 if "nllb" in model else 0.12)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"src_lang": self.NLLB[source]} if "nllb" in model else {}
        self.tok = AutoTokenizer.from_pretrained(model, **kw)
        self.m = AutoModelForSeq2SeqLM.from_pretrained(
            model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        self.gen = {"forced_bos_token_id": self.tok.convert_tokens_to_ids(self.NLLB[target])} if kw else {}
        for _ in range(3):  # warm-up: first CUDA calls are slow
            self.translate("Warm up the engine.")

    def translate(self, text: str, history=None) -> str:
        out = self._gen(text)
        if sum(c.isalpha() for c in out) < 2:
            # small MT models degenerate into '.....' / '- - -' on short lines;
            # one clean full stop usually fixes it
            retry = re.sub(r"\s*(\.{2,}|\u2026)\s*$", ".", text.strip())
            out = self._gen(retry if retry[-1:] in ".!?" else retry + ".")
        out = out.strip(" -.")
        return out if sum(c.isalpha() for c in out) >= 2 else ""  # still junk: show nothing

    def _gen(self, text: str) -> str:
        b = self.tok(self.prefix + text, return_tensors="pt", truncation=True, max_length=128).to(self.m.device)
        with self.torch.inference_mode():
            out = self.m.generate(**b, num_beams=self.beams, do_sample=False, max_length=min(128, int(b.input_ids.shape[1] * 2.5) + 8), **self.gen)  # cap runaway repetition
        return self.tok.decode(out[0], skip_special_tokens=True)


class Cached:
    def __init__(self, inner: Translator, size: int):
        self.inner, self.size, self._c = inner, size, OrderedDict()

    def translate(self, text: str, history=None) -> str:
        key = (text, tuple(h[0] for h in history)) if history else text  # same line after a different previous line may translate differently
        last.timings = None
        last.hit = key in self._c
        if last.hit:
            self._c.move_to_end(key)
            return self._c[key]
        out = self._c[key] = self.inner.translate(text, history) if history else self.inner.translate(text)
        if len(self._c) > self.size:
            self._c.popitem(last=False)
        return out


def build(cfg) -> Translator:
    # new backend = one class above + one entry here
    backends = {"echo": lambda: Echo(),
                "local": lambda: Local(cfg.model, cfg.source, cfg.target, cfg.beams, cfg.target_tag),
                "plamo_hf": lambda: PlamoHF(cfg.model, cfg.source, cfg.target),
                "llm": lambda: LLM(cfg.model, cfg.target),
                "chat": lambda: Chat(cfg.chat_url, cfg.api_model, cfg.target, cfg.api_key, cfg.temperature, cfg.chat_style, cfg.source),
                "http": lambda: Http(cfg.url, cfg.source, cfg.target)}
    if cfg.backend not in backends:
        raise ValueError(f"unknown translate backend {cfg.backend!r}; have {sorted(backends)}")
    return Cached(backends[cfg.backend](), cfg.cache_size)
