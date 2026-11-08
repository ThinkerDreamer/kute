#!/usr/bin/env python3
"""Qwen3-ASR as a kute transcription backend (transcribe.backend = "command").

kute starts this once per transcription run and sends it VAD chunks of 16 kHz mono audio; it
answers with segments and word timestamps from Qwen3-ForcedAligner. The protocol is described
in src/transcribe/command.rs. See README.md next to this file for setup.

Environment:
  KUTE_QWEN_MODEL        ASR model (default Qwen/Qwen3-ASR-1.7B; Qwen/Qwen3-ASR-0.6B is smaller)
  KUTE_QWEN_ALIGNER      aligner model (default Qwen/Qwen3-ForcedAligner-0.6B; "none" to skip)
  KUTE_QWEN_DEVICE       torch device (default cuda:0)
  KUTE_QWEN_MAX_CHUNK_S  longest chunk kute sends (default 30; on an 8 GB card shared with a
                         desktop, 45 s and up can run out of memory)
  KUTE_QWEN_JOIN_GAP_S   speech closer than this goes in one chunk (default 10; Qwen
                         punctuates short chunks poorly)
"""

import hashlib
import json
import os
import re
import sys
import warnings

# Only protocol lines may reach stdout: keep the real stdout for them and send everything
# else, including output from C libraries, to stderr.
PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from qwen_asr import Qwen3ASRModel  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

# transformers logs "Setting `pad_token_id`..." on every generate call.
hf_logging.set_verbosity_error()

SAMPLE_RATE = 16_000
# A word the aligner says lasts longer than this was stretched over a pause before it; its
# start is moved up. On a one-hour lesson about 5% of words were, three quarters of them
# inside a chunk rather than at its start, so every word is checked.
MAX_WORD_MS = 1_000
# A pause longer than this between words starts a new segment.
GAP_MS = 800
# A segment longer than this ends at the next sentence end.
MAX_SEGMENT_MS = 30_000

# kute's language codes and Qwen's names, for the 30 languages Qwen3-ASR knows.
LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "tl": "Filipino",
    "fa": "Persian", "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
}
CODES = {name.lower(): code for code, name in LANGUAGES.items()}
# Languages Qwen3-ForcedAligner can time; others come back without word timestamps.
ALIGNABLE = {"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}
WORDY = re.compile(r"\w", re.UNICODE)
# A chunk that runs out of GPU memory is split in half and retried, down to this length.
MIN_SPLIT_SAMPLES = 5 * SAMPLE_RATE


def is_oom(e):
    """Whether `e` is the GPU running out of memory, which cuBLAS reports in its own way."""
    text = str(e)
    return (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in text.lower()
        or "CUBLAS_STATUS_ALLOC_FAILED" in text
    )


def digest(samples):
    return hashlib.blake2b(samples.tobytes(), digest_size=16).digest()


def send(obj):
    PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTO.flush()


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            raise EOFError("stdin closed in the middle of the audio")
        buf += part
    return bytes(buf)


def detected_codes(text):
    """Qwen's language field, "Russian,English" say, as kute codes in order."""
    out = []
    for name in (text or "").split(","):
        code = CODES.get(name.strip().lower())
        if code and code not in out:
            out.append(code)
    return out


def words_from(text, items):
    """Pairs aligner items with the punctuated tokens of `text`. The aligner returns one item
    per whitespace token with a word character in it; if the counts don't line up, its own
    (unpunctuated) tokens are used rather than misaligning."""
    tokens = [t for t in text.split() if WORDY.search(t)]
    if len(tokens) != len(items):
        tokens = [i.text for i in items]
    words = []
    for token, item in zip(tokens, items):
        start = max(0, int(item.start_time * 1000))
        end = max(start, int(item.end_time * 1000))
        start = max(start, end - MAX_WORD_MS)
        words.append({"start_ms": start, "end_ms": end, "text": " " + token})
    return words


def segments_from(words):
    segments, current = [], []
    for w in words:
        if current and (
            w["start_ms"] - current[-1]["end_ms"] > GAP_MS
            or (
                w["start_ms"] - current[0]["start_ms"] > MAX_SEGMENT_MS
                and current[-1]["text"][-1:] in ".?!"
            )
        ):
            segments.append(current)
            current = []
        current.append(w)
    if current:
        segments.append(current)
    return [
        {
            "start_ms": s[0]["start_ms"],
            "end_ms": s[-1]["end_ms"],
            "text": "".join(w["text"] for w in s),
            "words": s,
        }
        for s in segments
    ]


class Backend:
    def __init__(self):
        name = os.environ.get("KUTE_QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")
        aligner = os.environ.get("KUTE_QWEN_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B")
        device = os.environ.get("KUTE_QWEN_DEVICE", "cuda:0")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        kwargs = dict(dtype=dtype, device_map=device, max_inference_batch_size=1,
                      max_new_tokens=1024)
        self.align = aligner.lower() != "none"
        if self.align:
            kwargs["forced_aligner"] = aligner
            kwargs["forced_aligner_kwargs"] = dict(dtype=dtype, device_map=device)
        self.model = Qwen3ASRModel.from_pretrained(name, **kwargs)
        self.name = name.rsplit("/", 1)[-1].lower()
        self.max_chunk_ms = int(float(os.environ.get("KUTE_QWEN_MAX_CHUNK_S", "30")) * 1000)
        self.join_gap_ms = int(float(os.environ.get("KUTE_QWEN_JOIN_GAP_S", "10")) * 1000)
        # kute detects the language of every chunk of a track, then transcribes them.
        # Detection is a full transcription, so keep each one (by a digest of its audio)
        # and reuse its text when that chunk is transcribed in a language it heard.
        self.detected = {}

    def run(self, samples, language, timestamps):
        return self.model.transcribe(
            audio=(samples, SAMPLE_RATE), language=language, return_time_stamps=timestamps
        )[0]

    def detect(self, samples, candidates):
        try:
            r = self.run(samples, None, False)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or len(samples) < 2 * MIN_SPLIT_SAMPLES:
                raise
            torch.cuda.empty_cache()
            print("kute-qwen3-asr: out of GPU memory, detecting on half", file=sys.stderr)
            return self.detect(samples[: len(samples) // 2], candidates)
        self.detected[digest(samples)] = r
        found = detected_codes(r.language)
        if not candidates:
            if not found:
                raise ValueError("no language detected")
            return found[0]
        # Qwen can't be limited to a list. When it hears several candidates (a teacher
        # explaining Russian in English, say), take the one listed first: forcing the main
        # language keeps the other one's words as they are, while forcing the other one
        # transliterates the main language. With none heard, take the first candidate.
        for code in candidates:
            if code in found:
                return code
        return candidates[0]

    def transcribe(self, samples, code):
        """Like `transcribe_once`, but a chunk that runs out of GPU memory (the desktop's
        share of the card varies) is split in half and each half transcribed."""
        try:
            return self.transcribe_once(samples, code)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or len(samples) < 2 * MIN_SPLIT_SAMPLES:
                raise
        torch.cuda.empty_cache()
        print("kute-qwen3-asr: out of GPU memory, splitting the chunk", file=sys.stderr)
        half = len(samples) // 2
        first = self.transcribe(samples[:half], code)
        second = self.transcribe(samples[half:], code)
        shift = int(half * 1000 / SAMPLE_RATE)
        for seg in second:
            seg["start_ms"] += shift
            seg["end_ms"] += shift
            for w in seg["words"]:
                w["start_ms"] += shift
                w["end_ms"] += shift
        return first + second

    def transcribe_once(self, samples, code):
        if code not in LANGUAGES:
            raise ValueError(f"Qwen3-ASR does not know language {code!r}")
        language = LANGUAGES[code]
        if len(samples) == 0:
            return []
        items = None
        cached = self.detected.pop(digest(samples), None)
        # Reuse a detection pass only if it heard just this language; a mixed chunk is
        # transcribed again with the language forced.
        if cached is not None and detected_codes(cached.language) != [code]:
            cached = None
        if cached is not None:
            r = cached
            if self.align and code in ALIGNABLE and (r.text or "").strip():
                try:
                    items = list(self.model.forced_aligner.align(
                        audio=(samples, SAMPLE_RATE), text=r.text, language=language)[0])
                except Exception as e:  # noqa: BLE001
                    if is_oom(e):
                        raise
                    print(f"kute-qwen3-asr: aligner failed, no word times: {e}", file=sys.stderr)
        elif self.align and code in ALIGNABLE:
            try:
                r = self.run(samples, language, True)
                items = list(r.time_stamps) if r.time_stamps is not None else []
            except Exception as e:  # noqa: BLE001
                if is_oom(e):
                    raise
                print(f"kute-qwen3-asr: aligner failed, no word times: {e}", file=sys.stderr)
        if items is None and cached is None:
            r = self.run(samples, language, False)
        text = (r.text or "").strip()
        if not text:
            return []
        words = words_from(text, items or [])
        if not words:
            # No word times: one segment over the whole chunk.
            end = int(len(samples) * 1000 / SAMPLE_RATE)
            return [{"start_ms": 0, "end_ms": end, "text": " " + text, "words": []}]
        return segments_from(words)


def main():
    backend = Backend()
    send({"name": backend.name, "max_chunk_ms": backend.max_chunk_ms,
          "join_gap_ms": backend.join_gap_ms})
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return
        try:
            req = json.loads(line)
            n = int(req["samples"])
            samples = np.frombuffer(read_exact(stdin, 4 * n), dtype="<f4").astype(np.float32)
        except EOFError:
            return
        try:
            if req["op"] == "detect":
                send({"language": backend.detect(samples, req.get("candidates") or [])})
            elif req["op"] == "transcribe":
                send({"segments": backend.transcribe(samples, req["language"])})
            else:
                send({"error": f"unknown op {req['op']!r}"})
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                torch.cuda.empty_cache()
                send({"error": f"out of GPU memory even in 5 s pieces; is the GPU busy? ({e})"})
            else:
                send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
