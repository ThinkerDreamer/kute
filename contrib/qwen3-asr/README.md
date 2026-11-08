# Qwen3-ASR for kute

[`kute-qwen3-asr.py`](kute-qwen3-asr.py) runs [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
locally as kute's transcriber, with Qwen3-ForcedAligner for word timestamps. It is a helper for
the `command` backend (see [configuration](../../docs/configuration.md#the-command-backend)).

On a test with a one-hour Russian lesson it transcribed accented Russian better than Whisper
large-v3-turbo and large-v3, and faster than turbo. It was weaker on English names.
Try it on your own calls with `kute retranscribe` before switching.

## Needs

- An NVIDIA GPU with 8 GB or more, and CUDA. The two models take about 5.7 GB, and the desktop
  often holds 1.5 GB or more of the card, so 8 GB has little room to spare. A chunk that runs
  out of memory is split in half and retried.
  It also runs on the CPU (`KUTE_QWEN_DEVICE=cpu`), slowly.
- About 6 GB for the Python environment and 6.2 GB for the models (`Qwen/Qwen3-ASR-1.7B`
  and `Qwen/Qwen3-ForcedAligner-0.6B`, downloaded from Hugging Face on first start).
- Python 3.12 and [uv](https://docs.astral.sh/uv/) (or plain `venv` and `pip`).

Qwen and kute's Whisper are never loaded at the same time: kute uses one backend, and the
helper only runs while kute transcribes.

## Setup

```sh
mkdir -p ~/kute-qwen && cd ~/kute-qwen
cp /path/to/kute/contrib/qwen3-asr/kute-qwen3-asr.py .
uv venv --python 3.12
uv pip install -U qwen-asr torch
```

Tested with qwen-asr 0.0.6, transformers 4.57, torch 2.15 (CUDA 13).

Then in `~/.config/kute/config.toml`:

```toml
[transcribe]
backend = "command"
command = ["~/kute-qwen/.venv/bin/python", "~/kute-qwen/kute-qwen3-asr.py"]
```

The models download the first time kute transcribes, into `~/.cache/huggingface` (set
`HF_HOME` in the daemon's environment to put them elsewhere). Restart the daemon to pick up
the new config.

## Languages

Qwen3-ASR knows 30 languages; word timestamps work for 11 of them (Chinese, Cantonese, English,
French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish). Other languages are
transcribed without word timestamps, one line per speech chunk.

Qwen can't be limited to a list of languages. With `language = "ru,en"`, a chunk where it
hears both is transcribed as the one listed first (Russian), which keeps the English words in
it as English; a chunk where it hears neither is also taken as the first.

Automatic detection can mistake accented speech for a related language: in testing, a
learner's Russian came out as Polish or Czech. Force the language on that track, for example
`kute language <id> --me ru --others ru,en`.

## Settings

Environment variables for the helper (set them with `env` in `command`, e.g.
`command = ["env", "KUTE_QWEN_MODEL=Qwen/Qwen3-ASR-0.6B", "~/kute-qwen/.venv/bin/python", ...]`):

| Variable | Default | |
|---|---|---|
| `KUTE_QWEN_MODEL` | `Qwen/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-0.6B` is smaller and faster. |
| `KUTE_QWEN_ALIGNER` | `Qwen/Qwen3-ForcedAligner-0.6B` | `none` skips word timestamps. |
| `KUTE_QWEN_DEVICE` | `cuda:0` | `cpu` works without a GPU. |
| `KUTE_QWEN_MAX_CHUNK_S` | `30` | Longest chunk kute sends. On 8 GB shared with a desktop, 45 s and up could run out of memory; 30 s was as fast. |
| `KUTE_QWEN_JOIN_GAP_S` | `10` | Speech closer than this goes in one chunk. Qwen leaves short chunks lowercase and unpunctuated; Whisper keeps 2 s, since it makes things up in long silences. |
