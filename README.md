# kute

*kute* is Toki Pona for "listen". kute transcribes your calls on a Linux desktop, locally, and
can summarize them. Your microphone and the other participants are recorded as separate tracks,
so the transcript knows who said what without guessing:

```
# Planning. 2026-10-08 15:00 (42 min, firefox)

## 15:00 to 15:42

[15:00:12] Me: Ciao! Hai visto la MR?
[15:00:15] Others: Sì, I left a couple of comments on the patch format.
```

**Status: early.** The offline pipeline works: import two recordings, get a transcript, a
summary per conversation, compressed audio and full-text search. Automatic call detection and
recording (a background daemon watching PipeWire) is being built next. See
[the design](docs/superpowers/specs/2026-10-08-kute-design.md) and [the to-do list](docs/TODO.md).

## How it works

- **Local transcription** with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) on your GPU
  (Vulkan by default, CUDA optional). Audio never leaves your machine unless you configure a
  remote summarizer.
- **Silence is skipped before Whisper sees it.** Silero VAD finds the speech, and only those
  chunks are transcribed. That avoids Whisper's hallucinations on silence and keeps hours-long
  recordings fast.
- **Mixed languages work.** The language is detected per speech chunk, so a call that switches
  between English and Italian keeps both.
- **Summaries are optional and pluggable:** any command (e.g. `claude -p`), any OpenAI-compatible
  endpoint (Ollama, llama.cpp, vLLM, OpenRouter), or the Anthropic API.
- **Plain files are the source of truth.** Each call is a folder with `transcript.md`,
  `summary.md`, the audio as Opus, and `meta.toml`. The search index is rebuilt from them.

## Requirements

- Linux
- `ffmpeg` and `ffprobe`
- Rust 1.88 or newer
- For GPU transcription: Vulkan drivers and headers (`vulkan-headers`, `shaderc`), or CUDA with
  `--no-default-features --features cuda`. Without either it runs on the CPU, more slowly.

## Install

```bash
cargo install --path .
kute model download        # large-v3-turbo (1.6 GB) and the Silero VAD model
```

`kute model list` shows the other Whisper models; `kute model download small` fetches one.
Downloads are checked against known SHA-256 sums.

## Use

Import two recordings, your microphone first, then everyone else:

```bash
kute import me.wav others.wav --start "2026-10-08 15:00" --title "Planning"
kute show last              # the transcript
kute show last --summary    # the summary
kute search "patch format"  # across all calls
kute list
```

Recordings in the `record-call` layout (`<prefix>-me.wav`, `<prefix>-others.wav`,
`<prefix>.start`) pick up their start time automatically.

Other commands: `title`, `open`, `summarize` (again, optionally `--lang it`), `retranscribe`
(e.g. with `--model`), `process` (finish anything that failed), `reindex`. See `kute --help`.

## Configure

`~/.config/kute/config.toml`. Every key is optional:

```toml
[user]
name = "Alex"                     # who "Me" is, for summaries

[labels]
me = "Me"
others = "Others"

[transcribe]
model = "large-v3-turbo"          # or a path to a ggml .bin file
language = "auto"                 # or force one, e.g. "it"

[summary]
backend = "command"               # none | command | openai | anthropic
command = ["claude", "-p"]        # gets the prompt on stdin, prints the summary
language = "en"                   # "auto" = the transcript's main language
min_words = 80                    # shorter conversations aren't summarized
# base_url = "http://localhost:11434/v1"   # openai backend (here: Ollama)
# model = "qwen3:8b"
# api_key_env = "OPENAI_API_KEY"

[notify]
open = ["xdg-open"]               # opens summary.md when you click "Summary ready"
```

Data lives in `~/.local/share/kute` (change it with `[storage] dir = ...`). The folders are
readable only by you.

## Consent and the law

kute records other people. It does not announce itself to the other participants, and laws
about recording conversations differ a lot between countries and states (some require everyone's
consent). Making sure you're allowed to record is your responsibility. Telling people you're
transcribing the call is good manners everywhere.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
