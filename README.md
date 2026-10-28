# kute

*kute* is Toki Pona for "listen". kute transcribes your calls on a Linux desktop, locally, and
can summarize them. Your microphone and the other participants are recorded as separate tracks,
so the transcript knows who said what without guessing:

```
# Planning. 2026-10-08 15:00 (42 min, Firefox)

## 15:00 to 15:42

[15:00:12] Me: Ciao! Hai visto la MR?
[15:00:15] Others: Sì, I left a couple of comments on the patch format.
```

**Status: early, in daily use by its author.** A background daemon notices when an app opens
your microphone, records the call, and transcribes and summarizes each conversation as it
ends. You can also import recordings made elsewhere. Next: a notification while recording, a
tray icon, and `kute service install`. See
[the design](docs/superpowers/specs/2026-10-08-kute-design.md) and [the to-do list](docs/TODO.md).

## How it works

- **Calls are detected, not started by hand.** The daemon watches PipeWire. When an app
  (Jitsi, Zoom, a browser) keeps the mic open for 20 s, it records; 15 s after the call ends,
  it stops. A website checking your mic for a few seconds leaves nothing behind. A long
  coworking call becomes one session, split into conversations wherever nobody spoke for
  5 minutes, and each is transcribed and summarized as soon as it ends.
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
  `summary.md`, the audio in the `opus` format, and `meta.toml`. The search index is rebuilt
  from them.

## Documentation

- [Installing](docs/install.md): requirements, building with Vulkan, CUDA or CPU only, models,
  running at login, updating and uninstalling
- [Configuration](docs/configuration.md): every config key, and summarizer examples
- [Privacy](docs/privacy.md): what leaves your machine, what's stored, how to delete it
- [Troubleshooting](docs/troubleshooting.md)
- [Contributing](CONTRIBUTING.md)

## Requirements

- Linux with PipeWire (the daemon uses `pw-dump` and `pw-record`)
- `ffmpeg` and `ffprobe`
- Rust 1.88 or newer, `cmake` and `clang`
- For GPU transcription: Vulkan drivers and headers (`vulkan-headers`, `shaderc`), or CUDA with
  `--no-default-features --features cuda`. Without either it runs on the CPU, more slowly.

## Install

```bash
cargo install --path .
kute model download        # large-v3-turbo (1.6 GB) and the Silero VAD model
```

Nix users: a flake is on its way. See [Installing](docs/install.md) for details.

`kute model list` shows the other Whisper models; `kute model download small` fetches one.
Downloads are checked against known SHA-256 sums.

## Record calls

Run the daemon. In a terminal to try it out:

```bash
kute daemon
```

or as a systemd user service, so it starts when you log in (`kute service install` will
write this for you; for now, save it as `~/.config/systemd/user/kute.service`, and see
[Installing](docs/install.md#running-the-daemon-at-login) for API keys):

```ini
[Unit]
Description=kute: record, transcribe and summarize calls
After=pipewire.service wireplumber.service
Wants=pipewire.service

[Service]
ExecStart=%h/.cargo/bin/kute daemon
Restart=on-failure
RestartSec=5
# include wherever your summarizer command lives (e.g. ~/.local/bin for claude)
Environment=PATH=%h/.local/bin:%h/.cargo/bin:/usr/local/bin:/usr/bin

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now kute
journalctl --user -u kute -f    # what it's doing
```

Then just take calls. When a summary is ready, a notification shows its first line; click it
to open `summary.md`. Control it with:

```bash
kute status                 # idle, recording, ...; the current session; the queue
kute record "Standup"       # record now, until `kute stop`
kute stop                   # end the current session now
kute discard                # end it and delete it
kute pause / kute resume    # stop listening for calls / start again
```

**Mute at the system level** (e.g. the Plasma microphone icon), not only in the call app.
Many call apps keep the microphone open while muted, so kute still hears what you say;
muting the system source silences kute too.

**Everything your computer plays is recorded**, not only the call: the "others" track is the
whole system audio output, so a video or notification sound during a call ends up in the
recording too.

## Import recordings

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

To steer a summary, give the session notes: who the other speakers are, what to focus on, the
tone or level of detail you want. They work during a call too, and the next summary uses them:

```bash
kute notes last "Others: Jordan (PM, asks about dates), Sam (backend). Keep it short."
kute notes last             # print them
kute notes last --clear
kute summarize last         # summarize again with them
```

Other commands: `title`, `open`, `summarize` (again, optionally `--lang it`), `retranscribe`
(e.g. with `--model`), `process` (finish anything that failed), `reindex`. See `kute --help`.
While the daemon runs, these are queued on it and the result shows up in its log (and as a
notification when a summary is written).

## Configure

`~/.config/kute/config.toml`. Every key is optional; [Configuration](docs/configuration.md)
describes each one:

```toml
[user]
name = "Alex"                     # who "Me" is, for summaries

[labels]
me = "Me"
others = "Others"

[detect]
min_duration = "20s"              # how long the mic must stay open before recording counts
stop_grace = "15s"                # how long after the call ends to keep recording
conversation_gap = "5m"           # silence that splits a long session into conversations
# ignore_apps = ["Firefox"]       # never treat these apps as calls (name or binary)

[capture]
# mic = "USB"                     # record this mic (substring of its name), not the default

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
readable only by you. [Privacy](docs/privacy.md) lists what's in them and what, if anything,
leaves your machine.

## Consent and the law

kute records other people. It does not announce itself to the other participants, and laws
about recording conversations differ a lot between countries and states (some require everyone's
consent). Making sure you're allowed to record is your responsibility. Telling people you're
transcribing the call is good manners everywhere.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
