# kute

*kute* is [Toki Pona](https://awensona.sunonena.eu/?utm_source=kute) for "listen". kute transcribes your calls on a Linux desktop, locally, and
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
ends. A tray icon shows what it's doing, and `kute doctor` checks your setup. You can also
import recordings made elsewhere. See
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
  between English and Italian keeps both. Tell kute which languages a call uses and it only picks
  among those.
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
kute doctor                # checks PipeWire, ffmpeg, the GPU backend, models and the summarizer
```

Nix users: a flake is on its way. See [Installing](docs/install.md) for details.

`kute model list` shows the other Whisper models; `kute model download small` fetches one.
Downloads are checked against known SHA-256 sums.

## Record calls

Run the daemon. In a terminal to try it out:

```bash
kute daemon
```

or as a systemd user service, so it starts when you log in:

```bash
kute service install               # writes ~/.config/systemd/user/kute.service, enables and starts it
journalctl --user -u kute -f       # what it's doing
```

See [Installing](docs/install.md#running-the-daemon-at-login) for API keys and for replacing a
unit you wrote by hand.

Then just take calls. While one is recorded, a notification "Recording: <app>" offers Stop and
Discard, and the mic in the tray icon (an ear and a mic) turns red. When a summary is ready, a
notification shows its first line; click it to open `summary.md`. The tray icon's menu records,
stops, turns call detection off and on, and opens the recordings folder. While a call is recorded, it also asks for
the call's title, who else is on it, its languages and notes for the summary, in a small dialog
(kdialog, or zenity where there is no kdialog). From a terminal:

```bash
kute status                 # idle, recording, ...; the current session; the queue
kute record "Standup"       # record now, until `kute stop` or the call ends
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

Name the others, and the transcript labels their track with the name instead of "Others" (and
the summary prompt says who that is):

```bash
kute others last Jordan          # or "Jordan, Sam"
kute others last                 # print the label
kute others last --clear         # back to labels.others from the config
```

When Whisper hears languages that aren't there (say, parts of a Russian lesson as another
language), tell kute what is spoken. One language is forced; several limit detection to those.
Set it at the start of a call, since each conversation is transcribed as soon as it ends:

```bash
kute language last ru          # only Russian
kute language last ru en       # Russian or English, detected per chunk
kute language last --me ru --others ru,en   # per track: a lesson where only the teacher
                                            # switches to English
kute language last             # print it
kute language last auto        # detect any language, whatever the config says
kute language last --clear     # back to the config's transcribe.language
kute retranscribe last --lang ru,en   # set it and transcribe a past call again
```

Forcing a language works best when you know it. Whisper writes everything in the language it's
given, so a chunk of Russian labelled English comes out translated or spelled in Latin letters,
and an aside in another language comes out as Russian. `retranscribe` and `import` take `--lang`,
`--me-lang` and `--others-lang`. To use the same languages for every call, set
`transcribe.language` in the config.

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
language = "auto"                 # or force one, e.g. "it", or pick among some, e.g. "ru,en"

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

[tray]
enabled = true                    # the tray icon
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
