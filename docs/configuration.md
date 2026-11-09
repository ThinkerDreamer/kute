# Configuration

kute reads `~/.config/kute/config.toml` (or `$XDG_CONFIG_HOME/kute/config.toml`). The file is
optional and every key has a default, so you only write the keys you want to change. The daemon
reads the config when it starts: restart it after an edit (`systemctl --user restart kute`).

`kute --config FILE <command>` uses another file. While the daemon runs, commands such as
`retranscribe` or `summarize` hand their work to it only if it uses the same file. Otherwise they
run in the terminal with the other file's settings and archive. If that file shares the daemon's
archive (no `[storage] dir` of its own), kute refuses, so the two don't work on the same sessions
at once: stop the daemon first, or give the file its own `dir`.

Durations are written as strings such as `"20s"`, `"5m"` or `"1h 30m"`.

## `[user]`

| Key | Default | |
|---|---|---|
| `name` | `""` | Your name. Summaries use it for the "Me" track, so action items read "Alex: send the draft" instead of "Me: ...". |

## `[labels]`

| Key | Default | |
|---|---|---|
| `me` | `"Me"` | Label for your microphone in transcripts. |
| `others` | `"Others"` | Label for everyone else (the system audio). A session can name them instead (`kute others`, or the tray). |

## `[storage]`

| Key | Default | |
|---|---|---|
| `dir` | `$XDG_DATA_HOME/kute`, normally `~/.local/share/kute` | Where sessions, models and the search index live. `~` is expanded. kute makes this folder readable only by you (mode 0700). |

## `[detect]`

How the daemon decides that a call started and ended. See [How it works](../README.md#how-it-works).

| Key | Default | |
|---|---|---|
| `min_duration` | `"20s"` | How long an app must keep the microphone open before kute keeps the recording. Shorter mic checks leave nothing behind. |
| `stop_grace` | `"15s"` | How long after the call ends kute keeps recording, in case it comes back (a reconnect). |
| `conversation_gap` | `"5m"` | Silence on both sides that splits a long session into separate conversations, each with its own summary. |
| `ignore_apps` | `[]` | Apps that never start a recording. |
| `allow_apps` | unset | When set, only these apps start a recording. |
| `ignore_recorders` | `true` | Screen and sound recorders (OBS, Audacity, SimpleScreenRecorder, Kooha, GPU Screen Recorder, vokoscreenNG) never start a recording or keep one going, so a call you also record in OBS ends when the call does. `false` treats them like any other app. An app listed in `allow_apps` always counts. |
| `end_manual` | `true` | A manual recording (`kute record`, or *Record now* in the tray) that a call joins, or that starts during a call, ends with that call like an automatic one. A call counts once its app has had the microphone open for `min_duration`, so a mic check doesn't end it. `false` records until `kute stop`. |

An entry in `ignore_apps` or `allow_apps` matches an app's name (as shown by `kute status` and in
the transcript header) or its binary, ignoring case: `["Firefox"]` or `["firefox"]`.

## `[capture]`

| Key | Default | |
|---|---|---|
| `mic` | the default source | Record this microphone instead: any part of its name, ignoring case (`"USB"`, `"Yeti"`). |

kute always records the default output's monitor as the "Others" track.

## `[transcribe]`

| Key | Default | |
|---|---|---|
| `backend` | `"whisper"` | `whisper` (built in) or `command`: a helper program such as the [Qwen3-ASR helper](../contrib/qwen3-asr/README.md). |
| `command` | | `command` backend: the program and its arguments. A leading `~` in any of them is expanded. |
| `model` | `"large-v3-turbo"` | `whisper` backend: a name from `kute model list`, or a path to a ggml `.bin` file. |
| `language` | `"auto"` | `"auto"` detects the language of each speech chunk, so mixed-language calls work. Set a code such as `"it"` to force one, or several such as `"ru,en"` to detect only among those. A session's own setting (`kute language`, for both tracks or one) wins. |
| `gpu` | unset | `whisper` backend: unset, kute transcribes on a discrete GPU and on the CPU when the only GPU is integrated, which is often slower than a desktop CPU. `true` or `false` forces the GPU or the CPU. `kute doctor --bench` measures which is faster and can set it. |

Every backend uses kute's own speech detection (Silero VAD) to cut the audio into chunks, so
the VAD model is needed either way.

### The `command` backend

kute starts the program once per transcription run, keeps it running while it transcribes,
and stops it afterwards, so a model that is slow to load only loads once and frees the GPU
when kute is done. It sends speech chunks on stdin and reads segments with word timestamps
from stdout, in JSON lines; `src/transcribe/command.rs` describes the protocol. The model
name the program reports is saved as each conversation's `transcriber` in `meta.toml`.

```toml
[transcribe]
backend = "command"
command = ["~/kute-qwen/.venv/bin/python", "~/kute-qwen/kute-qwen3-asr.py"]
```

`language` and `kute language` work as with Whisper. A helper that can't limit detection to a
list, as Qwen3-ASR can't, chooses among the candidates itself (see its README).
`kute retranscribe --model`, `gpu` and `kute doctor --bench` only apply to Whisper; the helper
picks its own device.

## `[summary]`

Summaries are off until you pick a backend. kute summarizes each conversation separately and
writes them to the session's `summary.md`.

| Key | Default | |
|---|---|---|
| `backend` | `"none"` | `none`, `command`, `openai` or `anthropic`. |
| `language` | `"auto"` | Language of the notes. `"auto"` uses the transcript's main language. |
| `min_words` | `80` | Conversations with fewer words aren't summarized. |
| `prompt` | built in | Path to your own prompt template (Markdown). See [the default](../src/summarize/prompt.md). |
| `timeout` | `"10m"` | How long one summary may take. |
| `command` | | `command` backend: the program and its arguments. |
| `base_url` | see below | `openai` and `anthropic` backends: the API's base URL. |
| `model` | see below | `openai` (required) and `anthropic` backends: the model name. |
| `api_key_env` | see below | The name of the environment variable that holds the API key. Never put the key itself in the config. |

kute appends a context block (date, time, app, title, who "Me" is, the language to write in), the
session's notes if it has any (`kute notes`), and the transcript to the prompt template, so a custom template only needs the instructions.

### `command`: any program

The prompt goes to the program's standard input, and whatever it prints is the summary. This
works with any CLI that answers a prompt, for example:

```toml
[summary]
backend = "command"
command = ["claude", "-p"]
```

```toml
[summary]
backend = "command"
command = ["ollama", "run", "qwen3:8b"]
```

### `openai`: OpenAI-compatible endpoints

Any server with an OpenAI-style `/chat/completions` endpoint: Ollama, llama.cpp's server, vLLM,
LM Studio, OpenRouter, OpenAI. `base_url` defaults to Ollama on `http://localhost:11434/v1`.
`api_key_env` is optional; leave it out for a local server without authentication.

```toml
[summary]
backend = "openai"
model = "qwen3:8b"                     # local Ollama, nothing leaves the machine
```

```toml
[summary]
backend = "openai"
base_url = "https://openrouter.ai/api/v1"
model = "some-provider/some-model"
api_key_env = "OPENROUTER_API_KEY"
```

### `anthropic`: the Anthropic API

`api_key_env` defaults to `ANTHROPIC_API_KEY`, and `model` to a current Claude model.

```toml
[summary]
backend = "anthropic"
# model = "..."
```

A daemon started by systemd doesn't inherit your shell's environment: see
[Installing](install.md#running-the-daemon-at-login) for passing the key to it.

## `[processing]`

By default kute transcribes (and summarizes) each conversation as soon as it closes, even while
the call goes on, so a long session is mostly done when it ends. On a machine where that work
is noticeable during a call (a busy CPU, a loud fan next to an open mic), these change when and
how hard the work runs.

| Key | Default | |
|---|---|---|
| `wait_for_recording_end` | `false` | Hold transcriptions and summaries until the recording ends, then run them. Edits (title, notes, others, languages) still apply right away, and a job already running when a recording starts finishes. Jobs queued by hand (`kute retranscribe`) wait too; `kute status` counts them as queued. |
| `low_priority` | `false` | Run the work at the lowest CPU and I/O priority (nice 19, idle I/O class), including a `command` backend or summarizer. Other programs, the call included, get the CPU first. It doesn't make the work smaller: on an otherwise idle machine it still uses the CPU fully, so to keep the fan quiet during a call use `wait_for_recording_end`. |

```toml
[processing]
wait_for_recording_end = true
low_priority = true
```

## `[notify]`

| Key | Default | |
|---|---|---|
| `open` | `["xdg-open"]` | The command that opens `summary.md` when you click a "Summary ready" notification, and the session folder when you click a failure or "Recording stopped" notification. The path is added as the last argument, e.g. `["kate"]` or `["code"]`. |

## `[tray]`

| Key | Default | |
|---|---|---|
| `enabled` | `true` | Show the tray icon. Its menu records (*Record now*), stops (*Stop and keep*, *Stop and delete*), turns call detection off and on, and opens the recordings folder. While a call is recorded, it also sets the call's title, who else is on it, its languages and notes, each in a dialog, which needs `kdialog` or `zenity`. It needs a StatusNotifierItem host: KDE Plasma, most other panels, or the AppIndicator extension on GNOME. |

## A complete example

```toml
[user]
name = "Alex"

[detect]
ignore_apps = ["Discord"]

[transcribe]
model = "large-v3-turbo-q5_0"

[summary]
backend = "openai"
model = "qwen3:8b"
language = "en"

[notify]
open = ["kate"]
```
