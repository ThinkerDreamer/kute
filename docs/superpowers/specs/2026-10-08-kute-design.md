# kute: design

*kute* is Toki Pona for "listen". kute records calls on a Linux desktop, transcribes them
locally, and optionally summarizes them. It grew out of two shell scripts shared by a
colleague (`record-call`, `transcribe-call`), which recorded the microphone and the system output
as two separate tracks with ffmpeg and merged two Whisper transcripts into one timeline.

Status: approved design, 2026-10-08.

## 1. Goals

- **Never miss a call.** A background daemon detects calls and records them without a
  deliberate action, while staying visible (tray icon, notifications).
- **Meeting notes.** Every conversation gets a summary with decisions and action items.
- **A searchable archive.** Find what was said, by whom (me / others) and when, across
  all past calls, and jump to the moment.
- **Local by default.** Audio never leaves the machine. Transcription runs on the local
  GPU. Summarization goes wherever the user configures, including fully local.
- **Open-source quality.** One binary, documented config, tests, MIT/Apache-2.0.

### Non-goals for v1

Each of these is a planned improvement; v1's boundaries are drawn so they fit later
without restructuring.

- Web UI (files are the source of truth; a UI will be a view on them).
- Rolling "keep the last N minutes" buffer.
- Speaker diarization of the Others track (v1 labels are Me / Others).
- Capturing only the call app's output instead of the whole default sink monitor.
- Remote transcription backends (the `Transcriber` trait exists; only whisper.cpp ships).
  Planned right after the offline pipeline: one provider-agnostic remote backend layer with
  Deepgram and ElevenLabs as the first two providers, selected in config. Strictly opt-in,
  because it sends audio off the machine.
- Echo cancellation (v1 assumes headphones; with speakers, the mic also picks up the
  other side).
- Live captions, macOS, Windows.

## 2. Scenarios

1. **1:1 call.** The user joins a Meet call in Firefox. kute starts recording when Firefox
   opens the mic, commits after 20 s, stops 15 s after the call ends, transcribes,
   summarizes and compresses. One session, one conversation.
2. **Coworking day.** The user's team sits in a Jitsi room for hours, mostly silent and
   mostly muted (via the KDE Plasma mic toggle, which mutes the PipeWire source). A
   short chat at 09:40, a ticket discussion at 11:00, another chat at 16:30. kute keeps
   one session for the whole day and produces three conversations, each transcribed and
   summarized shortly after it ends, while the session is still recording.
3. **Import.** Recordings in the `record-call` layout (`<prefix>-me.wav`,
   `<prefix>-others.wav`, `<prefix>.start`) are imported with `kute import` and processed like a recorded session.
4. **False alarm.** A website asks for the mic for 5 seconds. kute captures tentatively,
   then deletes the capture without notifying.

## 3. Architecture

One Rust binary, `kute`. `kute daemon` runs as a systemd `--user` service; every other
subcommand is a client.

- **Control commands** (`status`, `record`, `stop`, `discard`, `pause`, `resume`) talk to
  the daemon over a Unix socket at `$XDG_RUNTIME_DIR/kute.sock` (mode 0600), using
  newline-delimited JSON requests and responses.
- **Archive commands** (`list`, `show`, `search`, `title`, `open`) read the files and the
  index directly and work without the daemon.
- **Processing commands** (`summarize`, `retranscribe`, `import`, `reindex`) enqueue a job
  on the daemon if it is running, and otherwise run in the foreground.

### Threads (no async runtime)

```
 PipeWire thread ──graph events──▶ Detector ──start/stop──▶ Recorder ──▶ WAV files
   (registry watch,                    ▲                       │ live VAD frames
    capture streams)                   │                       ▼
 IPC thread ──commands──▶ Daemon state (owner of all mutable state)
 Tray thread ◀──status──┘              │ conversation closed / session ended
                                       ▼
                               Job queue, 1 worker:
             transcribe ─▶ merge ─▶ index ─▶ summarize   (per conversation)
             compress                                     (per session)
```

PipeWire's main loop runs on its own thread and talks to the rest through
`std::sync::mpsc` channels. Whisper is blocking GPU work, and the socket handles a few
requests an hour, so plain threads are enough. A single worker means transcription and
summarization never share the GPU at the same time (consumer GPUs with 8 GB of VRAM).

### Modules (one crate for v1)

| Module | Responsibility |
|---|---|
| `pw` | PipeWire graph watching, capture streams; emits `GraphEvent`s through a trait so the detector can be fed recorded traces |
| `detect` | Pure state machine: graph events + clock in, session decisions out |
| `recorder` | Writes the two WAVs, rewrites WAV headers periodically, runs live VAD, emits conversation boundaries |
| `queue` | Job queue and the single worker; rebuilt from `meta.toml` at startup |
| `transcribe` | `Transcriber` trait; whisper.cpp implementation via `whisper-rs` |
| `merge` | Turns per-track Whisper output into one timeline (port of the `transcribe-call` merge) |
| `summarize` | `Summarizer` trait; `openai`, `anthropic`, `command` backends |
| `store` | Session folders, `meta.toml`, atomic writes |
| `index` | SQLite FTS5 index, derived from the files |
| `ipc` | Socket protocol (server and client) |
| `tray` | StatusNotifierItem tray icon (`ksni`) and notifications (`notify-rust`) |
| `config` | Config loading and defaults |
| `cli` | Subcommands (`clap`) |

## 4. Storage

Default root `~/.local/share/kute/` (`$XDG_DATA_HOME/kute`), configurable. Created with
mode 0700, since it holds recordings of other people.

```
~/.local/share/kute/
├── sessions/
│   └── 2026-10-08T090312/          ← session ID: local start time, never changes
│       ├── meta.toml
│       ├── me.wav  others.wav      ← while recording and processing
│       ├── me.opus others.opus     ← after compression (WAVs deleted)
│       ├── me.json others.json     ← Whisper segments + words, all conversations
│       ├── transcript.md           ← derived
│       ├── summaries/01.md 02.md   ← one per summarized conversation
│       └── summary.md              ← derived: concatenation of summaries/
├── models/                         ← Whisper and Silero VAD model files
└── index.sqlite                    ← derived, rebuildable with `kute reindex`
```

- If two sessions would get the same ID, the second gets a `-2` suffix.
- `meta.toml`, JSON, transcripts and summaries are written to a temp file in the same
  folder and renamed into place.
- `transcript.md` and `summary.md` are derived and regenerated; they are not meant to be
  hand-edited. `meta.toml`'s `title` is the one user-editable field (via `kute title`, or
  by hand).

### Time model

- **Session time** `t = 0` is the wall-clock instant of the earliest first sample across
  both tracks, stored as `started_at` (RFC 9557, i.e. RFC 3339 plus the IANA time zone name, millisecond
  precision; times are `jiff::Zoned` so wall-clock arithmetic stays correct across DST).
- Each track stores `offset_ms`: how much later than `t = 0` its first sample arrived.
  `track_time = session_time - offset_ms`.
- Everything in `meta.toml` and `*.json` is in session-time milliseconds. Wall-clock times
  in `transcript.md` are `started_at + session_time`.

### `meta.toml`

```toml
id = "2026-10-08T090312"
title = "Coworking"                     # optional
kute_version = "0.1.0"
started_at = "2026-10-08T09:03:12.418+02:00[Europe/Rome]"
ended_at = "2026-10-08T17:41:05.002+02:00[Europe/Rome]"
state = "processing"                    # recording | processing | done
interrupted = false                     # true if recovered after a crash

[trigger]
kind = "app"                            # app | manual | import
app_name = "Firefox"
binary = "firefox"

[tracks.me]
device = "alsa_input.usb-Generic_USB_Microphone-00.mono-fallback"
offset_ms = 0
[tracks.others]
device = "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
offset_ms = 37

[summary]
language = "en"                         # resolved at session start; `--lang` overrides and persists

[[conversations]]
n = 1
start_ms = 2_211_000
end_ms = 3_034_500
stage = "summarized"                    # pending | transcribed | summarized | skipped | failed
languages = { me = "en", others = "it" }
whisper_model = "large-v3-turbo"
summarizer = "command: claude -p"
# failed_step = "transcribe"             # present when stage = "failed": transcribe | summarize
# error = "..."                         # present when stage = "failed"
# attempts = 2

[compression]
stage = "pending"                       # pending | done | failed
```

### `transcript.md`

```
# Coworking. 2026-10-08 09:03 (8 h 37 min, Firefox)

## 09:40 to 09:54

[09:40:11] Me: Ciao! Hai visto la MR?
[09:40:15] Others: Sì, I left a couple of comments on the patch format.
```

Every conversation gets a `##` heading, even when there is only one. The labels come from
config (`labels.me`, `labels.others`, default `Me` / `Others`). The format leaves room for
later per-speaker labels on the others side (`Others (A)`).

## 5. Detection and capture

### Definitions

- **Mic stream:** a node with `media.class = Stream/Input/Audio` linked to a node with
  `media.class = Audio/Source` (a real capture source, not a sink monitor).
- **Ignored stream:** belongs to kute's own PID; or is a peak-meter / level-monitoring
  stream, marked by `stream.monitor = true` (the Plasma volume applet's
  `PlasmaPA-VolumeMeter` streams, confirmed in the spike; see section 12); or its app
  matches `detect.ignore_apps`; or `detect.allow_apps` is set and the app is not in it.
  An app matches a rule if its `application.process.binary` or its `application.name` does,
  case-insensitively. Labels (tray, `meta.toml`) use `application.name` when set, since
  Electron apps all report the binary `electron` (the Jitsi Meet app is
  `application.name = "Jitsi Meet"`, binary `electron`).
- **Trigger app:** the app of the first non-ignored mic stream of a session.
- **Session alive:** any non-ignored mic stream exists, **or** one of the session's
  *bound output nodes* still exists. A bound output node is a `Stream/Output/Audio` node of
  the trigger app (same binary and `application.name`) that was present at some moment
  while a mic stream was alive, and it counts only while its state is `running`: the Jitsi
  Meet app's output stream stayed `running` through silence and went `idle` the moment the
  user left the room (spike, section 12). The second clause keeps a session alive if an app closes its
  mic on mute. It is bound to specific nodes, not to the app, so a YouTube video started in the same
  Firefox after leaving the call opens a new node and does not keep the session alive.
  Nodes are tracked by `object.serial`, never by node ID: PipeWire reuses IDs within
  seconds (the spike saw ID 85 go from Jitsi's mic to a Plasma meter).
- **Short gaps don't count:** a mic stream that closes and is replaced within 2 s keeps the
  session (and a Tentative capture) alive. The Jitsi Meet app closes its pre-join mic
  stream and opens a new one on joining.

### State machine

```
            mic stream appears                 alive for min_duration
  Idle ──────────────────────────▶ Tentative ─────────────────────────▶ Recording
   ▲                                  │                                   │   ▲
   │        not alive (any time)      │               not alive           │   │ alive again
   └──────── delete capture ◀─────────┘                                   ▼   │
   │                                                                     Grace
   └──────────────── finalize, enqueue ◀──────── stop_grace elapsed ──────┘
```

- **Tentative:** capture is running and files are written, but nothing is shown to the
  user. When it ends, the folder is deleted.
- **Recording:** the tray icon turns red and a notification "Recording: Firefox" offers
  Stop / Discard.
- **Grace:** recording continues. If the session becomes alive again, back to Recording,
  same session.
- **Paused** (`kute pause`): the detector ignores all streams until `kute resume`. Pausing
  during an automatic session ends it (Tentative is deleted, Recording or Grace is
  finalized), since pausing means "stop listening"; a manual session continues. The tray
  shows the paused state. A manual `kute record` still works while paused.
- **Manual:** `kute record [title]` starts a session in Recording state with
  `trigger.kind = "manual"`; it ends only with `kute stop` (no grace logic). If a session
  is already Recording or in Grace, `record` fails with "already recording <id>" (use
  `kute title` to name it); if one is Tentative, it is committed and turned manual. `kute stop`
  also ends an automatic session immediately. `kute discard` stops and deletes the
  current session, and removes anything already indexed from it.
- **After `stop` or `discard`:** the same call is not picked up again. Automatic detection
  stays blocked until no mic stream has been seen for 2 s (snapshots, not ticks, count: time
  without a graph never lifts the block). `kute record` still works.
- **Sessions the daemon ends itself** (a capture died, the graph was lost) do not block: if
  the call's mic is still open, a new session starts. The detector looks for a call on every
  graph snapshot and on every tick (500 ms) while Idle, since the graph may have settled
  before the session ended (seen when WirePlumber restarted mid-call). A capture that dies
  within 5 s of starting does block, so a broken `pw-record` can't create an empty session
  every tick.

The detector is a pure function of (state, event, now) and is tested without PipeWire.

### Capture

- **Me:** the default source, without a fixed target, so WirePlumber moves the stream when
  the default changes. Muting the source (KDE Plasma mic toggle, `pactl set-source-mute`)
  silences kute too; the README tells users to mute at the system level, not only in the
  call app, to keep speech out of kute. `capture.mic` pins a device by name substring; the
  docs note that the Plasma toggle then no longer reaches kute.
- **Others:** the default sink's monitor via `stream.capture.sink = true`, which also
  follows default-sink changes.
- Format requested from PipeWire: 16 kHz, mono, S16LE. WAV written with `hound`; the
  header is rewritten every 5 s so a crash leaves a valid file.
- Each track records the wall-clock time of its first sample. The first track to deliver a
  sample sets `started_at` (session `t = 0`); the other track's `offset_ms` is how much later
  its first sample arrived.
- Capture uses PipeWire's own tools, not bindings: `pw-dump --monitor` streams graph changes
  as JSON, and one `pw-record --raw` process per track writes 16 kHz mono S16LE PCM to stdout
  (without `--raw`, pw-record 1.6 writes a 24-byte AU header first).
  kute's capture streams are named `kute-me` and `kute-others` (`node.name`), and the
  detector ignores any stream whose `node.name` starts with `kute-`.
- If the source itself stops delivering (a device switch, an xrun), the recorder pads the gap
  with silence, so track time stays aligned with the wall clock. It pads only for time a
  single read blocked beyond what it returned, by more than 500 ms, capped so the track never
  runs ahead of the wall clock. A fast read draining a backlog never pads: comparing total
  elapsed time with samples written can't tell a device stall from the recorder itself
  falling behind (a slow VAD block), and padding then would shift the track for good.

### Conversation splitting (live)

A lightweight VAD runs on both tracks as frames arrive. A conversation opens at the first
speech frame after a gap and closes after `detect.conversation_gap` (default 5 min) with no
speech on either track, or when the session ends. Its range is padded by 2 s on each side
(clamped to the session). Precision is not critical here. The VAD is Silero via
`whisper-rs` (`WhisperVadContext`, CPU), run on each track in 10 s blocks as audio arrives.
A conversation closes only once both tracks have been analysed past the gap. The open
conversation's start is kept in `meta.toml` as `open_conversation_ms` so crash recovery can
close it. If the VAD model is missing, live splitting is off and the whole session becomes
one conversation when it ends.

A session with no speech at all keeps its audio and metadata and has zero conversations.

### Crash recovery

At daemon start, every session with `state = "recording"` gets its WAV headers repaired from
the file length, is marked `interrupted = true`, has any open conversation closed at the
end of the audio, and moves to `processing`.

## 6. Processing pipeline

**The folder is the queue.** All job state lives in `meta.toml`. At daemon start, the queue
is rebuilt from sessions in `processing` state, in session order, then conversation order.

### Per conversation

1. **Transcribe**, per track: read the conversation's range from the WAV (or decode the
   Opus via ffmpeg for `retranscribe`), shifted by the track's `offset_ms`. kute runs Silero
   VAD itself (`WhisperVadContext`), pads speech regions by 200 ms, joins those less than
   2 s apart into chunks of at most 120 s, and runs Whisper with token timestamps on each
   chunk, so Whisper never sees long silences. (whisper.cpp's built-in VAD option is not
   used: it remaps segment times to the original audio but not token times, so word
   timestamps would drift.) Language: `transcribe.language`, or with `auto`, detected per
   chunk for chunks of at least 3 s; shorter chunks take the language of the nearest long
   chunk, and if there is none, one detection runs on the first 30 s of speech. The track's
   language for the conversation is the one covering the most chunk time. Segment and word
   times are converted to session time and stored in `me.json` / `others.json` tagged with
   `conversation = n` (replacing any earlier entries for `n`). Audio is read in 10-minute
   VAD windows and then one chunk at a time, so memory stays bounded on multi-hour
   recordings.
2. **Merge:** for each track, for each segment:
   - no words: one line at the segment start with the segment text;
   - otherwise split the words into chunks wherever `next.start - prev.end > 1.0 s`;
   - if there is more than one chunk and the first chunk is a single word, move that word
     into the second chunk, taking the second chunk's start time (Whisper often pins a
     segment's first word to the segment start);
   - each chunk becomes a line `(start, label, joined words, trimmed)`.

   Sort all lines from both tracks by start time; regenerate `transcript.md` for all
   conversations.
3. **Index:** replace the conversation's rows in the index.
4. **Summarize** if a summarizer is configured and the conversation has at least
   `summary.min_words` words; otherwise `stage = "skipped"`. Write `summaries/NN.md`,
   regenerate `summary.md`, re-index the summary.

### Per session

After the session has ended and every conversation is `summarized`, `skipped`, or failed at
summarizing (a failed *transcription* keeps the WAVs, so its retry reads lossless audio):
**compress** both WAVs to Opus with ffmpeg (`libopus`, 24 kbit/s, `-application voip`),
verify with ffprobe that each Opus file's duration matches its WAV within 0.5 s, then delete
the WAVs and set `state = "done"`.

### Summarization

When the daemon writes summaries, a desktop notification "Summary ready: <title or app>"
shows the conversation's time range and the summary's first line; clicking it opens the
session's `summary.md` with `notify.open` (default `xdg-open`). Several summaries from one
job make one notification.

Input: the prompt template, then context (date, wall-clock range, trigger app, title,
`user.name` as the speaker behind the Me label, target language), then the conversation's
transcript section.

Default prompt rules:
- write in the target language (`summary.language`; `auto` = dominant language of the
  transcript);
- keep names, technical terms, identifiers, ticket/MR references as spoken, never
  translated;
- short quotes stay in their original language when the wording matters;
- sections: Summary, Decisions, Action items (`who: what`), Open questions; omit empty
  sections.

`summary.prompt` points to a custom template file.

Backends:

| Backend | Config | Notes |
|---|---|---|
| `openai` | `base_url`, `model`, `api_key_env` | Chat Completions; covers Ollama, llama.cpp, vLLM, OpenRouter |
| `anthropic` | `model`, `api_key_env` (default `ANTHROPIC_API_KEY`) | Messages API |
| `command` | `command` (argv list), `timeout` (default 10 min) | full prompt on stdin, stdout = summary; non-zero exit = failure with stderr captured |
| `none` | | summarization off |

`kute summarize <id> [--conversation n] [--lang xx] [--backend name]` re-runs it; `--lang`
is persisted to `meta.toml`.

### GPU and models

- Whisper model loads on the first transcription job and is dropped as soon as the next
  job in the queue is not a transcription (or the queue is empty).
- Default model `large-v3-turbo` (ggml). `kute model download [name]` fetches Whisper and
  the Silero VAD model from the whisper.cpp Hugging Face repository into `models/`, with
  checksum verification.
- Cargo features: `vulkan` (default), `cuda`; CPU fallback when neither is available.

### Errors

| Failure | Behavior |
|---|---|
| Transcription (e.g. GPU out of memory) | `stage = "failed"`, `error` set, retried with backoff (1, 5, 30 min), then left failed; notification after the last attempt |
| Summarization | `stage = "failed"`, notification, no automatic retry; `kute summarize` re-runs |
| Compression | `compression.stage = "failed"`, WAVs kept, notification; retried at next daemon start |
| PipeWire disconnect | the graph counts as empty: an automatic session ends as if the call had ended (Tentative after 2 s, Recording after `stop_grace`) unless the graph comes back with the call's mic in time; a manual session continues. Reconnect with backoff (1, 2, 5, 10, then 30 s) |
| A capture dies (`pw-record` exits) | end the session, keep what was written; a call still in progress starts a new session right away |
| Disk full while recording | stop the session, keep what was written, notification |

No failure path deletes audio, except `kute discard` and a Tentative session ending.

## 7. Config

`~/.config/kute/config.toml`. Every key has a default; a missing file is valid.

```toml
[user]
name = ""                       # used in summary prompts, e.g. "Alex"

[labels]
me = "Me"
others = "Others"

[storage]
dir = "~/.local/share/kute"

[detect]
min_duration = "20s"
stop_grace = "15s"
conversation_gap = "5m"
ignore_apps = []
# allow_apps = []

[capture]
# mic = "USB"                  # substring of the device name

[notify]
open = ["xdg-open"]             # opens summary.md when a "Summary ready" notification is clicked

[transcribe]
model = "large-v3-turbo"
language = "auto"

[summary]
backend = "none"                # none | openai | anthropic | command
language = "auto"
min_words = 80
# prompt = "~/.config/kute/prompt.md"
# command = ["claude", "-p"]
# timeout = "10m"
# base_url = "http://localhost:11434/v1"
# model = "qwen3:8b"
# api_key_env = "OPENAI_API_KEY"
```

## 8. CLI

| Command | Purpose |
|---|---|
| `daemon` | run the daemon (normally via systemd) |
| `status` | daemon state, current session, queue |
| `record [title]` / `stop` / `discard` | manual control |
| `pause` / `resume` | turn auto-detection off / on |
| `list [--since 7d]` | sessions, newest first, with title, duration, app, conversation count |
| `show <id> [--summary]` | print transcript or summary |
| `search <query>` | FTS over transcripts and summaries; results show session, wall-clock time, speaker, line |
| `title <id> <title>` | set the title |
| `open <id>` | open the session folder |
| `summarize <id> [...]` | (re)summarize |
| `retranscribe <id> [--model m]` | transcribe again from the stored audio |
| `import <me.wav> <others.wav> [--start time] [--title t] [--no-process]` | import external recordings; `--start` defaults to a sibling `.start` file (the `record-call` layout), then file birth time. Until the daemon's conversation splitting exists, an import is one conversation |
| `process <id>` | finish whatever is outstanding for a session (transcription, summaries, compression), e.g. after a failure |
| `reindex` | rebuild `index.sqlite` from the files |
| `model download [name]` / `model list` | manage models |
| `doctor` | check PipeWire, ffmpeg/ffprobe, GPU backend, models, summarizer reachability (command on `PATH`, URL responds) |
| `service install` | write and enable the systemd user unit, with a `PATH` that includes `~/.local/bin` and `~/.cargo/bin` |
| `debug trace [--out file]` | record PipeWire graph events to a file for bug reports and test fixtures |

`<id>` accepts a full ID, a unique prefix, or `last`.

## 9. Index

SQLite in WAL mode, with one writer at a time. While the daemon is running, every command
that changes a session or the index (`title`, `summarize`, `retranscribe`, `import`,
`reindex`) is sent to it and the worker performs the write; read-only commands open
read-only connections. Without the daemon, the CLI writes directly.

- `sessions(id, title, started_at, ended_at, app, conversation_count)`
- `lines` (FTS5): `session_id, conversation, t_ms, speaker, text`
- `summaries` (FTS5): `session_id, conversation, text`

## 10. Testing

- **Unit, no audio or PipeWire:** merge (fixtures of real Whisper JSON; the 1 s split and
  lone-first-word cases), detector (event sequences + fake clock), conversation splitting
  (VAD frame sequences), queue rebuild from `meta.toml`, crash recovery on truncated WAVs,
  time model conversions, config parsing and defaults.
- **Summarizers:** `command` with a fake script (success, non-zero exit, timeout);
  `openai` and `anthropic` against a local mock HTTP server.
- **Detector traces:** real event sequences recorded with `kute debug trace` on a development
  machine (Jitsi coworking with Plasma mute, a Meet call, a reconnect, pavucontrol open, a
  website mic check) replayed as fixtures. `kute debug trace` scrubs device serials,
  hostnames, user names and paths under `$HOME` before writing, so traces are safe to
  commit and to attach to bug reports.
- **Pipeline integration:** `kute import` on short synthesized speech clips (Piper or
  espeak, so the repo holds nobody's voice) with the `tiny` Whisper model, run in CI with
  the model cached.
- **Manual acceptance:** a checklist for scenarios 1–4.

## 11. Open source

- License: MIT OR Apache-2.0.
- Name `kute` is free on crates.io and the AUR (checked 2026-10-08).
- README: install, `kute doctor`, config, the "mute at the system level" rule, and a
  section on consent and recording laws: kute does not announce itself to other
  participants, laws differ (one-party vs all-party consent), and compliance is the user's
  responsibility.
- Ships a systemd user unit; AUR package once stable.

## 12. Spike results (2026-10-19)

Checked on the development machine (KDE Plasma on Wayland, NVIDIA GPU, PipeWire) with
the Jitsi Meet desktop app, which is Electron (Chromium), rather than Firefox. Method: a
script logged every change to PipeWire audio streams, their links and the default source's
mute flag while the user went through a call.

1. **Mute and silence (Jitsi Meet app): the mic stays open.** Muting in the Jitsi UI
   changed nothing PipeWire can see: the mic stream stayed `running`, not corked or muted.
   One output stream stayed `running` through 30 s of silence on both sides. So muted or
   quiet stretches don't split a session, and anything said while muted in Jitsi is
   recorded on the Me track. Other details:
   - The pre-join screen opens its own mic stream; joining closes it and opens a new one
     (hence the 2 s gap rule in section 5).
   - On joining, the app also creates several short-lived output streams (sounds), gone
     within ~5 s. Only one stays.
   - Leaving the room closes the mic stream at once and idles the output stream; closing
     the app removes it.
   - A Plasma (system) mute shows only as the default source's mute flag; the app's
     stream is untouched. Detect it from the source, not the stream.
   - The app's streams have `application.name = "Jitsi Meet"` but binary `electron`, so
     the binary alone can't tell Electron apps apart (section 5 matches on either).
   - Firefox was not tested; check it if someone uses Jitsi or Meet in a browser.
2. **Peak meters: `stream.monitor = true`.** The Plasma volume applet opens
   `Stream/Input/Audio` nodes named `PlasmaPA-VolumeMeter` (binary `plasmashell`, empty
   `application.name`), some linked to real sources and one to Jitsi's output stream, and
   churns through ~10 of them in a few seconds while open. pavucontrol was not tested.
3. ~~Does `whisper-rs` expose whisper.cpp's VAD API for live use?~~ Yes: whisper-rs 0.16 has
   `WhisperVadContext::segments_from_samples` (checked 2026-10-08), so live splitting uses Silero.
4. **`stream.capture.sink = true` follows the default sink.** A `pw-record` stream with it
   moved from the USB headset's monitor to HDMI and back as the default sink changed,
   without reconnecting.
5. **The `vulkan` feature builds and runs on the NVIDIA GPU** (checked 2026-10-08).
