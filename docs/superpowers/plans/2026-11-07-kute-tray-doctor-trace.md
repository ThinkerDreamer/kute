# kute plan 4b: tray icon, `kute doctor`, `kute debug trace`

**Spec:** `docs/superpowers/specs/2026-10-08-kute-design.md`, sections 3 (modules), 5 (Recording
state), 8 (`doctor`, `debug trace`) and 10 (detector traces). Follows plan 4a
(`2026-11-06-kute-service-and-notifications.md`).

**Goal:** the daemon is visible in the panel and can be driven from it; a user can check their
setup with one command; and a detection bug can be reported with a trace that is safe to share
and can become a test.

## Tray icon (`src/tray.rs`, `src/daemon.rs`)

- `ksni` 0.3 with its `blocking` API on the `async-io` runtime (no tokio). It shares `zbus` with
  `notify-rust`.
- `view(&Status) -> View` is pure: icon, headline, detail lines, the session Stop/Discard act on.
  - Recording or Grace: `media-record` (a red dot), "Recording: <app>" ("Recording" for a manual
    session), Grace adds "The call seems over; stopping soon".
  - Tentative counts as listening: it may be a mic check (section 5 shows nothing for it).
  - Paused: `media-playback-pause`. Otherwise `audio-input-microphone`, "Listening for calls".
  - Details: the running job, how many are queued, how many wait to retry.
- Item status is `Active` while recording and `Passive` otherwise, so Plasma keeps the idle icon
  in the overflow and brings it out when a call is recorded.
- Menu (left click opens it too): headline and details (disabled), then Stop and keep / Stop and
  delete while recording, or Record now; a "Detect calls" checkmark (pause/resume); Open
  recordings folder (`notify.open` on `sessions/`, with the desktop's display variables, like
  the notification opener).
- Menu choices go to the daemon as `Event::Tray(TrayAction)`. Stop and Discard carry the session
  the menu showed and are ignored if another one is recording by then, like notification buttons
  (`Daemon::act` serves both).
- The daemon loop sends the tray the status after each event, only when it changed.
- The service starts before the desktop, so the tray is spawned with `assume_sni_available`: no
  watcher yet is not an error, and the icon registers when the panel starts. Only a missing
  session bus disables it (logged).
- `[tray] enabled = true` turns it off.

## `kute doctor` (`src/doctor.rs`)

Runs before the config is loaded (a broken config is one of its findings) and prints
`ok`/`WARN`/`FAIL` lines; exits non-zero on a failure.

- config loads; storage folder writable, warn under 2 GB free;
- `pw-dump`, `pw-record`, `ffmpeg`, `ffprobe` (fail), `xdg-open` (warn), on the shell's `PATH`
  and, when `kute.service` exists, on the `PATH` in its `Environment=` line;
- PipeWire: one `pw-dump`, a default source and sink, `capture.mic` matching a source;
- GPU: the compiled backend, `transcribe.gpu`, and a Vulkan ICD file or `/dev/nvidiactl`;
- the Whisper model and the VAD model downloaded (both needed to transcribe);
- summarizer: the command on `PATH`; for HTTP backends the API key (in `~/.config/kute/env`
  for the service; only in the shell is a warning) and any HTTP answer from `base_url`;
- daemon answering on the socket, service active; notification server; a
  `org.kde.StatusNotifierWatcher` on the bus (via `busctl`).

## `kute debug trace` / `kute debug replay` (`src/pw/trace.rs`)

- JSON lines: a header `{"kute_trace": 1, "kute_version": ...}`, then
  `{"t_ms": ..., "objects": [...]}` per `pw-dump` update. `pw::read_arrays` and
  `pw::spawn_pw_dump` are shared with the monitor.
- `Scrubber`: learns secrets from each update before replacing (a node may come before its
  device): `device.serial`, Bluetooth addresses, bus paths, process host/user/machine-id,
  security labels, plus `$USER` and the host name (3+ characters). Each secret gets a stable
  placeholder (`SERIAL-1`), so scrubbed node names still tell devices apart. `$HOME` becomes `~`,
  MAC-shaped text becomes zeros.
- Writes are flushed per line, so Ctrl-C leaves a complete file. `--duration` stops it;
  `--out -` writes to stdout. Stream changes are described on stderr as they happen.
- `trace::snapshots` replays entries into snapshots; `detect::replay` feeds them to a
  `Detector` with a 500 ms tick, as the daemon does. `kute debug replay` prints both with the
  config's `[detect]` settings. `tests/traces.rs` runs the fixtures in `tests/fixtures/traces/`.

## Not done

Real traces from a development machine (only two hand-written ones), recording time in the
tray, loading the model in `doctor`: see `docs/TODO.md`.
