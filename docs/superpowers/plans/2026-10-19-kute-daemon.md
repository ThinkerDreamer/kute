# kute Daemon Implementation Plan (plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `kute daemon` notices when an app opens the microphone, records both tracks, splits the recording into conversations live, and hands each finished conversation to a single background worker that runs the existing offline pipeline. `kute status / record / stop / discard / pause / resume` control it over a Unix socket.

**Architecture:** Plain threads and `std::sync::mpsc`, no async runtime. A monitor thread runs `pw-dump --monitor` and turns its JSON into a `Graph`; the daemon thread owns all mutable state and feeds stream snapshots and clock ticks to a pure `Detector` state machine. Its actions start and stop a `LiveSession`, which runs one `pw-record` process and one reader thread per track: the reader writes the WAV, runs Silero VAD on 10 s blocks, and sends events on the live session's own channel; the daemon drains it on every tick into a pure `Splitter` that opens and closes conversations. Closed conversations become `Job`s for a single worker thread, which runs `pipeline::Pipeline` (the same code the CLI uses) with retry backoff. An IPC thread serves newline-delimited JSON on `$XDG_RUNTIME_DIR/kute.sock`.

**Tech Stack:** Rust 2024 (MSRV 1.88; `std::io::pipe` is used in tests), existing crates (hound, serde, serde_json, jiff, whisper-rs, rusqlite, clap), plus `libc` (new, only for `libc::kill(-pgid, SIGKILL)` in Task 3). Runtime tools: `pw-dump` and `pw-record` (PipeWire 1.x), ffmpeg/ffprobe as before.

**Spec:** `docs/superpowers/specs/2026-10-08-kute-design.md`, sections 3, 5, 6 (Errors, GPU and models), 8 (control commands), 9 and 12. Read section 5 and section 12 (the PipeWire spike results) before starting. Tray, notifications, `doctor`, `service install` and `debug trace` are plan 4.

## Global Constraints

Everything in plan 1's Global Constraints still holds (`docs/superpowers/plans/2026-10-08-kute-offline-pipeline.md`). In addition:

- No async runtime. Threads + `std::sync::mpsc` only.
- PipeWire is reached only through `pw-dump --monitor --no-colors` and `pw-record`. No PipeWire bindings, no `unsafe` outside the one `libc::kill` call in Task 3.
- Capture: `pw-record --rate 16000 --channels 1 --format s16 -` (raw PCM on stdout). Me: default source, or `--target <node.name>` when `capture.mic` matches a source. Others: `-P '{ stream.capture.sink = true ... }'`. Stream `node.name`s are `kute-me` and `kute-others`; any stream whose `node.name` starts with `kute-` is ignored by detection.
- Graph objects are tracked by `object.serial` (u64). Node IDs are only used to resolve links within one graph snapshot.
- Detection defaults: `detect.min_duration = "20s"`, `detect.stop_grace = "15s"`, `detect.conversation_gap = "5m"`; mic gap tolerance 2 s (`MIC_GAP_MS = 2_000`, not configurable); conversation padding 2 s (`PAD_MS = 2_000`).
- WAV header rewritten every 5 s of audio (`FLUSH_EVERY = 80_000` samples). Gap fill threshold 500 ms.
- Live VAD block: 10 s (`VAD_BLOCK_MS = 10_000`), Silero on CPU.
- Transcription retry backoff: 1, 5, 30 minutes after attempts 1, 2, 3; after the 4th failed attempt the conversation is left failed.
- Socket: `$XDG_RUNTIME_DIR/kute.sock`, mode `0600`, one JSON object per line each way. Missing `XDG_RUNTIME_DIR` is an error ("XDG_RUNTIME_DIR is not set; is this a desktop session?").
- Every write to `meta.toml` from the daemon or the pipeline goes through `Session::update_meta` (Task 1), never a load-then-`save_meta` across a long operation.
- Logging is `eprintln!("kute: ...")` (journald picks up stderr). Notifications are plan 4.

## Review Focus

1. **kute hearing itself.** kute's own `pw-record` streams, and Plasma's peak meters attached to them, must never start or extend a session; otherwise one call becomes an endless recording. Pinned by `own_capture_streams_are_ignored` and `peak_meters_never_start_a_session` in Task 5.
2. **Two writers on one `meta.toml`.** The worker saves conversation 1 while the live session appends conversation 2; neither write may be lost. Pinned by `update_meta_serializes_concurrent_writers` (Task 1) and `process_keeps_conversations_added_while_it_runs` (Task 2).
3. **No VAD model at record time.** Recording must still work, and the session must end with exactly one conversation covering the audio, never zero. Pinned by `without_live_vad_the_session_is_one_conversation` in Task 8.
4. **A track that stops delivering samples for a while** (default device switched, an xrun): later speech must keep its wall-clock time. Pinned by `gap_fill_keeps_track_time_aligned` in Task 7.
5. **The daemon killed mid-call** (crash, reboot, `systemctl stop` timeout): at next start the WAVs are readable, the open conversation is closed at the end of the audio, and the session is processed. Pinned by `recover_repairs_an_interrupted_session` in Task 8.

---

### Task 1: Config, meta fields and safe meta updates

**Files:**
- Modify: `src/config.rs`, `src/store/meta.rs`, `src/store/mod.rs`

**Interfaces:**
- Produces:
  - `config::DetectConfig { min_duration: SignedDuration, stop_grace: SignedDuration, conversation_gap: SignedDuration, ignore_apps: Vec<String>, allow_apps: Option<Vec<String>> }` as `Config.detect`
  - `config::CaptureConfig { mic: Option<String> }` as `Config.capture`
  - `Meta.open_conversation_ms: Option<u64>` (`#[serde(default, skip_serializing_if = "Option::is_none")]`)
  - `Meta::app_label` now prefers `trigger.app_name`, then `binary`
  - `Session::update_meta(&self, f: impl FnOnce(&mut Meta)) -> Result<Meta>`: load, apply, `save_meta`, all under one process-wide `static META_LOCK: Mutex<()>`; returns the saved meta
  - `Session::delete(self) -> Result<()>`: `remove_dir_all` of the session folder

- [ ] **Step 1: Write the failing tests**

In `src/config.rs` tests:
```rust
#[test]
fn detect_and_capture_defaults() {
    let c = Config::parse("").unwrap();
    assert_eq!(c.detect.min_duration, SignedDuration::from_secs(20));
    assert_eq!(c.detect.stop_grace, SignedDuration::from_secs(15));
    assert_eq!(c.detect.conversation_gap, SignedDuration::from_mins(5));
    assert!(c.detect.ignore_apps.is_empty() && c.detect.allow_apps.is_none());
    assert!(c.capture.mic.is_none());
}

#[test]
fn detect_values_parse() {
    let c = Config::parse(
        "[detect]\nmin_duration = \"30s\"\nconversation_gap = \"2m\"\nallow_apps = [\"Jitsi Meet\"]\n[capture]\nmic = \"USB\"\n",
    )
    .unwrap();
    assert_eq!(c.detect.min_duration, SignedDuration::from_secs(30));
    assert_eq!(c.detect.conversation_gap, SignedDuration::from_mins(2));
    assert_eq!(c.detect.allow_apps.as_deref(), Some(&["Jitsi Meet".to_string()][..]));
    assert_eq!(c.capture.mic.as_deref(), Some("USB"));
}
```

In `src/store/meta.rs` tests: `app_label_prefers_app_name` (trigger `App` with `app_name = "Jitsi Meet"`, `binary = "electron"` gives `"Jitsi Meet"`; with no `app_name` gives the binary) and `open_conversation_ms_round_trips` (set `Some(12_000)`, serialize to TOML, the text contains `open_conversation_ms = 12000`, deserialize equal; `None` is not written).

In `src/store/mod.rs` tests:
```rust
#[test]
fn update_meta_serializes_concurrent_writers() {
    // Two threads each append 50 conversations through update_meta; none may be lost.
    let (_tmp, session) = session_with_meta(); // helper: Store in a tempdir + saved Meta
    std::thread::scope(|s| {
        for t in 0..2u32 {
            let session = &session;
            s.spawn(move || {
                for i in 0..50u32 {
                    session
                        .update_meta(|m| m.conversations.push(Conversation::new(t * 100 + i, 0, 1)))
                        .unwrap();
                }
            });
        }
    });
    assert_eq!(session.load_meta().unwrap().conversations.len(), 100);
}

#[test]
fn delete_removes_the_folder() { /* create_session, save meta, delete, folder gone */ }
```

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --lib --no-default-features -- config:: store::`
Expected: compile errors (`detect`, `update_meta`, `open_conversation_ms` unknown).

- [ ] **Step 3: Implement**

`DetectConfig`/`CaptureConfig` follow the existing `SummaryConfig` pattern (`#[serde(default)]`, manual `Default` for the durations). `update_meta` takes `META_LOCK` (recover from poisoning with `into_inner`), so the guarantee holds within one process; the daemon is the only writer while it runs (spec section 9).

- [ ] **Step 4: Run the tests**

Run: `cargo test --no-default-features`
Expected: all pass (existing `meta.toml` fixtures still parse; `app_label` changes no existing assertion, check `tests/cli.rs` and `render` tests).

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets --no-default-features -- -D warnings
git add src/config.rs src/store
git commit -m "feat: detect and capture config, open conversation in meta, update_meta"
```

---

### Task 2: The pipeline works on a session that is still recording

**Files:**
- Modify: `src/pipeline.rs`, `src/cli.rs`, `docs/TODO.md`

**Interfaces:**
- Consumes: `Session::update_meta` (Task 1)
- Produces: `Pipeline::process(&mut self, session: &Session) -> Result<Report>` accepts `state = "recording"` and then works only on the conversations already in `meta.conversations` (they are all closed; the open one lives in `open_conversation_ms`). It never compresses while `ended_at` is `None` (already true). `retranscribe` still refuses a recording session. The CLI's `kute process <id>` refuses `state = "recording"` itself, with the existing message.

- [ ] **Step 1: Write the failing tests** (in `src/pipeline.rs` tests)

```rust
#[test]
fn process_handles_closed_conversations_while_recording() {
    let f = fixture(2.0, 2.0, 2_000);
    f.session.update_meta(|m| { m.state = SessionState::Recording; m.ended_at = None; m.open_conversation_ms = Some(1_500); }).unwrap();
    let mut asr = FakeProvider::new();
    let sum = summarizer(false);
    let report = pipeline(&f, &mut asr, Some(&sum)).process(&f.session).unwrap();
    assert_eq!(report.transcribed, vec![1]);
    assert!(!report.compressed);
    let meta = f.session.load_meta().unwrap();
    assert_eq!(meta.state, SessionState::Recording);
    assert_eq!(meta.open_conversation_ms, Some(1_500));
    assert!(f.session.wav_path(Track::Me).is_file());
}

#[test]
fn process_keeps_conversations_added_while_it_runs() {
    // A summarizer that, while summarizing conversation 1, appends conversation 2 the way the
    // live session does. After process(), both exist; 1 is summarized, 2 is still pending.
    let f = fixture(2.0, 2.0, 2_000);
    let sum = AppendingSummarizer { session: &f.session }; // summarize(): update_meta push Conversation::new(2, 1_000, 2_000), return "It went fine."
    let mut asr = FakeProvider::new();
    pipeline(&f, &mut asr, Some(&sum)).process(&f.session).unwrap();
    let meta = f.session.load_meta().unwrap();
    assert_eq!(meta.conversations.len(), 2);
    assert_eq!(meta.conversation(1).unwrap().stage, Stage::Summarized);
    assert_eq!(meta.conversation(2).unwrap().stage, Stage::Pending);
}
```

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --lib --no-default-features pipeline::`
Expected: the first fails with "is still recording", the second with 1 conversation instead of 2.

- [ ] **Step 3: Implement**

Remove the `Recording` bail from `process` (keep it in `retranscribe`, and add it to `Command::Process` in `src/cli.rs`). Replace every `session.save_meta(&meta)` in `pipeline.rs` with an `update_meta` that copies over only what this run owns: the touched conversation (by `n`), `compression`, and `state` when compression finished. Re-read `ended_at` from the fresh meta inside `compress_if_ready` (`update_meta(|_| {})` returns it). Keep the local `meta` for reading. `write_documents` and `index_session` run on the meta returned by the last `update_meta`.

- [ ] **Step 4: Run all tests**

Run: `cargo test --no-default-features`
Expected: all pass, including the existing pipeline tests.

- [ ] **Step 5: Close the related TODO item and commit**

In `docs/TODO.md`, tick "A model-load failure counts as an attempt on every conversation" with: "By design: `attempts` counts processing runs that tried the conversation, and the worker's backoff (plan 3) reads it that way, so a failed model load is one failed run for each waiting conversation."

```bash
git add src/pipeline.rs src/cli.rs docs/TODO.md
git commit -m "feat: process closed conversations while a session is still recording"
```

---

### Task 3: A summarizer timeout kills the whole process group

**Files:**
- Modify: `src/summarize/command.rs`, `Cargo.toml` (add `libc = "0.2"`), `docs/TODO.md`

**Interfaces:**
- Produces: no API change. `CommandSummarizer::summarize` spawns with `std::os::unix::process::CommandExt::process_group(0)` and on timeout calls `libc::kill(-(child.id() as i32), libc::SIGKILL)` before `wait`.

- [ ] **Step 1: Write the failing test** (in `src/summarize/command.rs` tests)

```rust
#[test]
fn timeout_kills_grandchildren() {
    // sh starts a background sleep that writes its PID, then waits. After the timeout the
    // grandchild must be gone too.
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let script = format!("sleep 30 & echo $! > {}; wait", pidfile.display());
    let s = CommandSummarizer::new(vec!["sh".into(), "-c".into(), script], std::time::Duration::from_millis(500));
    assert!(s.summarize("prompt").is_err());
    let pid: i32 = std::fs::read_to_string(&pidfile).unwrap().trim().parse().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists(), "grandchild {pid} survived");
}
```

- [ ] **Step 2: Run it and see it fail**

Run: `cargo test --lib --no-default-features summarize::command::tests::timeout_kills_grandchildren`
Expected: FAIL, "grandchild ... survived".

- [ ] **Step 3: Implement** as in Interfaces; the `unsafe` block gets a `// SAFETY:` comment (plain syscall, negative PID = the group we created).

- [ ] **Step 4: Run the summarizer tests**

Run: `cargo test --lib --no-default-features summarize::`
Expected: all pass.

- [ ] **Step 5: Tick the TODO item and commit**

```bash
git add Cargo.toml Cargo.lock src/summarize/command.rs docs/TODO.md
git commit -m "fix: kill the summarizer's whole process group on timeout"
```

---

### Task 4: PipeWire graph from `pw-dump --monitor`

**Files:**
- Create: `src/pw/mod.rs`, `src/pw/graph.rs`, `src/pw/monitor.rs`
- Modify: `src/lib.rs` (`pub mod pw;`)

**Interfaces:**
- Produces:
  ```rust
  // src/pw/graph.rs
  #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
  pub struct App { pub name: Option<String>, pub binary: Option<String> }
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum StreamKind { Mic, Output }
  #[derive(Debug, Clone, PartialEq)]
  pub struct Stream {
      pub serial: u64,
      pub kind: StreamKind,
      pub app: App,
      pub node_name: Option<String>,
      pub pid: Option<u32>,
      pub monitor: bool,   // stream.monitor = true (peak meters)
      pub running: bool,   // info.state == "running"
  }
  #[derive(Debug, Clone, Default, PartialEq)]
  pub struct Snapshot { pub streams: Vec<Stream>, pub default_source: Option<String>, pub default_sink: Option<String> }
  #[derive(Default)]
  pub struct Graph { /* nodes, links, defaults keyed by object id */ }
  impl Graph {
      pub fn apply(&mut self, objects: &[serde_json::Value]);
      pub fn snapshot(&self) -> Snapshot;              // streams sorted by serial
      pub fn find_source(&self, substring: &str) -> Option<String>; // node.name of the first Audio/Source whose node.name or node.description contains it, case-insensitively
  }
  // src/pw/monitor.rs
  pub fn read_updates(r: impl std::io::Read, on_change: impl FnMut(Snapshot)) -> anyhow::Result<()>;
  pub fn spawn(tx: std::sync::mpsc::Sender<MonitorEvent>) -> std::thread::JoinHandle<()>;
  pub enum MonitorEvent { Snapshot(Snapshot), Disconnected(String) }
  ```

Rules for `apply` (from `pw-dump`'s format, verified on PipeWire 1.6):
- Input is one JSON array per update; `read_updates` parses the concatenated arrays with `serde_json::Deserializer::from_reader(r).into_iter::<Vec<Value>>()` and calls `on_change` after each array only when the snapshot differs from the last one.
- An object with `"info": null` (or `"type": null`) removes whatever has that `id` (node, link or metadata).
- `PipeWire:Interface:Node`: keep `info.props` (`object.serial`, `media.class`, `node.name`, `application.name`, `application.process.binary`, `application.process.id`, `stream.monitor`) and `info.state`. An update without `info.props` keeps the previous props and only changes the state.
- `PipeWire:Interface:Link`: `info["output-node-id"]` and `info["input-node-id"]`.
- `PipeWire:Interface:Metadata` with `props["metadata.name"] == "default"`: entries in `metadata[]` with keys `default.audio.source` / `default.audio.sink`; the value is `{"name": "..."}` (accept a JSON-encoded string too).
- A `Stream/Input/Audio` node is a **Mic** stream only if some link goes from a node with `media.class == "Audio/Source"` into it. `Stream/Output/Audio` nodes are **Output** streams. Other nodes (video, devices, input streams fed by monitors or other streams) are not in the snapshot.
- `stream.monitor` may be the string `"true"` or a JSON bool.

`spawn` runs `pw-dump --monitor --no-colors` in a loop on its own thread: `read_updates` into `MonitorEvent::Snapshot`, and when the process exits or fails to start, send `Disconnected(reason)`, wait (1, 2, 5, 10, then 30 s repeatedly; reset after a run that lasted over 60 s), start again with a fresh `Graph`. It stops when `tx.send` fails.

- [ ] **Step 1: Write the failing tests** (in `src/pw/graph.rs` and `src/pw/monitor.rs`)

Build objects with small helpers in the test module, e.g. `node(id, serial, class, &[("node.name","x"), ...], state)`, `link(id, out, inn)`, `removed(id)`, `defaults(id, source, sink)`. Tests:
- `mic_stream_needs_a_link_from_a_real_source`: input stream linked from an `Audio/Source` is `Mic`; one linked only from a `Stream/Output/Audio` node (Plasma's meter on Jitsi's output, as in the spike) is absent.
- `removal_drops_node_and_its_links`: after `removed(node)`, the stream is gone; after `removed(link)`, the mic stream is gone.
- `state_only_update_keeps_props`: second object for the same id with `info: {"state": "idle"}` and no props, the stream keeps its app and `running == false`.
- `peak_meter_flag_and_app_fields`: `stream.monitor = "true"`, `application.name = "Jitsi Meet"`, `application.process.binary = "electron"`, `application.process.id = 4242` all land in `Stream`.
- `defaults_come_from_metadata`: source and sink names; a string-encoded value `"{\"name\":\"x\"}"` works too.
- `find_source_matches_a_substring`: `find_source("usb")` returns the USB source's `node.name`.
- In `monitor.rs`: `read_updates_reports_only_changes`: three concatenated arrays (add mic stream, a no-op update of an unrelated device, remove it) produce exactly 2 snapshots.

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --lib --no-default-features pw::`
Expected: compile errors (module missing).

- [ ] **Step 3: Implement `graph.rs` and `monitor.rs`** per the rules above.

- [ ] **Step 4: Run the tests, then try it live**

Run: `cargo test --lib --no-default-features pw::`
Expected: PASS.
Live check, not committed: save `timeout 3 pw-dump --monitor --no-colors > /tmp/dump.json` while a call app is open, feed the file to `read_updates` in a scratch test, and compare the last snapshot with `wpctl status`. Delete the scratch test and the file afterwards, since they carry this machine's device names.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/pw
git commit -m "feat: PipeWire graph snapshots from pw-dump --monitor"
```

---

### Task 5: The detector state machine

**Files:**
- Create: `src/detect.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `pw::graph::{Stream, StreamKind, App}` (Task 4), `config::DetectConfig` (Task 1)
- Produces:
  ```rust
  pub const MIC_GAP_MS: u64 = 2_000;
  #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
  #[serde(rename_all = "lowercase")]
  pub enum Phase { Idle, Tentative, Recording, Grace }
  #[derive(Debug, Clone, PartialEq)]
  pub enum Action {
      Start { trigger: Option<App> },   // None = manual
      Commit,                            // Tentative -> Recording (plan 4 notifies here)
      Finish,                            // finalize and enqueue
      Discard,                           // delete the capture
  }
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum DetectError { AlreadyRecording, NotRecording }
  pub struct Detector { /* config, phase, manual, paused, trigger, bound outputs, dead_since, timers */ }
  impl Detector {
      pub fn new(config: &DetectConfig, own_pid: u32) -> Detector;
      pub fn streams(&mut self, streams: &[Stream], now_ms: u64) -> Vec<Action>;
      pub fn tick(&mut self, now_ms: u64) -> Vec<Action>;
      pub fn record(&mut self, now_ms: u64) -> Result<Vec<Action>, DetectError>;
      pub fn stop(&mut self) -> Result<Vec<Action>, DetectError>;    // -> [Finish] (Tentative: also [Finish], it becomes a real session)
      pub fn discard(&mut self) -> Result<Vec<Action>, DetectError>; // -> [Discard]
      pub fn pause(&mut self) -> Vec<Action>;
      pub fn resume(&mut self);
      pub fn phase(&self) -> Phase;
      pub fn paused(&self) -> bool;
      pub fn manual(&self) -> bool;
      pub fn trigger(&self) -> Option<&App>;
  }
  ```
  `DetectError` implements `Display` ("already recording", "nothing is being recorded") by hand; no new crate.

Rules (spec section 5 as amended; times are the caller's monotonic milliseconds):
- **Ignored:** `node_name` starts with `kute-`; `pid == own_pid`; `monitor`; app in `ignore_apps`; `allow_apps` set and app not in it. An app matches a rule when `name` or `binary` equals it case-insensitively.
- **Alive:** a non-ignored `Mic` stream exists, or a bound output serial is present with `running == true`. While a non-ignored mic stream exists, every `Output` stream whose `app` equals the trigger app (both fields) is added to the bound set.
- **Not alive** only counts once it has lasted `MIC_GAP_MS` (`dead_since + MIC_GAP_MS <= now`), checked in `streams` and `tick`.
- Idle + alive (and not paused) -> `[Start { trigger: Some(app of the lowest-serial mic stream) }]`, phase Tentative.
- Tentative, alive, `now - started >= min_duration` -> `[Commit]`, Recording. Tentative, confirmed not alive -> `[Discard]`, Idle.
- Recording (automatic), confirmed not alive -> Grace (grace starts at `dead_since`). Grace + alive -> Recording. Grace with `now - grace_start >= stop_grace` -> `[Finish]`, Idle.
- `record`, `stop` and `discard` work while paused (spec: a manual `kute record` still works).
- Manual sessions (`record` from Idle) -> `[Start { trigger: None }, Commit]`; ignore liveness; only `stop`/`discard` end them. `record` in Tentative -> `[Commit]`, becomes manual. `record` in Recording/Grace -> `Err(AlreadyRecording)`. `stop`/`discard` in Idle -> `Err(NotRecording)`.
- `pause`: sets paused; Tentative -> `[Discard]`; automatic Recording/Grace -> `[Finish]`; manual -> no action. While paused, `streams` only stores nothing and returns no actions. `resume` clears paused (a call already in progress then starts a new Tentative on the next snapshot).
- After `Finish`/`Discard`, the bound set, trigger and timers reset.

- [ ] **Step 1: Write the failing tests**

Helpers: `mic(serial, app)`, `out(serial, app, running)`, `meter(serial)` (`monitor: true`, `node_name: Some("PlasmaPA-VolumeMeter")`, app name `None`, binary `plasmashell`), `kute(serial)` (`node_name: Some("kute-me")`), `jitsi()` = `App { name: Some("Jitsi Meet"), binary: Some("electron") }`, `cfg()` = defaults. The sequences mirror the spike in spec section 12.

```rust
#[test]
fn website_mic_check_is_discarded() {
    let mut d = Detector::new(&cfg(), 1);
    assert!(matches!(d.streams(&[mic(10, firefox())], 0)[..], [Action::Start { .. }]));
    assert!(d.streams(&[], 5_000).is_empty());          // gap not yet confirmed
    assert_eq!(d.tick(7_000), vec![Action::Discard]);   // 5 s + 2 s
    assert_eq!(d.phase(), Phase::Idle);
}

#[test]
fn jitsi_lobby_to_call_is_one_session() {
    // Pre-join mic closes, the call's mic opens 0.5 s later.
    let mut d = Detector::new(&cfg(), 1);
    d.streams(&[mic(85, jitsi())], 0);
    d.streams(&[], 11_000);
    assert!(d.streams(&[mic(88, jitsi()), out(114, jitsi(), true)], 11_500).is_empty());
    assert_eq!(d.tick(20_000), vec![Action::Commit]);
}
```
Also:
- `call_commits_then_finishes_after_grace`: mic+output alive from 0; `tick(20_000) == [Commit]`; at 100 s a snapshot without the mic and with the output `running: false` (leaving the room). Grace starts at `dead_since` (100 s) once the gap is confirmed at 102 s: `tick(114_999)` is empty, `tick(115_000) == [Finish]`.
- `bound_output_keeps_session_alive_only_while_running`: mic closes but output 114 stays `running` (an app closing its mic on mute): no Grace after 60 s; output goes `running: false`: Grace, then `Finish`.
- `new_output_node_after_the_call_does_not_count`: after the mic closes, a new output 200 of the same app appears running: still Grace -> `Finish`.
- `peak_meters_never_start_a_session`: `[meter(1), meter(2)]` for 60 s of ticks: no actions.
- `own_capture_streams_are_ignored`: `[kute(5)]` and a mic stream with `pid == own_pid`: no actions; and during a session they do not keep it alive.
- `ignore_and_allow_apps_match_name_or_binary`: `ignore_apps = ["ELECTRON"]` ignores Jitsi; `allow_apps = ["jitsi meet"]` ignores Firefox, accepts Jitsi.
- `manual_record_ignores_liveness`: `record(0) == [Start{trigger: None}, Commit]`; 10 min of empty snapshots and ticks: no actions; `stop() == [Finish]`; `stop()` again is `Err(NotRecording)`.
- `record_during_tentative_commits_as_manual` and `record_while_recording_fails`.
- `pause_ends_automatic_sessions_and_ignores_streams`: Recording + `pause() == [Finish]`; mic streams while paused: no actions; `resume()` then a snapshot with a mic: `[Start{..}]`.

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --lib --no-default-features detect::`
Expected: compile errors.

- [ ] **Step 3: Implement `src/detect.rs`** per the rules. Store the last snapshot so `tick` can evaluate liveness without new streams.

- [ ] **Step 4: Run the tests**

Run: `cargo test --lib --no-default-features detect::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/detect.rs
git commit -m "feat: call detection state machine"
```

---

### Task 6: Live conversation splitting

**Files:**
- Create: `src/split.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces:
  ```rust
  pub const PAD_MS: u64 = 2_000;
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum SplitEvent { Opened { start_ms: u64 }, Closed { start_ms: u64, end_ms: u64 } }
  pub struct Splitter { /* gap_ms, open: Option<(first_speech_ms, last_speech_end_ms)> */ }
  impl Splitter {
      pub fn new(gap_ms: u64) -> Splitter;
      /// Speech on either track, in session time.
      pub fn speech(&mut self, start_ms: u64, end_ms: u64) -> Option<SplitEvent>;
      /// Both tracks have been analysed up to `analysed_ms` (session time).
      pub fn advance(&mut self, analysed_ms: u64) -> Option<SplitEvent>;
      /// The session ended at `end_ms`.
      pub fn finish(&mut self, end_ms: u64) -> Option<SplitEvent>;
      pub fn open_start_ms(&self) -> Option<u64>; // padded start of the open conversation
  }
  ```
  Opened start = `first_speech.saturating_sub(PAD_MS)`. `advance` closes when `analysed_ms >= last_speech_end + gap_ms`, with end = `last_speech_end + PAD_MS`. `finish` closes with end = `min(last_speech_end + PAD_MS, end_ms)`. Speech that arrives out of order (the slower track) only extends `last_speech_end` if later.

- [ ] **Step 1: Write the failing tests**

- `two_chats_far_apart_are_two_conversations`: gap 300_000; speech 600_000..610_000 -> `Opened{598_000}`; speech 620_000..630_000 -> None; `advance(929_999)` None; `advance(930_000)` -> `Closed{598_000, 632_000}`; speech 2_000_000..2_001_000 -> `Opened{1_998_000}`.
- `pauses_shorter_than_the_gap_stay_in_one`: speech at 0..1_000 and 200_000..201_000 with `advance` calls between: one `Opened`, no `Closed` until `advance(501_000)`.
- `padding_clamps_at_zero_and_session_end`: speech 500..1_000 -> `Opened{0}`; `finish(1_500)` -> `Closed{0, 1_500}`.
- `no_speech_no_conversation`: `advance(10_000_000)` and `finish(10_000_000)` are `None`.
- `late_speech_from_the_slower_track_extends_the_open_conversation`: speech 0..1_000, then 50_000..60_000 reported after `advance(40_000)`: no close at `advance(301_000)`, close at `advance(360_000)` with end `62_000`.

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --lib --no-default-features split::`
Expected: compile errors.

- [ ] **Step 3: Implement `src/split.rs`.**

- [ ] **Step 4: Run the tests** (same command, PASS).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/split.rs
git commit -m "feat: live conversation splitting"
```

---

### Task 7: Track recording

**Files:**
- Create: `src/recorder.rs`, `src/capture.rs`
- Create: `src/testing.rs` (`#[doc(hidden)] pub mod testing;` in `src/lib.rs`): test fakes shared by unit tests and `tests/daemon.rs`, which can't see `#[cfg(test)]` items
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `transcribe::{SpeechDetector, Span}`, `store::Track`, `audio::{samples_to_ms, ms_to_samples}`
- Produces:
  ```rust
  // src/capture.rs
  pub struct Capture { pub reader: Box<dyn std::io::Read + Send>, stop: Box<dyn FnOnce() + Send> }
  impl Capture { pub fn new(reader: ..., stop: impl FnOnce() + Send + 'static) -> Capture; pub fn stop(self) }
  pub trait CaptureSource: Send {
      /// `target`: a node.name to record from (Me with capture.mic); None = the default.
      fn open(&self, track: Track, target: Option<&str>) -> anyhow::Result<Capture>;
  }
  pub struct PwRecord; // impl CaptureSource: spawns pw-record, reader = child stdout, stop = kill + wait
  pub fn pw_record_args(track: Track, target: Option<&str>) -> Vec<String>;

  // src/recorder.rs
  pub const FLUSH_EVERY: u32 = 80_000;   // samples (5 s)
  pub const VAD_BLOCK_MS: u64 = 10_000;
  pub struct TrackWriter { /* hound::WavWriter<BufWriter<File>>, samples: u64 */ }
  impl TrackWriter {
      pub fn create(path: &Path) -> Result<TrackWriter>;   // 16 kHz, mono, 16-bit
      pub fn write(&mut self, samples: &[i16]) -> Result<()>; // flush() header every FLUSH_EVERY samples
      pub fn pad_to(&mut self, total_samples: u64) -> Result<()>; // append zeros up to this length
      pub fn samples(&self) -> u64;
      pub fn finish(self) -> Result<u64>;
  }
  pub fn repair_wav_header(path: &Path) -> Result<u64>; // returns samples
  #[derive(Debug)]
  pub enum RecorderEvent {
      FirstSample { track: Track, at: jiff::Zoned },
      Speech { track: Track, start_ms: u64, end_ms: u64 }, // track time
      Analysed { track: Track, until_ms: u64 },            // track time
      Ended { track: Track, samples: u64 },
      Failed { track: Track, error: String },
  }
  pub type VadFactory = std::sync::Arc<dyn Fn() -> Option<Box<dyn SpeechDetector>> + Send + Sync>;
  pub struct TrackOptions { pub gap_fill: bool }
  pub fn spawn_track(track: Track, capture: Capture, path: PathBuf, vad: VadFactory, options: TrackOptions, tx: Sender<RecorderEvent>) -> std::thread::JoinHandle<()>;
  ```
- `pw_record_args(Track::Me, None)` = `["--rate","16000","--channels","1","--format","s16","-P","{ node.name = \"kute-me\" media.name = \"kute (me)\" }","-"]`; with a target, `"--target", target` before `"-"`. Others: `-P` value `{ node.name = "kute-others" media.name = "kute (others)" stream.capture.sink = true }`.
- The track thread: reads 100 ms (3 200 bytes) at a time, handles a trailing odd byte across reads, sends `FirstSample` with `time::now()` on the first bytes, writes through `TrackWriter`. Gap fill: before each write, `expected = elapsed_since_first_sample * 16`; if `expected > written + 8_000` (500 ms), `pad_to(expected)` first. Every `VAD_BLOCK_MS` of written audio it runs the VAD (created once by the factory inside the thread, since whisper.cpp contexts are not `Send`) on that block as f32 and sends `Speech` per span (offset to track time) and `Analysed`. No VAD (factory returned `None`): no `Speech` and no `Analysed` events. On EOF: run VAD on the partial block, `finish`, send `Ended`. On any write error: send `Failed` and stop reading (the daemon handles disk full).

- [ ] **Step 1: Write the fake and the failing tests**

In `src/testing.rs`: `pub struct RunVad;` implementing `SpeechDetector`, one span per run of non-zero samples at 10 ms resolution, and `pub fn run_vad() -> VadFactory`.

Tests in `src/recorder.rs`:
- `writer_keeps_a_valid_header_while_writing`: write 100 000 samples without `finish`, then `hound::WavReader::open` on the path reports at least 80 000 samples (the flush), and after `finish` exactly 100 000.
- `repair_wav_header_fixes_a_crashed_file`: write 50 000 samples, `std::mem::forget` the writer (no finalize), call `repair_wav_header` -> `50_000`, `WavReader` reads 50 000.
- `gap_fill_keeps_track_time_aligned`: unit-test the rule through a helper `fn gap_fill_target(elapsed_ms: u64, written: u64) -> Option<u64>`: `(1_000, 16_000)` -> None; `(2_000, 16_000)` -> `Some(32_000)`; `(1_400, 16_000)` -> None (under 500 ms).
- `track_thread_writes_wav_and_reports_speech`: `std::io::pipe()`; write 3 s of zeros, 2 s of 1000s, 20 s of zeros as S16LE, drop the writer; `spawn_track` with `RunVad`; collect events: one `FirstSample`, `Speech { start_ms: 3_000, end_ms: 5_000 }` (allow ±10 ms), `Analysed` values 10_000 then 20_000 then 25_000, `Ended { samples: 400_000 }`; the WAV has 400 000 samples. (Gap fill would add samples here, because the pipe delivers 25 s instantly: give the thread a `gap_fill: bool` parameter via a `TrackOptions { gap_fill: bool }` struct, `true` in the daemon and `false` in this test.)
- `pw_record_args_for_both_tracks`: exact vectors as above.

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --lib --no-default-features recorder:: capture::`
Expected: compile errors.

- [ ] **Step 3: Implement `src/capture.rs` and `src/recorder.rs`.**

`repair_wav_header`: hound's header is the canonical 44 bytes for this format; set the RIFF size (offset 4) to `len - 8` and the data size (offset 40) to `len - 44`, rounded down to whole samples; return `(len - 44) / 2`.

- [ ] **Step 4: Run the tests, then one live check**

Run: `cargo test --lib --no-default-features recorder:: capture::`
Expected: PASS.
Live: a scratch `#[ignore]` test that opens `PwRecord` for both tracks for 3 s and checks each WAV has 44 000 to 50 000 samples. Run it once with `-- --ignored`, keep it `#[ignore]` (CI has no PipeWire).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/recorder.rs src/capture.rs src/testing.rs
git commit -m "feat: record tracks from pw-record with live VAD"
```

---

### Task 8: Live sessions and crash recovery

**Files:**
- Create: `src/live.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: Tasks 1, 6, 7; `Store::create_session`, `Meta::new`, `time::{now, ms_between}`
- Produces:
  ```rust
  pub struct LiveSession { /* session, captures' stop handles, threads, its own Receiver<RecorderEvent>, splitter, t0: Option<Zoned>, offsets, analysed per track, vad_ok, ended tracks */ }
  pub struct StartRequest<'a> { pub trigger: Option<App>, pub title: Option<String>, pub mic_target: Option<String>, pub devices: (Option<String>, Option<String>) }
  impl LiveSession {
      pub fn start(store: &Store, config: &Config, req: StartRequest, source: &dyn CaptureSource, vad: VadFactory) -> Result<LiveSession>;
      pub fn id(&self) -> &str;
      /// Drains the session's own event channel (`try_recv`) through `on_event`; returns the
      /// numbers of conversations that just closed (to enqueue). The daemon calls it every tick.
      pub fn pump(&mut self) -> Result<Vec<u32>>;
      fn on_event(&mut self, ev: RecorderEvent) -> Result<Vec<u32>>;
      pub fn finish(self) -> Result<Session>;
      pub fn discard(self) -> Result<()>;
      pub fn failed(&self) -> Option<&str>; // set when a track sent Failed
  }
  pub fn recover(store: &Store) -> Result<Vec<Session>>;
  ```

Behavior:
- `start`: `Store::create_session(&time::now())`; `Meta::new` with `TriggerKind::App` (`app_name`, `binary` from the `App`) or `Manual`, title, `summary.language` from config (as `import` does), `tracks.me.device` / `tracks.others.device` from `devices` (default source / default sink + `.monitor`, or the mic target); save; open both captures; spawn both track threads with `TrackOptions { gap_fill: true }`. `vad_ok` = the factory returned `Some` once at start (call it once to probe and drop the result).
- `on_event`: first `FirstSample` sets `t0` and `started_at` (`update_meta`); a later one sets that track's `offset_ms = ms_between(t0, at)`. `Speech`/`Analysed` are shifted to session time by the track's offset; `Analysed` updates that track's value, then `splitter.advance(min of both tracks)` (a track that hasn't analysed anything yet holds it back). `Opened` writes `open_conversation_ms`; `Closed` appends `Conversation::new(next_n, start, end)` with `stage = Pending` and clears `open_conversation_ms` in one `update_meta`, and returns `n`. `Failed` records the error.
- `finish`: stop both captures, join both track threads (they exit after EOF, having sent everything), then drain the channel completely through `on_event`. Because the channel belongs to the session, no event can be lost to another receiver. Session end = the longer track's `offset + samples_to_ms(samples)`. Without VAD: one conversation `0..end` if `end > 0`. With VAD: `splitter.finish(end)`. Then one `update_meta`: append the last conversation, `open_conversation_ms = None`, `ended_at = started_at + end`, `state = Processing`. Return the `Session`.
- `discard`: stop captures, join threads, `Session::delete`.
- `recover`: for each session with `state = Recording`: `repair_wav_header` on each existing WAV, `interrupted = true`, `end = max over tracks of offset + duration`, close `open_conversation_ms` at `end` as a new conversation, `ended_at`, `state = Processing`; return them.

- [ ] **Step 1: Write the failing tests** (in `src/live.rs`, using a `testing::FakeSource: CaptureSource`, added to `src/testing.rs` here: it hands out `std::io::pipe()` readers, keeps the writers for the test (`writer(track)`), and its `stop` drops them; plus `RunVad`)

- `a_session_with_two_chats_becomes_two_conversations`: config `conversation_gap = 30s`; write 100 s to both pipes up front: Me speaks at 1–3 s, Others at 60–62 s, silence otherwise; call `pump` in a loop (sleeping 10 ms) until it has returned `[1]` and both tracks are analysed past 100 s; close the writers; `finish` -> meta has 2 conversations (`1: 0..5_000`, `2: 58_000..64_000`, both `Pending`), `state = Processing`, `ended_at` set, `open_conversation_ms = None`.
- `offsets_come_from_first_samples`: feed `on_event` synthetic `FirstSample` events 37 ms apart -> `tracks.others.offset_ms == 37`, `started_at` = Me's instant.
- `without_live_vad_the_session_is_one_conversation`: factory returns `None`; 10 s of audio with speech -> after `finish`, exactly one conversation `0..10_000`.
- `discard_removes_the_folder`.
- `recover_repairs_an_interrupted_session`: build a session folder by hand: meta `Recording` with `open_conversation_ms = Some(1_000)`, two WAVs written with `TrackWriter` and then `forget`; `recover` -> returns it; meta `interrupted`, `Processing`, conversation `1_000..<WAV duration>`, `ended_at` set; both WAVs open with `hound`.

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --lib --no-default-features live::`
Expected: compile errors.

- [ ] **Step 3: Implement `src/live.rs`.**

- [ ] **Step 4: Run the tests** (same command, PASS), then the whole suite.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/live.rs src/testing.rs
git commit -m "feat: live sessions with conversation splitting and crash recovery"
```

---

### Task 9: Jobs and the worker

**Files:**
- Create: `src/jobs.rs`, `src/worker.rs`
- Modify: `src/lib.rs`, `src/cli.rs` (the processing commands call `jobs::run`), `src/index.rs` (busy timeout)

**Interfaces:**
- Consumes: `pipeline::*`, `import::*`, `summarize::*`
- Produces:
  ```rust
  // src/jobs.rs
  #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
  #[serde(tag = "job", rename_all = "snake_case")]
  pub enum Job {
      Process { id: String },
      Retranscribe { id: String, model: Option<String> },
      Summarize { id: String, conversation: Option<u32>, lang: Option<String>, backend: Option<SummaryBackend> },
      Import { me: PathBuf, others: PathBuf, start: Option<String>, title: Option<String>, process: bool },
      Title { id: String, title: String },
      Reindex,
  }
  pub struct Outcome { pub message: String, pub session: Option<String> }
  pub fn run(store: &Store, config: &Config, asr: &mut Option<WhisperProvider>, job: &Job) -> Result<Outcome>;
  pub fn retry_delay(meta: &Meta) -> Option<std::time::Duration>;

  // src/worker.rs
  pub enum WorkerMsg { Run(Job), Shutdown }
  pub struct WorkerDone { pub job: Job, pub result: Result<Outcome, String>, pub retry_after: Option<Duration> }
  pub fn spawn(store: Store, config: Config, rx: Receiver<WorkerMsg>, done: Sender<WorkerDone>) -> JoinHandle<()>;
  pub fn startup_jobs(store: &Store) -> Result<Vec<Job>>;
  ```
- `jobs::run` is the body the CLI already has for import / process / summarize / retranscribe / title / reindex, moved out of `cli.rs` so the CLI (no daemon) and the worker share it; `Summarize` with `lang` saves it with `update_meta` first, `backend` overrides `config.summary.backend` for this job only. `Process` and `Import { process: true }` build the `WhisperProvider` lazily into `asr` (kept between jobs; dropped by the worker when the next queued job doesn't transcribe or the queue is empty, spec section 6).
- `retry_delay`: the minimum over conversations with `stage = Failed`, `failed_step = Transcribe` and `attempts` 1, 2 or 3 of 1, 5, 30 minutes respectively; `None` otherwise.
- `worker::spawn`: loops on `rx`; for each `Run(job)` calls `jobs::run`, then for `Process` (and `Import` with processing) loads the meta and sets `retry_after = retry_delay(..)`; sends `WorkerDone`. A panic inside a job must not kill the worker (`std::panic::catch_unwind` with `AssertUnwindSafe`, reported as an error).
- `startup_jobs`: `Process { id }` for every session with `state = Processing`, in session order.
- `Index::open` sets `conn.busy_timeout(Duration::from_secs(5))`.

- [ ] **Step 1: Write the failing tests**

- `retry_delay_follows_the_backoff` (in `jobs.rs`): attempts 1 -> 60 s, 2 -> 300 s, 3 -> 1 800 s, 4 -> None, failed at summarize -> None, two failed conversations with attempts 1 and 3 -> 60 s.
- `job_json_round_trips` : `Job::Summarize { id: "x".into(), conversation: Some(2), lang: Some("it".into()), backend: None }` serializes to an object with `"job": "summarize"` and back.
- `startup_jobs_lists_processing_sessions_in_order` (in `worker.rs`): three sessions (`Done`, `Processing`, `Processing`) -> two `Process` jobs, oldest first.
- `worker_survives_a_failing_job`: `spawn` with a store; send `Run(Job::Process { id: "missing".into() })` then `Run(Job::Reindex)`; two `WorkerDone`, the first `Err`, the second `Ok`.
- The existing `tests/cli.rs` tests keep passing (they now go through `jobs::run`).

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --no-default-features jobs:: worker::`
Expected: compile errors.

- [ ] **Step 3: Implement**, moving code out of `cli.rs` rather than duplicating it.

- [ ] **Step 4: Run all tests**

Run: `cargo test --no-default-features`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/jobs.rs src/worker.rs src/cli.rs src/index.rs
git commit -m "feat: job queue worker with transcription retry backoff"
```

---

### Task 10: The control socket

**Files:**
- Create: `src/ipc.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `jobs::Job`, `detect::Phase`
- Produces:
  ```rust
  #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
  #[serde(tag = "cmd", rename_all = "snake_case")]
  pub enum Request { Status, Record { title: Option<String> }, Stop, Discard, Pause, Resume, Job { job: Job } }
  #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
  pub struct Status { pub phase: Phase, pub paused: bool, pub session: Option<String>, pub app: Option<String>, pub queued: usize, pub running_job: Option<String> }
  #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
  #[serde(tag = "reply", rename_all = "snake_case")]
  pub enum Response { Ok { message: String }, Status { status: Status }, Error { message: String } }
  pub fn socket_path() -> Result<PathBuf>;
  pub struct Incoming { pub request: Request, pub reply: Sender<Response> }
  pub fn serve(path: &Path, tx: Sender<Incoming>) -> Result<JoinHandle<()>>;
  pub fn request(path: &Path, req: &Request) -> Result<Option<Response>>; // Ok(None) = no daemon
  ```
  `serve`: if `path` exists, try to connect: success -> `bail!("kute daemon is already running")`; failure -> remove the stale file. Bind, `chmod 0600`, then a thread accepts connections; each connection is handled on its own short thread: read one line, parse (`Error` reply on bad JSON), send `Incoming`, wait up to 30 s for the reply, write it as one line. `request`: `NotFound` or `ConnectionRefused` on connect -> `Ok(None)`.

- [ ] **Step 1: Write the failing tests**

- `request_round_trip`: `serve` in a tempdir; a thread answers `Status` with a fixed `Status`; `request` returns it.
- `no_daemon_means_none`: `request` on a path that doesn't exist -> `Ok(None)`.
- `stale_socket_is_replaced_and_a_live_one_refused`: create a `UnixListener` at the path, drop it (stale file) -> `serve` succeeds; a second `serve` on the same path while the first runs -> error containing "already running".
- `socket_is_private`: mode `0o600`.
- `bad_json_gets_an_error_reply`: write `"nonsense\n"` with a raw `UnixStream` -> an `Error` line.

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --lib --no-default-features ipc::`
Expected: compile errors.

- [ ] **Step 3: Implement `src/ipc.rs`.**

- [ ] **Step 4: Run the tests** (PASS).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/ipc.rs
git commit -m "feat: daemon control socket"
```

---

### Task 11: The daemon loop and the control commands

**Files:**
- Create: `src/daemon.rs`, `tests/daemon.rs`
- Modify: `src/lib.rs`, `src/cli.rs`

**Interfaces:**
- Consumes: everything above
- Produces:
  ```rust
  pub enum Event { Monitor(MonitorEvent), Ipc(Incoming), Worker(WorkerDone), Tick }
  pub struct Daemon { /* store, config, detector, live: Option<LiveSession>, snapshot, queue: VecDeque<Job>, running: Option<Job>, retries: Vec<(Instant-like ms, Job)>, worker tx, source, vad, own pid */ }
  impl Daemon {
      pub fn new(store: Store, config: Config, source: Box<dyn CaptureSource>, vad: VadFactory, worker: Sender<WorkerMsg>) -> Daemon;
      pub fn handle(&mut self, event: Event, now_ms: u64) -> Result<()>;
      pub fn status(&self) -> ipc::Status;
  }
  pub fn run(store: Store, config: Config) -> Result<()>; // the real thing, called by `kute daemon`
  ```

`handle` dispatches:
- `Monitor(Snapshot)` -> store it, `detector.streams(..)`; `Monitor(Disconnected)` -> log, and if a session is live, act as `Finish` (spec: "close the current session as if it ended").
- `Tick` -> `live.pump()` and enqueue `Process { id }` when it returns conversations (no duplicate if one is already queued for that session); `detector.tick(now)`; start due retries; if `live.failed()` is set (disk full) -> `Finish` and log.
- `Ipc` -> `Status` reply; `Record { title }` -> `detector.record(now)` (`AlreadyRecording` -> `Error { "already recording <id>" }`), title applied with `update_meta` when the session exists; `Stop`/`Discard`/`Pause`/`Resume` -> detector; `Job { job }` -> enqueue, reply `Ok { "queued" }`.
- `Worker(done)` -> log errors, schedule `retry_after`, start the next queued job.
- Actions: `Start` -> `LiveSession::start` with `mic_target = config.capture.mic.and_then(|m| graph.find_source(m))` and devices from the snapshot defaults (log and reset the detector to Idle via `discard()` if it fails, e.g. `pw-record` missing); `Commit` -> log "recording <id> (<app>)"; `Finish` -> `live.finish(..)`, enqueue `Process`; `Discard` -> `live.discard()`.

`run`: `Store::open`, `live::recover` (enqueue their `Process` jobs), `worker::startup_jobs`, spawn worker, `ipc::serve(socket_path)` (its `Incoming`s forwarded into the event channel), `pw::monitor::spawn`, a tick thread (500 ms), then loop `handle` with `now_ms` from one `std::time::Instant` taken at start. SIGTERM/SIGINT: not handled specially in plan 3; recovery covers it (plan 4's service install sets `KillSignal`/`TimeoutStopSec`).

CLI (`src/cli.rs`):
- New commands: `daemon`, `status`, `record [title...]`, `stop`, `discard`, `pause`, `resume`. Control commands with no daemon fail with "the kute daemon is not running (start it with `kute daemon`)".
- `status` prints the phase, paused flag, current session and app, queue length and running job, one per line.
- `import`, `process`, `summarize`, `retranscribe`, `title`, `reindex`: if `ipc::request(Job)` returns `Some`, print the reply ("queued ...") and return; otherwise run `jobs::run` in the foreground as today.

- [ ] **Step 1: Write the failing end-to-end test** (`tests/daemon.rs`, no PipeWire, no models)

```rust
#[test]
fn a_detected_call_is_recorded_split_and_queued() {
    // testing::FakeSource, testing::run_vad(), a worker channel the test reads itself.
    // t=0: snapshot with a Jitsi mic stream and a running output. Write 70 s into both pipes
    // up front (speech at 1-3 s on Me). Tick every 500 ms of fake time.
    // t=20 s: phase Recording. t=60 s: snapshot without the mic, output not running.
    // By t=78 s: Finish happened (FakeSource's stop closes the pipes).
    // Expect: one session folder, meta state Processing, trigger app_name "Jitsi Meet",
    // 1 conversation starting at 0, both WAVs readable, and WorkerMsg::Run(Process{id}) sent.
}
#[test]
fn a_mic_check_leaves_nothing_behind() { /* mic for 5 s -> no session folder, no job */ }
#[test]
fn manual_record_and_stop_over_the_daemon_api() { /* handle(Ipc(Record{title:"Standup"})) -> Ok; status phase Recording; Stop -> session with title "Standup" queued */ }
```

Also a CLI test in `tests/cli.rs`: `control_commands_without_a_daemon` (set `XDG_RUNTIME_DIR` to an empty tempdir for the child) -> `kute status` fails with "not running".

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --no-default-features --test daemon --test cli`
Expected: compile errors.

- [ ] **Step 3: Implement `src/daemon.rs` and the CLI changes.**

- [ ] **Step 4: Run everything**

Run: `cargo fmt && cargo clippy --all-targets --no-default-features -- -D warnings && cargo test --no-default-features`
Expected: clean, all PASS.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: kute daemon with status, record, stop, discard, pause, resume"
```

---

### Task 12: Acceptance on a real desktop, docs

**Files:**
- Modify: `README.md`, `docs/TODO.md`, `docs/superpowers/specs/2026-10-08-kute-design.md` (section 12 notes only if something differed)

This task needs the user at the machine (KDE Plasma, PipeWire, Jitsi Meet app, the real models).

- [ ] **Step 1: Build and start the daemon in a terminal**

Run: `cargo build --release && ./target/release/kute daemon`
Expected: it starts, logs nothing alarming, `kute status` from another terminal shows `idle`.

- [ ] **Step 2: Walk through spec section 2's scenarios** and record each result in the PR/commit message:
  1. A short Jitsi call (two devices): one session, committed after 20 s, finished 15 s after leaving, transcript and summary appear without any command.
  2. A coworking stretch: two chats more than `conversation_gap` apart (set `conversation_gap = "1m"` for the test) -> two conversations, the first transcribed while the session is still recording.
  3. A mic check under 20 s (e.g. the Jitsi pre-join screen, then cancel) -> no folder left.
  4. `kute record "Test"` / `kute stop`, `kute pause` during a call (session finishes), `kute resume`.
  5. Kill the daemon with `kill -9` mid-call, restart it: the session is recovered and processed.
  6. Open the Plasma volume applet during idle: nothing starts.

- [ ] **Step 3: Update docs**

README: a "Running the daemon" section (`kute daemon`, the control commands, "mute at the system level, not only in the call app" from spec section 5, a note that the Jitsi Meet app keeps the mic open while muted in the app). Status line: automatic recording works; tray and service install are next. `docs/TODO.md`: anything found in Step 2 that isn't fixed.

- [ ] **Step 4: Commit**

```bash
git add README.md docs
git commit -m "docs: running the daemon"
```

---

## Appendix: decisions made during execution

Tasks 1-11 ran subagent-driven (2026-10-19 to 2026-10-21), each reviewed, then a
whole-branch review and one fix wave. Where the plan above disagrees with the code, these
decisions are why. The spec has been updated to match.

**Plan text that changed**
- `pw-record` needs `--raw` (1.6 otherwise writes a 24-byte AU header to stdout).
- Gap fill (Task 7): pads only for time a single `read()` blocked beyond what it returned,
  by more than 500 ms, capped at the wall clock. The plan's "elapsed vs written" rule padded
  after the recorder itself fell behind, shifting the track for good.
- `repair_wav_header` walks the RIFF chunks instead of assuming a 44-byte header; crash
  tests build broken WAVs by zeroing the size fields of a finished file (`testing::crash_wav`).
- A mic's source may be any `Audio/Source*` class (noise-suppression filters are
  `Audio/Source/Virtual`).
- `Capture` exposes a run-once `StopHandle`, and the last handle dropped stops the capture.
- `LiveSession` owns its event channel; the daemon drains it on every tick (`pump`), so
  `finish` can join the track threads and then drain without losing events.
- `StartRequest` has no lifetime; `WorkerDone` carries the session id, so an `Import`'s
  retry processes the imported session instead of importing again.
- Snapshots carry the list of sources, so the daemon resolves `capture.mic` itself.

**Behavior**
- `kute process` refuses a recording session only in the foreground CLI; the worker
  processes closed conversations of a live session by design. `kute summarize` works on a
  recording session.
- After the user's `stop` or `discard`, automatic detection is blocked until 2 s without a
  mic in real snapshots (time without a graph doesn't count). Sessions the daemon ends
  itself never block, except a capture that dies within 5 s of starting.
- A PipeWire disconnect counts as an empty graph (the call ends normally unless it comes
  back in time), not as a stop. An idle detector also looks for a call on every tick,
  because a restart can leave the graph settled before the session ends (found in real use).
- A capture that ends on its own ends the session (`failed()`); a Tentative one with no
  audio is discarded.
- Without live VAD (model missing, or lost mid-session), everything after the last closed
  conversation becomes one conversation; crash recovery does the same for a session with no
  conversations. The splitter extends an open conversation for late speech from a lagging
  track (tracks lag by about one 10 s block, far below the gap).
- `kute discard` also removes the session from the search index and drops its queued and
  retrying jobs.
- `recover` runs before startup jobs are listed, and the daemon binds the socket first, so
  a second daemon fails before it can mark the first one's live session as interrupted.

**Added after the plan**
- Desktop notification when the daemon writes summaries, opening `summary.md` on click
  (`[notify] open`).
- `kute summarize/retranscribe/...` queued on the daemon say where the result is logged.
