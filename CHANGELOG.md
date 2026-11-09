# Changelog

All notable changes to kute are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and kute uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a new feature or a
breaking change bumps the minor version and a fix bumps the patch version.

## [0.4.0] - 2026-11-09

### Added

- **daemon:** Optional low-priority and after-recording processing
- **summary:** Record why a conversation's summary was skipped
- **cli:** Add `kute status --json`
- **transcribe:** Use the CPU by default when the only GPU is integrated

### Fixed

- **deps:** Update rusqlite to 0.41 for the bundled SQLite fix
- **detect:** Don't let screen and sound recorders hold a call open
- **tray:** Draw the icons with fills, not strokes
- **tray:** Show the icon on Cinnamon before 6.8

### Packaging

- Prepare the crate for crates.io

## [0.3.0] - 2026-11-09

### Added

- **detect:** End a manual session with the call it follows

### Fixed

- **deps:** Bump rustls, rustls-webpki and webpki-roots for advisories
- **detect:** Restart the follow count when another app's mic takes over
- **cli:** End quietly when stdout is a closed pipe
- **cli:** Run jobs here when --config is not the daemon's config
- **cli:** Ask the daemon which config it runs with
- **cli:** Queue on the daemon only when its archive matches too
- **cli:** Fall back to an old daemon only when it can't parse Identity
- **daemon:** Report the archive as an absolute path

### Packaging

- Strip and LTO the release binary

## [0.2.0] - 2026-11-08

### Added

- **tray:** Draw kute's own ear-and-mic tray icon ([#11](https://github.com/ThinkerDreamer/kute/pull/11))
- Benchmark GPU against CPU with `kute doctor --bench` ([#13](https://github.com/ThinkerDreamer/kute/pull/13))
- Selectable transcription backend, with a Qwen3-ASR helper ([#10](https://github.com/ThinkerDreamer/kute/pull/10))

### Fixed

- **detect:** Don't bind outputs that were playing before the call
- **detect:** Only exclude outputs that were running before the call

### Packaging

- Tagged releases with a git-cliff changelog

## [0.1.0] - 2026-11-08

### Added

- Scaffold crate with config and time helpers
- Add session store and meta.toml model
- Merge per-track Whisper output into one timeline
- Render transcript.md and summary.md
- Add WAV range reading and ffmpeg helpers
- Chunk speech and transcribe tracks per chunk
- Add whisper.cpp transcriber and Silero VAD
- Add model registry with checksum-verified downloads
- Add SQLite FTS5 search index
- Add summarizer trait, prompt, and command backend
- Add OpenAI-compatible and Anthropic summarizers
- Add the processing pipeline
- Import recordings made outside kute
- Add the kute command line
- Detect and capture config, open conversation in meta, update_meta
- Process closed conversations while a session is still recording
- PipeWire graph snapshots from pw-dump --monitor
- Call detection state machine
- Live conversation splitting
- Record tracks from pw-record with live VAD
- Live sessions with conversation splitting and crash recovery
- Job queue worker with transcription retry backoff
- Daemon control socket
- Kute daemon with status, record, stop, discard, pause, resume
- Notify when the daemon writes a summary
- Session notes for the summarizer
- Per-session and per-track transcription languages ([#4](https://github.com/ThinkerDreamer/kute/pull/4))
- Kute service install, recording and failure notifications ([#6](https://github.com/ThinkerDreamer/kute/pull/6))
- Tray icon, kute doctor and kute debug trace
- Type in a call's details from the tray

### Fixed

- Address final review findings
- Stop the model download progress from flooding the terminal
- Kill the summarizer's whole process group on timeout
- Don't join the summarizer's helper threads on timeout
- Count virtual sources as mics, drop links of removed nodes
- Don't restart a call the user just stopped or discarded
- Gap fill on blocked reads only, stop captures on drop
- Recover VAD-less sessions, stop captures on drop, rebase t0
- Name the session when an import's processing fails, absolute import paths
- Discard removes the session from the index and drops its jobs
- A PipeWire hiccup or a failed capture doesn't block the call
- Audio after a lost live VAD becomes one conversation
- Say where a queued job's result goes
- The worker doesn't print partial results on the daemon's stdout
- Kute status shows sessions waiting to retry transcription
- A PipeWire disconnect doesn't lift the user's restart block
- Discarding a mic check doesn't open the search index
- Start a session on a tick, not only on a new snapshot
- Summary headings sit under their conversation
- Use UTC when the system has no time zone
- Review findings on doctor and debug trace
- Doctor accepts a matched capture.mic without a default; trace reports early pw-dump failures
- Apply tray detail edits at once, and label search hits per session
- Skip a dialog program without the execute bit
- Recording and the tray icon on Linux Mint 22 (PipeWire 1.0, Cinnamon) ([#12](https://github.com/ThinkerDreamer/kute/pull/12))

### Packaging

- Add a Nix flake for the Vulkan and CPU builds ([#5](https://github.com/ThinkerDreamer/kute/pull/5))

### Documentation

- Install, configuration, privacy, troubleshooting and contributing guides ([#1](https://github.com/ThinkerDreamer/kute/pull/1))

[0.4.0]: https://github.com/ThinkerDreamer/kute/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ThinkerDreamer/kute/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ThinkerDreamer/kute/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ThinkerDreamer/kute/releases/tag/v0.1.0

