# To do

Small issues found in the final review of the offline pipeline (plan 1) and deferred.
None of them loses data. Tick them off as they land.

## Before the daemon (plan 3)

These start to matter once kute runs as a long-lived process.

- [x] **Summarizer timeout leaves grandchildren running.** `CommandSummarizer::summarize`
  (`src/summarize/command.rs`) now spawns the command in its own process group and, on
  timeout, kills the whole group and joins the reader threads.
- [x] **A model-load failure counts as an attempt on every conversation.** By design:
  `attempts` counts processing runs that tried the conversation, and the worker's backoff
  (plan 3) reads it that way, so a failed model load is one failed run for each waiting
  conversation.
- [x] **Temp files survive a Ctrl-C.** `Pipeline::process` now removes `.decoded-*.wav` and
  `.*.opus.tmp` before it starts.
- [x] **Search index has no schema version.** It has `PRAGMA user_version` now; `pipeline::open_index`
  drops an index from another version and reindexes every session. Bump `SCHEMA_VERSION` in
  `src/index.rs` on any schema change.

## From the first real call

Found by comparing kute's transcript of a short English/Italian test call with the
`transcribe-call` script's transcript of the same recording.

- [ ] **Vocabulary for names and jargon.** Whisper heard "kute" as "Kutu"/"Kutte" and "VAD" as
  "BAD". Add a `transcribe.vocabulary` config list (e.g. `["kute", "VAD", "meta.toml"]`) and pass
  it to Whisper through `FullParams::set_initial_prompt` for every chunk in
  `WhisperTranscriber::transcribe`. Remote backends (Deepgram, ElevenLabs) have the same idea as
  "keyterms", so keep the setting provider-agnostic.
- [ ] **Speaker names per session.** Labels are global (`[labels]`), so a 1:1 call can't show real
  names. Add `kute import ... --me Alex --others Jordan` (and a way to set them later, like
  `kute title`), store them in `meta.toml`, and fall back to `[labels]` when unset.
- [ ] **Record the language chosen for each chunk, and revisit how short chunks borrow one.** A
  short Italian "Sì" came out as English "See": chunks under 3 s take the language of the nearest
  long chunk, which can be on the other side of a language switch. Store the per-chunk language
  in the track JSON so cases like this can be diagnosed, then consider borrowing only from a
  neighbour that isn't separated by a long pause, or detecting on the short chunk plus its
  surrounding audio.

## Anytime

- [x] **`kute model download` floods the terminal.** Now redraws only when the one-decimal value
  changes, and prints a line every 10% when stderr isn't a terminal.
- [x] **`kute list` shows oldest first.** Newest first now.
- [ ] **One very long speech region isn't split.** `plan_chunks` (`src/transcribe/mod.rs`) caps
  chunk length only when joining regions, so a single VAD region over 120 s stays one chunk.
  Practically unreachable (Silero splits on 100 ms of silence) and whisper.cpp windows internally,
  but the doc comment overstates it. Set `max_speech_duration_s` on `WhisperVadParams` or split in
  `plan_chunks`.
- [ ] **A new Whisper state per call.** `WhisperTranscriber::detect_language` and `transcribe`
  (`src/transcribe/whisper.rs`) each create a `WhisperState` (KV cache + backend buffers). Keep one
  state in the transcriber; whisper.cpp resets it on every `full` call.
- [ ] **whisper.cpp logs are swallowed.** `cli::run` installs the logging hooks with no backend, so
  messages like "no GPU found, using CPU" never reach the user. Add a `-v` flag with a minimal
  logger, or report the backend in `kute doctor`.
- [ ] **`Store::open` changes permissions on an existing folder.** It sets `storage.dir` to 0700 on
  every run, so pointing it at a shared or existing directory silently changes its mode. Only
  chmod directories kute creates, or document the behavior.

## From the daemon (plan 3)

Found in the final review of plan 3 and deferred.

- [ ] **One stalled track holds live splitting back.** `LiveSession` advances the splitter to
  `min(a, b)` of what each track has analysed (`src/live.rs`), so a track that stops delivering
  audio without ending stops conversations from closing for the rest of the session. They still
  close at `finish`. A silent-track timeout would keep splitting going.
- [ ] **Sessions with conversations that failed 4+ times are re-run on every daemon start.**
  `worker::startup_jobs` queues every `processing` session, and one whose transcription failed
  for good stays `processing`, so each start pays a model load for nothing.
- [ ] **WAVs over 4 GiB fail.** hound's RIFF sizes are 32-bit: about 37 h of 16 kHz audio per
  track.
- [ ] **No `pw-dump` repeats the same log line.** The monitor reconnects every 30 s and logs "lost
  the PipeWire graph" each time. `kute doctor` (plan 4) should check for `pw-dump` and
  `pw-record`.
- [ ] **Suspend/resume during a session is untested.** The daemon's clock is monotonic and pauses
  in suspend while the wall clock jumps; check what the detector and the track offsets make of it.
- [ ] **Live VAD lost, then a crash.** After a track loses its speech detection, `finish` makes the
  audio after the last closed conversation one conversation, but nothing records that in
  `meta.toml`: `live::recover` after a crash only closes an open conversation, so the audio after
  the loss isn't transcribed unless one was open. Saving `open_conversation_ms` at the loss would
  cover it.

- [ ] **`kute list | head` panics with "Broken pipe".** `println!` panics when stdout closes early.
  Exit quietly on `BrokenPipe` in the CLI's output (or restore the default SIGPIPE behavior at
  startup).
- [ ] **A session is labeled with the first app that opened the mic.** On 2026-10-21 a Zoom call
  ran inside a session labeled "Element", because Element's video room had opened the mic first.
  Consider relabeling when the trigger app's mic closes and another app's stays open.
- [ ] **A closed conversation can be lost on a write error.** When `update_meta` fails (disk
  full) for a `Closed` conversation, the splitter has already dropped it (`src/live.rs`), so it
  never reaches `meta.toml`.
- [ ] **A failed `finish` leaves the session `recording`.** On a write error (disk full) the meta
  stays `recording` until the next daemon start, where recovery handles it. At least log it.
- [ ] **Stop, then pause + disconnect + resume, can lift the restart block.** All three together
  after `kute stop` let the stopped call be recorded again. Parked as an edge case.
- [ ] **A Tentative session can enqueue processing and then be discarded.** Only when
  `conversation_gap` is shorter than the 20 s commit time.
- [ ] **Summarizer process-group leftovers.** The normal-exit path still waits on reader joins
  if a grandchild keeps the pipes after the child exits (`src/summarize/command.rs`), and
  `process_group(0)` means Ctrl-C in a foreground `kute summarize` doesn't reach the summarizer.
- [ ] **IPC request handling.** A non-UTF-8 request gets no `Error` reply, and the read timeout
  is per read, not per request (`src/ipc.rs`).
- [ ] **Old indexes keep the binary name as the app label.** Indexes built before 45316c0 store
  `electron` instead of the app name until `kute reindex`.
- [ ] **Firefox mic check during a Firefox video.** The mic check binds the video's running
  output, which keeps the session alive (spec open question).

## Packaging

Nix comes first: a `flake.nix` in this repo, then the same derivation upstreamed to nixpkgs
(`pkgs/by-name/ku/kute/package.nix`). Whatever package ships kute must build with
`GGML_NATIVE=OFF` (see the AUR notes below).

### AUR (on hold)

The name `kute` was free on the AUR as of 2026-10-08, but new-account registration has been
closed since 2026-08-01 and probably won't reopen before Q1 or Q2 2027. These notes wait for
that.

- [ ] **Decide when to release 0.1.0.** Before the daemon, the AUR description has to promise only
  the offline pipeline (import, transcribe, summarize, search), since anyone installing `kute`
  will expect it to record calls. Or wait for plan 3.
- [ ] **Publish `kute` from tagged releases**, not `kute-git`: yay only picks up `-git` updates with
  `--devel`, and a tag is a fixed, tested snapshot. Tag with `git tag v0.1.0 && git push --tags`;
  the PKGBUILD downloads GitHub's generated tarball
  (`https://github.com/ThinkerDreamer/kute/archive/refs/tags/v$pkgver.tar.gz`). Add `kute-git`
  later (`provides=(kute)`, `conflicts=(kute)`) if people want to test unreleased work.
- [ ] **Write the PKGBUILD** following the Arch wiki's Rust package guidelines (`cargo fetch
  --locked` in `prepare()`, `cargo build --frozen --release` in `build()`), with:
  - `makedepends`: `cargo`, `cmake`, `clang` (bindgen for whisper.cpp), `vulkan-headers`,
    `shaderc`
  - `depends`: `ffmpeg`, `vulkan-icd-loader`, `gcc-libs`, `glibc`
  - `license=('MIT OR Apache-2.0')`, and install both license files to
    `/usr/share/licenses/kute/`
  - the default Vulkan build only; mention `--features cuda` in the README instead of a
    `kute-cuda` package until someone asks for one
  - no models in the package: a post-install message pointing to `kute model download`
  - `export GGML_NATIVE=OFF` in `build()`: whisper.cpp otherwise compiles with `-march=native`,
    and a package built on one CPU can crash with SIGILL on another (CI hit this, see
    `.github/workflows/ci.yml`). Local `cargo install` builds keep native for speed.
- [ ] **Test it before the first push**: build in a clean chroot with `pkgctl build` (devtools),
  which catches dependencies that only happen to be installed locally, and run `namcap` on the
  PKGBUILD and the built package.
- [ ] **Publish**: add an SSH key to the AUR account, `git clone
  ssh://aur@aur.archlinux.org/kute.git`, commit `PKGBUILD` and `.SRCINFO` (regenerate it with
  `makepkg --printsrcinfo > .SRCINFO` on every change), push to `master`.

## From plan 4a

- [ ] **Notification buttons vanish in the KDE history.** Plasma drops the actions of a
  notification once it leaves the popup, so "Open summary" can't be clicked from the history.
  A resident note keeps them but stays on screen; an XDG portal `OpenURI` call wouldn't help
  either, since the click never reaches kute.
- [ ] **No activation token for the opener.** The notification server sends one with the click
  (`ActivationToken`, spec 1.2), but notify-rust doesn't pass it on, so on Wayland the editor may
  open behind other windows. Set `XDG_ACTIVATION_TOKEN` for the child once it's available.
