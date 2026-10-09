# To do

Small issues found in the final review of the offline pipeline (plan 1) and deferred.
None of them loses data. Tick them off as they land.

## Before the daemon (plan 3)

These start to matter once kute runs as a long-lived process.

- [ ] **Summarizer timeout leaves grandchildren running.** `CommandSummarizer::summarize`
  (`src/summarize/command.rs`) kills only the direct child, so with `sh -c ...` or a tool that
  forks, the grandchildren and the three pipe-reader threads outlive the timeout. Spawn the
  command in its own process group (`CommandExt::process_group(0)`) and kill the group.
- [ ] **A model-load failure counts as an attempt on every conversation.** When
  `AsrProvider::get` fails, `Pipeline::process` calls `Conversation::fail` for each pending
  conversation, so `attempts` grows for all of them. Count it once per session (or not at all)
  before the daemon's retry backoff starts reading `attempts`.
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

## Packaging (AUR)

The name `kute` was free on the AUR as of 2026-10-08. Registration was closed that day, so this
also waits on an account.

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
