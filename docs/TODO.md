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
- [ ] **Temp files survive a Ctrl-C.** `.decoded-<track>.wav` (`AudioSource`, can be ~1 GB) and
  `.<track>.opus.tmp` (`compress_tracks`) stay in the session folder if kute is interrupted.
  Remove stale dotfiles at the start of `Pipeline::process`.
- [ ] **Search index has no schema version.** `src/index.rs` uses `CREATE ... IF NOT EXISTS`, so a
  later change to the FTS tables can't migrate. Set `PRAGMA user_version` and rebuild the index
  (it's derived) when the version differs.

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

- [ ] **`kute model download` floods the terminal.** The progress callback in `src/cli.rs` prints on
  every 64 KiB chunk with whole-MB rounding ("0 / 0 MB" for the VAD model, ~25,000 lines for
  large-v3-turbo when stderr isn't a terminal). Print only when the shown value changes, use one
  decimal, end each file with a newline.
- [ ] **`kute list` shows oldest first.** Spec section 8 says newest first; reverse the iteration in
  `list` (`src/cli.rs`).
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
