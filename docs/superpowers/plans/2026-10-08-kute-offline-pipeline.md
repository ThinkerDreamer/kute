# kute Offline Pipeline Implementation Plan (plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `kute` CLI that imports a pair of call recordings (my mic / everyone else), transcribes them locally with whisper.cpp, merges them into one timestamped transcript, summarizes each conversation with a configurable backend, compresses the audio to Opus, and makes everything searchable.

**Architecture:** One Rust crate (library + `kute` binary). Session folders on disk are the source of truth (`meta.toml`, per-track Whisper JSON, `transcript.md`, `summaries/NN.md`); a SQLite FTS5 index is derived from them. Transcription runs Silero VAD first, cuts the audio into speech chunks, then runs Whisper per chunk, so Whisper never sees long silences and word timestamps stay correct. Summarizers sit behind a `Summarizer` trait (command, OpenAI-compatible, Anthropic), transcription behind `Transcriber` / `SpeechDetector` traits.

**Tech Stack:** Rust 2024, whisper-rs 0.16 (whisper.cpp, Vulkan by default), rusqlite 0.40 (bundled SQLite with FTS5), ureq 3 (blocking HTTP), jiff 0.2 (time), hound (WAV), clap 4, serde + toml 1 + serde_json, sha2, wait-timeout; ffmpeg/ffprobe as runtime tools.

**Spec:** `docs/superpowers/specs/2026-10-08-kute-design.md`. This plan covers the offline part of spec sections 4, 6, 7, 8, 9 and 10. The daemon (section 5, IPC, tray) is plans 3 and 4.

## Global Constraints

- Single crate `kute`, `edition = "2024"`, `rust-version = "1.88"`, license `MIT OR Apache-2.0`.
- No async runtime (no tokio). HTTP is blocking `ureq` 3.
- Dates and durations use `jiff` only (no chrono, no humantime). Wall-clock times are `jiff::Zoned`, serialized as RFC 9557 (`2026-10-08T09:03:12.418+02:00[Europe/Rome]`), truncated to milliseconds. Session time is `u64` milliseconds since `started_at`.
- Storage root default `$XDG_DATA_HOME/kute` (`dirs::data_dir()/kute`); root, `sessions/` and each session dir are mode `0700`.
- Every write of `meta.toml`, track JSON, `transcript.md`, `summary.md`, `summaries/NN.md` goes through `store::write_atomic` (temp file + rename).
- Working audio is 16 kHz mono 16-bit PCM WAV. Archive audio is Ogg Opus, `libopus`, 24 kbit/s, `-application voip`. WAVs are deleted only after each Opus file's duration matches its WAV within 500 ms.
- Whisper models come from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/<file>`, the VAD model from `https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin`; downloads are verified against the SHA-256 values in Task 8. Default model `large-v3-turbo`.
- Whisper's built-in VAD (`FullParams::enable_vad`) must NOT be used: whisper.cpp remaps segment timestamps but not token timestamps, so word times would be wrong. kute runs `WhisperVadContext` itself and transcribes chunks. Flash attention stays off.
- Labels default `Me` / `Others`; `summary.language` default `auto`; `summary.min_words` default `80`; summarizer timeout default `10m`.
- Anthropic backend: `POST {base}/v1/messages`, headers `x-api-key`, `anthropic-version: 2023-06-01`, `anthropic-beta: server-side-fallback-2026-07-01`; body has `"fallbacks": "default"`, `"max_tokens": 16000`; default model `claude-opus-5-5`; `stop_reason == "refusal"` is an error; only `type == "text"` content blocks are read.
- Committed files contain no real people's names, devices or personal config. Example user name is `Alex`.
- Before every commit: `cargo fmt` and `cargo clippy --all-targets -- -D warnings` must be clean. Every commit message ends with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Tests needing real models are `#[ignore]` and read the model directory from the env var `KUTE_TEST_MODELS` (containing `ggml-tiny.bin` and `ggml-silero-v6.2.0.bin`). Tests may assume `ffmpeg` and `ffprobe` on `PATH`.

## Review Focus

1. **Italian accented characters split across Whisper tokens** ("perché", "città" arrive as separate byte pieces): the words must come out intact, never with `�`. Pinned by `tokens_to_words_rejoins_split_utf8` in Task 7.
2. **Search queries containing FTS5 syntax** (`l'ho`, `C++`, `"`, `-x`, `AND`): no SQL error, matched literally, and accent-insensitive (`perche` finds `perché`). Pinned by `search_handles_fts_syntax_literally` in Task 9.
3. **One side silent for a whole conversation** (muted the whole call, or nobody else spoke): the transcript shows only the other side, that track's language is `None`, nothing panics. Pinned by `no_speech_means_no_calls` (Task 6) and `silent_track_still_produces_transcript` (Task 12).
4. **Compression failing halfway**: WAVs stay, `compression.stage = "failed"` with the error, and a later `kute process` finishes it. Pinned by `failed_compression_keeps_wavs_and_can_resume` in Task 12.
5. **Tracks of different lengths / conversation ranges past a track's end**: reads clamp to the file, times stay correct, nothing panics. Pinned by `read_range_clamps_past_end` (Task 5) and `track_offset_and_short_track_are_handled` (Task 12).

---

### Task 1: Crate scaffold, config, time helpers

**Files:**
- Create: `Cargo.toml`, `.gitignore`, `LICENSE-MIT`, `LICENSE-APACHE`, `src/lib.rs`, `src/config.rs`, `src/time.rs`

**Interfaces:**
- Produces:
  - `config::Config { user: UserConfig { name: String }, labels: Labels { me: String, others: String }, storage: StorageConfig { dir: Option<PathBuf> }, transcribe: TranscribeConfig { model: String, language: String, gpu: bool }, summary: SummaryConfig }`
  - `config::SummaryConfig { backend: SummaryBackend, language: String, min_words: usize, prompt: Option<PathBuf>, command: Vec<String>, timeout: jiff::SignedDuration, base_url: Option<String>, model: Option<String>, api_key_env: Option<String> }`
  - `config::SummaryBackend { None, Openai, Anthropic, Command }`
  - `Config::load(path: Option<&Path>) -> Result<Config>`, `Config::parse(&str) -> Result<Config>`, `Config::data_dir(&self) -> Result<PathBuf>`, `config::expand_tilde(&Path) -> PathBuf`
  - `time::now() -> Zoned`, `time::truncate_ms(&Zoned) -> Zoned`, `time::parse_start(&str) -> Result<Zoned>`, `time::add_ms(&Zoned, u64) -> Zoned`, `time::ms_between(&Zoned, &Zoned) -> u64`, `time::format_duration(u64) -> String`, `time::session_id(&Zoned) -> String`

- [ ] **Step 1: Create the crate files**

`Cargo.toml`:
```toml
[package]
name = "kute"
version = "0.1.0"
edition = "2024"
rust-version = "1.88"
description = "Records, transcribes and summarizes your calls, locally"
license = "MIT OR Apache-2.0"
readme = "README.md"
keywords = ["transcription", "whisper", "pipewire", "meetings"]
categories = ["command-line-utilities", "multimedia::audio"]

[features]
default = ["vulkan"]
vulkan = ["whisper-rs/vulkan"]
cuda = ["whisper-rs/cuda"]

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
dirs = "7"
hound = "3"
jiff = { version = "0.2", features = ["serde"] }
rusqlite = { version = "0.40", features = ["bundled"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
toml = "1"
ureq = { version = "3", features = ["json"] }
wait-timeout = "0.2"
whisper-rs = "0.16"

[dev-dependencies]
mockito = "1"
tempfile = "3"
```

`.gitignore`:
```
/target
```

`LICENSE-MIT`:
```
MIT License

Copyright (c) 2026 The kute contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`LICENSE-APACHE`: fetch the canonical text.
```bash
curl -sSfL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE-APACHE
head -3 LICENSE-APACHE   # expect "Apache License" / "Version 2.0, January 2004"
```

`src/lib.rs`:
```rust
//! kute: records, transcribes and summarizes calls, locally.

pub mod config;
pub mod time;
```

- [ ] **Step 2: Write the failing tests for config and time**

`src/config.rs` (tests only for now, the module body comes in Step 4):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_gives_defaults() {
        let c = Config::parse("").unwrap();
        assert_eq!(c.labels.me, "Me");
        assert_eq!(c.labels.others, "Others");
        assert_eq!(c.transcribe.model, "large-v3-turbo");
        assert_eq!(c.transcribe.language, "auto");
        assert!(c.transcribe.gpu);
        assert_eq!(c.summary.backend, SummaryBackend::None);
        assert_eq!(c.summary.language, "auto");
        assert_eq!(c.summary.min_words, 80);
        assert_eq!(c.summary.timeout, jiff::SignedDuration::from_secs(600));
        assert!(c.storage.dir.is_none());
    }

    #[test]
    fn full_example_parses() {
        let c = Config::parse(
            r#"
            [user]
            name = "Alex"
            [labels]
            me = "Alex"
            others = "Team"
            [storage]
            dir = "~/Calls"
            [transcribe]
            model = "small"
            language = "it"
            gpu = false
            [summary]
            backend = "command"
            command = ["claude", "-p"]
            language = "en"
            min_words = 10
            timeout = "2m"
            "#,
        )
        .unwrap();
        assert_eq!(c.user.name, "Alex");
        assert_eq!(c.labels.others, "Team");
        assert_eq!(c.transcribe.model, "small");
        assert!(!c.transcribe.gpu);
        assert_eq!(c.summary.backend, SummaryBackend::Command);
        assert_eq!(c.summary.command, vec!["claude", "-p"]);
        assert_eq!(c.summary.timeout, jiff::SignedDuration::from_secs(120));
        let home = dirs::home_dir().unwrap();
        assert_eq!(c.data_dir().unwrap(), home.join("Calls"));
    }

    #[test]
    fn unknown_backend_is_an_error() {
        let err = Config::parse("[summary]\nbackend = \"gpt\"").unwrap_err();
        assert!(format!("{err:#}").contains("gpt"), "{err:#}");
    }

    #[test]
    fn missing_default_file_is_fine_but_missing_explicit_file_is_not() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope.toml");
        assert!(Config::load(Some(&missing)).is_err());
    }

    #[test]
    fn expand_tilde_only_touches_leading_tilde() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_tilde(Path::new("~/a/b")), home.join("a/b"));
        assert_eq!(expand_tilde(Path::new("/x/~/y")), PathBuf::from("/x/~/y"));
    }
}
```

`src/time.rs` (tests only for now):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_start_accepts_zoned_offset_and_naive_forms() {
        let z = parse_start("2026-10-08T09:03:12.418+02:00[+02:00]").unwrap();
        assert_eq!(z.time(), jiff::civil::time(9, 3, 12, 418_000_000));

        let z = parse_start("2026-10-08T09:03:12+02:00").unwrap();
        assert_eq!(z.timestamp(), "2026-10-08T07:03:12Z".parse::<jiff::Timestamp>().unwrap());

        for s in ["2026-10-08T15:00:12", "2026-10-08 15:00:12", " 2026-10-08T15:00:12\n"] {
            let z = parse_start(s).unwrap();
            assert_eq!(z.datetime().to_string(), "2026-10-08T15:00:12", "{s:?}");
        }
        assert!(parse_start("yesterday").is_err());
    }

    #[test]
    fn add_and_measure_ms() {
        let a = parse_start("2026-10-08T09:00:00+02:00[+02:00]").unwrap();
        let b = add_ms(&a, 90_500);
        assert_eq!(b, "2026-10-08T09:01:30.5+02:00[+02:00]".parse::<Zoned>().unwrap());
        assert_eq!(ms_between(&a, &b), 90_500);
        assert_eq!(ms_between(&b, &a), 0);
    }

    #[test]
    fn format_duration_cases() {
        assert_eq!(format_duration(0), "<1 min");
        assert_eq!(format_duration(59_999), "<1 min");
        assert_eq!(format_duration(60_000), "1 min");
        assert_eq!(format_duration(3_599_999), "59 min");
        assert_eq!(format_duration(31_020_000), "8 h 37 min");
    }

    #[test]
    fn session_id_and_truncation() {
        let z = parse_start("2026-10-08T09:03:12.418999+02:00[+02:00]").unwrap();
        assert_eq!(session_id(&z), "2026-10-08T090312");
        assert_eq!(truncate_ms(&z).subsec_nanosecond(), 418_000_000);
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --lib config:: time::`
Expected: compile errors (`Config`, `parse_start`, ... not found). The first build compiles whisper.cpp and takes several minutes.

- [ ] **Step 4: Implement config and time**

Top of `src/config.rs` (above the tests):
```rust
//! User configuration (`~/.config/kute/config.toml`). Every key has a default.

use anyhow::{Context, Result, bail};
use jiff::SignedDuration;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub user: UserConfig,
    pub labels: Labels,
    pub storage: StorageConfig,
    pub transcribe: TranscribeConfig,
    pub summary: SummaryConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct UserConfig {
    /// Used in summary prompts as the person behind the "me" track.
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Labels {
    pub me: String,
    pub others: String,
}

impl Default for Labels {
    fn default() -> Self {
        Self { me: "Me".into(), others: "Others".into() }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TranscribeConfig {
    /// A model name from `kute model list`, or a path to a ggml `.bin` file.
    pub model: String,
    /// `auto`, or a Whisper language code such as `it`.
    pub language: String,
    pub gpu: bool,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self { model: "large-v3-turbo".into(), language: "auto".into(), gpu: true }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SummaryBackend {
    #[default]
    None,
    Openai,
    Anthropic,
    Command,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SummaryConfig {
    pub backend: SummaryBackend,
    /// `auto` (dominant language of the transcript) or a language code.
    pub language: String,
    pub min_words: usize,
    pub prompt: Option<PathBuf>,
    pub command: Vec<String>,
    pub timeout: SignedDuration,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub api_key_env: Option<String>,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            backend: SummaryBackend::None,
            language: "auto".into(),
            min_words: 80,
            prompt: None,
            command: Vec::new(),
            timeout: SignedDuration::from_secs(600),
            base_url: None,
            model: None,
            api_key_env: None,
        }
    }
}

impl Config {
    pub fn default_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("kute").join("config.toml"))
    }

    /// Loads `path`, or the default path. A missing default file means "all defaults";
    /// a missing explicit file is an error.
    pub fn load(path: Option<&Path>) -> Result<Config> {
        let path = match path {
            Some(p) => {
                if !p.is_file() {
                    bail!("config file {} does not exist", p.display());
                }
                p.to_path_buf()
            }
            None => match Self::default_path() {
                Some(p) if p.is_file() => p,
                _ => return Ok(Config::default()),
            },
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Config> {
        Ok(toml::from_str(text)?)
    }

    pub fn data_dir(&self) -> Result<PathBuf> {
        match &self.storage.dir {
            Some(d) => Ok(expand_tilde(d)),
            None => dirs::data_dir()
                .map(|d| d.join("kute"))
                .context("no XDG data directory found; set storage.dir in the config"),
        }
    }
}

/// Expands a leading `~` to the home directory.
pub fn expand_tilde(p: &Path) -> PathBuf {
    if let Ok(rest) = p.strip_prefix("~") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    p.to_path_buf()
}
```

Top of `src/time.rs`:
```rust
//! Wall-clock helpers. Session time is `u64` milliseconds since a session's `started_at`.

use anyhow::{Result, bail};
use jiff::{RoundMode, SignedDuration, Timestamp, Unit, Zoned, ZonedRound, civil, tz::TimeZone};

pub fn now() -> Zoned {
    truncate_ms(&Zoned::now())
}

pub fn truncate_ms(z: &Zoned) -> Zoned {
    z.round(ZonedRound::new().smallest(Unit::Millisecond).mode(RoundMode::Trunc))
        .expect("truncating to milliseconds cannot overflow")
}

/// Parses a start time: RFC 9557 (`...+02:00[Europe/Rome]`), RFC 3339 with an offset,
/// or a naive local date-time (`2026-10-08T15:00:12`, `2026-10-08 15:00`), which is
/// interpreted in the system time zone.
pub fn parse_start(s: &str) -> Result<Zoned> {
    let s = s.trim();
    if let Ok(z) = s.parse::<Zoned>() {
        return Ok(truncate_ms(&z));
    }
    if let Ok(t) = s.parse::<Timestamp>() {
        return Ok(truncate_ms(&t.to_zoned(TimeZone::system())));
    }
    if let Ok(dt) = s.parse::<civil::DateTime>() {
        return Ok(truncate_ms(&dt.to_zoned(TimeZone::system())?));
    }
    bail!("cannot read {s:?} as a date and time (example: 2026-10-08T15:00:00)")
}

pub fn add_ms(z: &Zoned, ms: u64) -> Zoned {
    z.checked_add(SignedDuration::from_millis(ms as i64))
        .expect("session offsets fit in jiff's range")
}

/// Milliseconds from `a` to `b`, or 0 if `b` is before `a`.
pub fn ms_between(a: &Zoned, b: &Zoned) -> u64 {
    b.duration_since(a).as_millis().clamp(0, u64::MAX as i128) as u64
}

pub fn format_duration(ms: u64) -> String {
    let mins = ms / 60_000;
    match mins {
        0 => "<1 min".to_string(),
        1..=59 => format!("{mins} min"),
        _ => format!("{} h {} min", mins / 60, mins % 60),
    }
}

/// Session IDs are the local start time: `2026-10-08T090312`.
pub fn session_id(z: &Zoned) -> String {
    z.strftime("%Y-%m-%dT%H%M%S").to_string()
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib config:: time::`
Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add Cargo.toml Cargo.lock .gitignore LICENSE-MIT LICENSE-APACHE src/
git commit -m "feat: scaffold crate with config and time helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Session store and `meta.toml`

**Files:**
- Create: `src/store/mod.rs`, `src/store/meta.rs`, `src/store/tracks.rs`
- Modify: `src/lib.rs` (add `pub mod store;`)

**Interfaces:**
- Consumes: `time::session_id`, `time::ms_between`
- Produces:
  - `store::Track { Me, Others }` with `Track::BOTH: [Track; 2]`, `Track::as_str(self) -> &'static str` (`"me"`, `"others"`)
  - `store::Store::open(impl Into<PathBuf>) -> Result<Store>`, `.root()`, `.sessions_dir()`, `.models_dir()`, `.index_path()`, `.create_session(&Zoned) -> Result<Session>`, `.session(&str) -> Result<Session>`, `.list() -> Result<Vec<Session>>`, `.resolve(&str) -> Result<Session>`
  - `store::Session { id: String, dir: PathBuf }` with `meta_path`, `load_meta() -> Result<Meta>`, `save_meta(&Meta) -> Result<()>`, `wav_path(Track)`, `opus_path(Track)`, `json_path(Track)`, `transcript_path()`, `summary_path()`, `summaries_dir()`, `conversation_summary_path(u32)`, `load_track(Track) -> Result<TrackTranscript>`, `save_track(Track, &TrackTranscript) -> Result<()>`
  - `store::write_atomic(&Path, &[u8]) -> Result<()>`
  - `store::Meta` and friends exactly as in Step 3 (`SessionState`, `Trigger`, `TriggerKind`, `Tracks`, `TrackMeta`, `SummaryMeta`, `Conversation`, `Stage`, `Step`, `Languages`, `Compression`, `CompressionStage`)
  - `store::TrackTranscript { segments: Vec<Segment> }`, `Segment { conversation: u32, start_ms: u64, end_ms: u64, text: String, words: Vec<Word> }`, `Word { start_ms: u64, end_ms: u64, text: String }`, `TrackTranscript::replace_conversation(&mut self, u32, Vec<Segment>)`, `TrackTranscript::conversation(&self, u32) -> Vec<&Segment>`

- [ ] **Step 1: Write the failing tests**

At the bottom of `src/store/mod.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn at(s: &str) -> Zoned {
        s.parse().unwrap()
    }

    fn store() -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        (tmp, store)
    }

    fn sample_meta(id: &str, started: &Zoned) -> Meta {
        let mut m = Meta::new(
            id.to_string(),
            started.clone(),
            Trigger { kind: TriggerKind::Import, app_name: None, binary: None },
            "auto".into(),
        );
        m.state = SessionState::Processing;
        m.conversations.push(Conversation::new(1, 0, 60_000));
        m
    }

    fn create_with_meta(store: &Store, t: &str) -> Session {
        let z = at(t);
        let s = store.create_session(&z).unwrap();
        s.save_meta(&sample_meta(&s.id, &z)).unwrap();
        s
    }

    #[test]
    fn create_session_uses_start_time_and_suffixes_collisions() {
        let (_tmp, store) = store();
        let t = at("2026-10-08T09:03:12.418+02:00[+02:00]");
        let a = store.create_session(&t).unwrap();
        let b = store.create_session(&t).unwrap();
        assert_eq!(a.id, "2026-10-08T090312");
        assert_eq!(b.id, "2026-10-08T090312-2");
        assert!(a.dir.is_dir() && b.dir.is_dir());
    }

    #[test]
    fn store_and_session_dirs_are_private() {
        let (_tmp, store) = store();
        let s = store.create_session(&at("2026-10-08T09:00:00+02:00[+02:00]")).unwrap();
        for p in [store.root().to_path_buf(), store.sessions_dir(), s.dir.clone()] {
            let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "{}", p.display());
        }
    }

    #[test]
    fn meta_round_trips_through_toml() {
        let (_tmp, store) = store();
        let z = at("2026-10-08T09:03:12.418+02:00[+02:00]");
        let s = store.create_session(&z).unwrap();
        let mut m = sample_meta(&s.id, &z);
        m.title = Some("Planning".into());
        m.conversations[0].fail(Step::Summarize, &anyhow::anyhow!("boom"));
        s.save_meta(&m).unwrap();
        assert_eq!(s.load_meta().unwrap(), m);
        let text = std::fs::read_to_string(s.meta_path()).unwrap();
        assert!(text.contains("stage = \"failed\""), "{text}");
        assert!(text.contains("failed_step = \"summarize\""), "{text}");
    }

    #[test]
    fn resolve_by_exact_prefix_and_last() {
        let (_tmp, store) = store();
        create_with_meta(&store, "2026-10-07T10:00:00+02:00[+02:00]");
        create_with_meta(&store, "2026-10-08T09:00:00+02:00[+02:00]");
        create_with_meta(&store, "2026-10-08T15:30:00+02:00[+02:00]");
        // A folder without meta.toml is not a session.
        std::fs::create_dir(store.sessions_dir().join("2026-10-09T000000")).unwrap();

        assert_eq!(store.list().unwrap().len(), 3);
        assert_eq!(store.resolve("last").unwrap().id, "2026-10-08T153000");
        assert_eq!(store.resolve("2026-10-07").unwrap().id, "2026-10-07T100000");
        assert_eq!(store.resolve("2026-10-08T090000").unwrap().id, "2026-10-08T090000");
        assert!(format!("{:#}", store.resolve("2026-10-08").unwrap_err()).contains("matches 2"));
        assert!(store.resolve("2025").is_err());
    }

    #[test]
    fn write_atomic_replaces_and_leaves_no_temp_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.txt");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "two");
        let names: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names, vec![std::ffi::OsString::from("a.txt")]);
    }

    #[test]
    fn tracks_load_empty_when_missing_and_round_trip() {
        let (_tmp, store) = store();
        let s = store.create_session(&at("2026-10-08T09:00:00+02:00[+02:00]")).unwrap();
        assert!(s.load_track(Track::Me).unwrap().segments.is_empty());
        let mut t = TrackTranscript::default();
        t.replace_conversation(
            2,
            vec![Segment { conversation: 2, start_ms: 5, end_ms: 9, text: " hi".into(), words: vec![] }],
        );
        s.save_track(Track::Me, &t).unwrap();
        assert_eq!(s.load_track(Track::Me).unwrap(), t);
    }
}
```

At the bottom of `src/store/tracks.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn seg(conversation: u32, start_ms: u64) -> Segment {
        Segment { conversation, start_ms, end_ms: start_ms + 10, text: format!("c{conversation}"), words: vec![] }
    }

    #[test]
    fn replace_conversation_swaps_only_that_conversation_and_keeps_order() {
        let mut t = TrackTranscript::default();
        t.replace_conversation(1, vec![seg(1, 100), seg(1, 0)]);
        t.replace_conversation(2, vec![seg(2, 500)]);
        t.replace_conversation(1, vec![seg(1, 50)]);
        let starts: Vec<_> = t.segments.iter().map(|s| (s.conversation, s.start_ms)).collect();
        assert_eq!(starts, vec![(1, 50), (2, 500)]);
        assert_eq!(t.conversation(2).len(), 1);
        assert!(t.conversation(3).is_empty());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib store::`
Expected: compile errors (module `store` not found). Add `pub mod store;` to `src/lib.rs` first so the errors point at the missing items.

- [ ] **Step 3: Implement `meta.rs`**

`src/store/meta.rs`:
```rust
//! `meta.toml`: everything about a session except audio and text.

use super::Track;
use crate::time;
use jiff::Zoned;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub kute_version: String,
    pub started_at: Zoned,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<Zoned>,
    pub state: SessionState,
    #[serde(default)]
    pub interrupted: bool,
    pub trigger: Trigger,
    #[serde(default)]
    pub tracks: Tracks,
    pub summary: SummaryMeta,
    #[serde(default)]
    pub conversations: Vec<Conversation>,
    #[serde(default)]
    pub compression: Compression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    Recording,
    Processing,
    Done,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trigger {
    pub kind: TriggerKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TriggerKind {
    App,
    Manual,
    Import,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tracks {
    #[serde(default)]
    pub me: TrackMeta,
    #[serde(default)]
    pub others: TrackMeta,
}

impl Tracks {
    pub fn get(&self, track: Track) -> &TrackMeta {
        match track {
            Track::Me => &self.me,
            Track::Others => &self.others,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrackMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// How much later than session time 0 this track's first sample arrived.
    #[serde(default)]
    pub offset_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryMeta {
    pub language: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    pub n: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub stage: Stage,
    #[serde(default)]
    pub languages: Languages,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whisper_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summarizer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_step: Option<Step>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub attempts: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Pending,
    Transcribed,
    Summarized,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Step {
    Transcribe,
    Summarize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Languages {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub me: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub others: Option<String>,
}

impl Languages {
    pub fn set(&mut self, track: Track, language: Option<String>) {
        match track {
            Track::Me => self.me = language,
            Track::Others => self.others = language,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Compression {
    pub stage: CompressionStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompressionStage {
    #[default]
    Pending,
    Done,
    Failed,
}

impl Meta {
    pub fn new(id: String, started_at: Zoned, trigger: Trigger, summary_language: String) -> Meta {
        Meta {
            id,
            title: None,
            kute_version: env!("CARGO_PKG_VERSION").to_string(),
            started_at,
            ended_at: None,
            state: SessionState::Recording,
            interrupted: false,
            trigger,
            tracks: Tracks::default(),
            summary: SummaryMeta { language: summary_language },
            conversations: Vec::new(),
            compression: Compression::default(),
        }
    }

    pub fn duration_ms(&self) -> Option<u64> {
        self.ended_at.as_ref().map(|end| time::ms_between(&self.started_at, end))
    }

    /// What triggered the session, for headers and listings.
    pub fn app_label(&self) -> String {
        match self.trigger.kind {
            TriggerKind::App => self
                .trigger
                .binary
                .clone()
                .or_else(|| self.trigger.app_name.clone())
                .unwrap_or_else(|| "app".into()),
            TriggerKind::Manual => "manual".into(),
            TriggerKind::Import => "import".into(),
        }
    }

    pub fn conversation(&self, n: u32) -> Option<&Conversation> {
        self.conversations.iter().find(|c| c.n == n)
    }

    pub fn conversation_mut(&mut self, n: u32) -> Option<&mut Conversation> {
        self.conversations.iter_mut().find(|c| c.n == n)
    }
}

impl Conversation {
    pub fn new(n: u32, start_ms: u64, end_ms: u64) -> Conversation {
        Conversation {
            n,
            start_ms,
            end_ms,
            stage: Stage::Pending,
            languages: Languages::default(),
            whisper_model: None,
            summarizer: None,
            failed_step: None,
            error: None,
            attempts: 0,
        }
    }

    pub fn fail(&mut self, step: Step, err: &anyhow::Error) {
        self.stage = Stage::Failed;
        self.failed_step = Some(step);
        self.error = Some(format!("{err:#}"));
        self.attempts += 1;
    }

    pub fn clear_error(&mut self) {
        self.failed_step = None;
        self.error = None;
    }

    pub fn needs_transcription(&self) -> bool {
        self.stage == Stage::Pending
            || (self.stage == Stage::Failed && self.failed_step == Some(Step::Transcribe))
    }

    pub fn needs_summary(&self) -> bool {
        self.stage == Stage::Transcribed
            || (self.stage == Stage::Failed && self.failed_step == Some(Step::Summarize))
    }

    /// True once nothing more will happen to this conversation without a user command.
    pub fn is_settled(&self) -> bool {
        matches!(self.stage, Stage::Summarized | Stage::Skipped | Stage::Failed)
    }
}
```

- [ ] **Step 4: Implement `tracks.rs`**

`src/store/tracks.rs`:
```rust
//! `me.json` / `others.json`: Whisper output in session time, tagged by conversation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrackTranscript {
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub conversation: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default)]
    pub words: Vec<Word>,
}

/// A word as Whisper produced it. `text` keeps Whisper's leading space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

impl TrackTranscript {
    pub fn replace_conversation(&mut self, n: u32, segments: Vec<Segment>) {
        self.segments.retain(|s| s.conversation != n);
        self.segments.extend(segments);
        self.segments.sort_by_key(|s| (s.start_ms, s.conversation));
    }

    pub fn conversation(&self, n: u32) -> Vec<&Segment> {
        self.segments.iter().filter(|s| s.conversation == n).collect()
    }
}
```

- [ ] **Step 5: Implement `store/mod.rs`**

Top of `src/store/mod.rs`:
```rust
//! Session folders on disk: the source of truth for everything kute knows.

mod meta;
mod tracks;

pub use meta::*;
pub use tracks::*;

use crate::time;
use anyhow::{Context, Result, bail};
use jiff::Zoned;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Track {
    Me,
    Others,
}

impl Track {
    pub const BOTH: [Track; 2] = [Track::Me, Track::Others];

    pub fn as_str(self) -> &'static str {
        match self {
            Track::Me => "me",
            Track::Others => "others",
        }
    }
}

pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Opens (creating if needed) the storage root. Root, `sessions/` and `models/` are 0700.
    pub fn open(root: impl Into<PathBuf>) -> Result<Store> {
        let root = root.into();
        create_private_dir(&root)?;
        create_private_dir(&root.join("sessions"))?;
        create_private_dir(&root.join("models"))?;
        Ok(Store { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub fn index_path(&self) -> PathBuf {
        self.root.join("index.sqlite")
    }

    /// Creates an empty session folder named after `started_at`, adding `-2`, `-3`, ...
    /// if that name is taken.
    pub fn create_session(&self, started_at: &Zoned) -> Result<Session> {
        let base = time::session_id(started_at);
        for i in 1u32.. {
            let id = if i == 1 { base.clone() } else { format!("{base}-{i}") };
            let dir = self.sessions_dir().join(&id);
            match fs::create_dir(&dir) {
                Ok(()) => {
                    set_private(&dir)?;
                    return Ok(Session { id, dir });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).with_context(|| format!("creating {}", dir.display())),
            }
        }
        unreachable!("u32 range exhausted")
    }

    pub fn session(&self, id: &str) -> Result<Session> {
        let s = Session { id: id.to_string(), dir: self.sessions_dir().join(id) };
        if !s.meta_path().is_file() {
            bail!("no session {id:?}");
        }
        Ok(s)
    }

    /// All sessions (folders with a `meta.toml`), oldest first.
    pub fn list(&self) -> Result<Vec<Session>> {
        let mut out = Vec::new();
        let dir = self.sessions_dir();
        for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry?;
            let Some(id) = entry.file_name().to_str().map(str::to_string) else { continue };
            let s = Session { id, dir: entry.path() };
            if s.meta_path().is_file() {
                out.push(s);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// Finds a session by full ID, unique prefix, or `last`.
    pub fn resolve(&self, query: &str) -> Result<Session> {
        let all = self.list()?;
        if query == "last" {
            return all.into_iter().last().context("there are no sessions yet");
        }
        if let Some(s) = all.iter().find(|s| s.id == query) {
            return Ok(s.clone());
        }
        let mut matches: Vec<Session> = all.into_iter().filter(|s| s.id.starts_with(query)).collect();
        match matches.len() {
            0 => bail!("no session matches {query:?}"),
            1 => Ok(matches.remove(0)),
            n => bail!("{query:?} matches {n} sessions; use a longer prefix"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub dir: PathBuf,
}

impl Session {
    pub fn meta_path(&self) -> PathBuf {
        self.dir.join("meta.toml")
    }

    pub fn load_meta(&self) -> Result<Meta> {
        let path = self.meta_path();
        let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save_meta(&self, meta: &Meta) -> Result<()> {
        write_atomic(&self.meta_path(), toml::to_string_pretty(meta)?.as_bytes())
    }

    pub fn wav_path(&self, track: Track) -> PathBuf {
        self.dir.join(format!("{}.wav", track.as_str()))
    }

    pub fn opus_path(&self, track: Track) -> PathBuf {
        self.dir.join(format!("{}.opus", track.as_str()))
    }

    pub fn json_path(&self, track: Track) -> PathBuf {
        self.dir.join(format!("{}.json", track.as_str()))
    }

    pub fn transcript_path(&self) -> PathBuf {
        self.dir.join("transcript.md")
    }

    pub fn summary_path(&self) -> PathBuf {
        self.dir.join("summary.md")
    }

    pub fn summaries_dir(&self) -> PathBuf {
        self.dir.join("summaries")
    }

    pub fn conversation_summary_path(&self, n: u32) -> PathBuf {
        self.summaries_dir().join(format!("{n:02}.md"))
    }

    pub fn load_track(&self, track: Track) -> Result<TrackTranscript> {
        let path = self.json_path(track);
        if !path.is_file() {
            return Ok(TrackTranscript::default());
        }
        let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save_track(&self, track: Track, transcript: &TrackTranscript) -> Result<()> {
        write_atomic(&self.json_path(track), serde_json::to_string_pretty(transcript)?.as_bytes())
    }
}

/// Writes `bytes` to a temp file next to `path`, syncs it, then renames it into place.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().context("path has no parent directory")?;
    let name = path.file_name().context("path has no file name")?.to_string_lossy();
    let tmp = dir.join(format!(".{name}.tmp"));
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))
}

fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;
    set_private(path)
}

fn set_private(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("setting permissions on {}", path.display()))
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --lib store::`
Expected: 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: add session store and meta.toml model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Merging two tracks into one timeline

**Files:**
- Create: `src/merge.rs`
- Modify: `src/lib.rs` (add `pub mod merge;`)

**Interfaces:**
- Consumes: `store::{Segment, Track, Word}`
- Produces: `merge::Line { start_ms: u64, track: Track, text: String }`, `merge::SPLIT_GAP_MS: u64 = 1_000`, `merge::track_lines<'a>(impl IntoIterator<Item = &'a Segment>, Track) -> Vec<Line>`, `merge::merge<'a>(me: impl IntoIterator<Item = &'a Segment>, others: impl IntoIterator<Item = &'a Segment>) -> Vec<Line>`

This is the port of the `transcribe-call` merge: Whisper segments can span long pauses, so a segment is split wherever two words are more than 1 s apart (strictly more), and a lone first word, which Whisper tends to pin to the segment start, is moved to the words that follow it.

- [ ] **Step 1: Write the failing tests**

`src/merge.rs` (tests first):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn w(start_ms: u64, end_ms: u64, text: &str) -> Word {
        Word { start_ms, end_ms, text: text.into() }
    }

    fn seg(start_ms: u64, text: &str, words: Vec<Word>) -> Segment {
        Segment { conversation: 1, start_ms, end_ms: start_ms + 1, text: text.into(), words }
    }

    fn texts(lines: &[Line]) -> Vec<(u64, &str)> {
        lines.iter().map(|l| (l.start_ms, l.text.as_str())).collect()
    }

    #[test]
    fn segment_without_words_becomes_one_line() {
        let s = [seg(500, "  Ciao a tutti ", vec![])];
        assert_eq!(texts(&track_lines(&s, Track::Me)), vec![(500, "Ciao a tutti")]);
    }

    #[test]
    fn splits_on_gaps_longer_than_one_second() {
        let s = [seg(0, "", vec![w(0, 400, " Hello"), w(450, 900, " there."), w(3_000, 3_300, " Anyway")])];
        assert_eq!(texts(&track_lines(&s, Track::Me)), vec![(0, "Hello there."), (3_000, "Anyway")]);
    }

    #[test]
    fn a_gap_of_exactly_one_second_does_not_split() {
        let s = [seg(0, "", vec![w(0, 400, " a"), w(1_400, 1_500, " b")])];
        assert_eq!(texts(&track_lines(&s, Track::Me)), vec![(0, "a b")]);
    }

    #[test]
    fn lone_first_word_moves_to_the_next_chunk() {
        let s = [seg(0, "", vec![w(0, 200, " So"), w(5_000, 5_300, " the"), w(5_300, 5_600, " plan")])];
        assert_eq!(texts(&track_lines(&s, Track::Me)), vec![(5_000, "So the plan")]);
    }

    #[test]
    fn lone_word_alone_in_segment_stays_put() {
        let s = [seg(0, "", vec![w(700, 900, " Yes.")])];
        assert_eq!(texts(&track_lines(&s, Track::Me)), vec![(700, "Yes.")]);
    }

    #[test]
    fn whitespace_only_text_is_dropped() {
        let s = [seg(0, "   ", vec![]), seg(10, "", vec![w(10, 20, " ")])];
        assert!(track_lines(&s, Track::Me).is_empty());
    }

    #[test]
    fn merge_sorts_by_time_and_puts_me_first_on_ties() {
        let me = [seg(1_000, "mine", vec![]), seg(5_000, "later", vec![])];
        let others = [seg(1_000, "theirs", vec![]), seg(2_000, "middle", vec![])];
        let lines = merge(&me, &others);
        let got: Vec<_> = lines.iter().map(|l| (l.start_ms, l.track, l.text.as_str())).collect();
        assert_eq!(
            got,
            vec![
                (1_000, Track::Me, "mine"),
                (1_000, Track::Others, "theirs"),
                (2_000, Track::Others, "middle"),
                (5_000, Track::Me, "later"),
            ]
        );
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib merge::`
Expected: compile errors (`track_lines`, `merge`, `Line` not found).

- [ ] **Step 3: Implement**

Top of `src/merge.rs`:
```rust
//! Turns per-track Whisper segments into one time-ordered list of lines.

use crate::store::{Segment, Track, Word};

/// Words further apart than this start a new line.
pub const SPLIT_GAP_MS: u64 = 1_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    pub start_ms: u64,
    pub track: Track,
    pub text: String,
}

pub fn track_lines<'a>(segments: impl IntoIterator<Item = &'a Segment>, track: Track) -> Vec<Line> {
    let mut out = Vec::new();
    for seg in segments {
        let Some((first, rest)) = seg.words.split_first() else {
            push_line(&mut out, seg.start_ms, track, &seg.text);
            continue;
        };
        let mut chunks: Vec<Vec<Word>> = vec![vec![first.clone()]];
        for word in rest {
            let prev_end = chunks.last().and_then(|c| c.last()).map_or(0, |w| w.end_ms);
            if word.start_ms.saturating_sub(prev_end) > SPLIT_GAP_MS {
                chunks.push(Vec::new());
            }
            chunks.last_mut().expect("chunks is never empty").push(word.clone());
        }
        // Whisper often pins a segment's first word to the segment start, well before
        // the rest: a lone leading word goes with the words that follow it.
        if chunks.len() > 1 && chunks[0].len() == 1 {
            let lone = chunks.remove(0).remove(0);
            let start_ms = chunks[0][0].start_ms;
            chunks[0].insert(0, Word { start_ms, ..lone });
        }
        for chunk in chunks {
            let text: String = chunk.iter().map(|w| w.text.as_str()).collect();
            push_line(&mut out, chunk[0].start_ms, track, &text);
        }
    }
    out
}

/// Both tracks as one list, ordered by start time; on equal times "me" comes first.
pub fn merge<'a>(
    me: impl IntoIterator<Item = &'a Segment>,
    others: impl IntoIterator<Item = &'a Segment>,
) -> Vec<Line> {
    let mut lines = track_lines(me, Track::Me);
    lines.extend(track_lines(others, Track::Others));
    lines.sort_by_key(|l| l.start_ms);
    lines
}

fn push_line(out: &mut Vec<Line>, start_ms: u64, track: Track, text: &str) {
    let text = text.trim();
    if !text.is_empty() {
        out.push(Line { start_ms, track, text: text.to_string() });
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib merge::`
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: merge per-track Whisper output into one timeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Rendering `transcript.md` and `summary.md`

**Files:**
- Create: `src/render.rs`
- Modify: `src/lib.rs` (add `pub mod render;`)

**Interfaces:**
- Consumes: `config::Labels`, `merge::{merge, Line}`, `store::{Meta, Conversation, Track, TrackTranscript}`, `time::{add_ms, format_duration}`
- Produces: `render::header(&Meta) -> String`, `render::time_range(&Meta, &Conversation) -> String` (`"09:40 to 09:54"`), `render::label<'a>(&'a Labels, Track) -> &'a str`, `render::conversation_lines(&Conversation, &TrackTranscript, &TrackTranscript) -> Vec<Line>`, `render::render_lines(&Meta, &Labels, &[Line]) -> String`, `render::render_transcript(&Meta, &Labels, &TrackTranscript, &TrackTranscript) -> String`, `render::render_summary(&Meta, &[(&Conversation, String)]) -> String`, `render::word_count(&[Line]) -> usize`

- [ ] **Step 1: Write the failing tests**

`src/render.rs` (tests first):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Segment, Stage, Trigger, TriggerKind, Word};

    fn meta() -> Meta {
        let start: jiff::Zoned = "2026-10-08T09:03:12+02:00[+02:00]".parse().unwrap();
        let mut m = Meta::new(
            "2026-10-08T090312".into(),
            start.clone(),
            Trigger { kind: TriggerKind::App, app_name: Some("Firefox".into()), binary: Some("firefox".into()) },
            "en".into(),
        );
        m.title = Some("Coworking".into());
        m.ended_at = Some(crate::time::add_ms(&start, 31_020_000));
        let mut c1 = Conversation::new(1, 2_211_000, 3_034_500);
        c1.stage = Stage::Transcribed;
        let mut c2 = Conversation::new(2, 7_000_000, 7_100_000);
        c2.stage = Stage::Transcribed;
        let c3 = Conversation::new(3, 9_000_000, 9_100_000); // still pending
        m.conversations = vec![c1, c2, c3];
        m
    }

    fn one(conversation: u32, start_ms: u64, text: &str) -> TrackTranscript {
        TrackTranscript {
            segments: vec![Segment {
                conversation,
                start_ms,
                end_ms: start_ms + 500,
                text: text.into(),
                words: vec![Word { start_ms, end_ms: start_ms + 500, text: text.into() }],
            }],
        }
    }

    #[test]
    fn header_variants() {
        let mut m = meta();
        assert_eq!(header(&m), "# Coworking. 2026-10-08 09:03 (8 h 37 min, firefox)");
        m.title = None;
        m.ended_at = None;
        m.trigger.kind = TriggerKind::Import;
        assert_eq!(header(&m), "# 2026-10-08 09:03 (import)");
    }

    #[test]
    fn transcript_layout() {
        let m = meta();
        let me = one(1, 2_211_500, " Ciao! Hai visto la MR?");
        let others = one(1, 2_214_000, " Sì, I left a couple of comments.");
        let labels = Labels { me: "Alex".into(), others: "Others".into() };
        let got = render_transcript(&m, &labels, &me, &others);
        let want = "\
# Coworking. 2026-10-08 09:03 (8 h 37 min, firefox)

## 09:40 to 09:53

[09:40:03] Alex: Ciao! Hai visto la MR?
[09:40:06] Others: Sì, I left a couple of comments.

## 10:59 to 11:01

_No speech detected._
";
        assert_eq!(got, want);
    }

    #[test]
    fn summary_layout_and_word_count() {
        let m = meta();
        let c1 = &m.conversations[0];
        let got = render_summary(&m, &[(c1, "## Summary\n\nWe met.\n\n".into())]);
        assert_eq!(
            got,
            "# Coworking. 2026-10-08 09:03 (8 h 37 min, firefox)\n\n## 09:40 to 09:53\n\n## Summary\n\nWe met.\n"
        );
        let lines = conversation_lines(c1, &one(1, 2_211_500, " one two"), &one(1, 2_212_000, " three"));
        assert_eq!(word_count(&lines), 3);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib render::`
Expected: compile errors (`header`, `render_transcript`, ... not found).

- [ ] **Step 3: Implement**

Top of `src/render.rs`:
```rust
//! Markdown rendering of transcripts and summaries. Both files are derived and rewritten
//! whenever the underlying data changes.

use crate::config::Labels;
use crate::merge::{Line, merge};
use crate::store::{Conversation, Meta, Track, TrackTranscript};
use crate::time::{add_ms, format_duration};

/// `# Title. 2026-10-08 09:03 (42 min, firefox)`; title and duration are optional.
pub fn header(meta: &Meta) -> String {
    let mut s = String::from("# ");
    if let Some(title) = &meta.title {
        s.push_str(title);
        s.push_str(". ");
    }
    s.push_str(&meta.started_at.strftime("%Y-%m-%d %H:%M").to_string());
    let mut extras = Vec::new();
    if let Some(ms) = meta.duration_ms() {
        extras.push(format_duration(ms));
    }
    extras.push(meta.app_label());
    s.push_str(&format!(" ({})", extras.join(", ")));
    s
}

pub fn time_range(meta: &Meta, c: &Conversation) -> String {
    format!(
        "{} to {}",
        add_ms(&meta.started_at, c.start_ms).strftime("%H:%M"),
        add_ms(&meta.started_at, c.end_ms).strftime("%H:%M")
    )
}

pub fn label(labels: &Labels, track: Track) -> &str {
    match track {
        Track::Me => &labels.me,
        Track::Others => &labels.others,
    }
}

pub fn conversation_lines(c: &Conversation, me: &TrackTranscript, others: &TrackTranscript) -> Vec<Line> {
    merge(me.conversation(c.n), others.conversation(c.n))
}

/// `[HH:MM:SS] Label: text` per line.
pub fn render_lines(meta: &Meta, labels: &Labels, lines: &[Line]) -> String {
    let mut out = String::new();
    for l in lines {
        let at = add_ms(&meta.started_at, l.start_ms).strftime("%H:%M:%S");
        out.push_str(&format!("[{at}] {}: {}\n", label(labels, l.track), l.text));
    }
    out
}

/// The whole transcript: header, then one section per transcribed conversation.
pub fn render_transcript(meta: &Meta, labels: &Labels, me: &TrackTranscript, others: &TrackTranscript) -> String {
    let mut out = header(meta);
    out.push('\n');
    for c in meta.conversations.iter().filter(|c| !c.needs_transcription()) {
        out.push_str(&format!("\n## {}\n\n", time_range(meta, c)));
        let lines = conversation_lines(c, me, others);
        if lines.is_empty() {
            out.push_str("_No speech detected._\n");
        } else {
            out.push_str(&render_lines(meta, labels, &lines));
        }
    }
    out
}

/// Header, then each conversation's summary under its time range.
pub fn render_summary(meta: &Meta, summaries: &[(&Conversation, String)]) -> String {
    let mut out = header(meta);
    out.push('\n');
    for (c, text) in summaries {
        out.push_str(&format!("\n## {}\n\n{}\n", time_range(meta, c), text.trim()));
    }
    out
}

pub fn word_count(lines: &[Line]) -> usize {
    lines.iter().map(|l| l.text.split_whitespace().count()).sum()
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib render::`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: render transcript.md and summary.md

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Audio I/O (WAV ranges, ffmpeg, ffprobe)

**Files:**
- Create: `src/audio.rs`
- Modify: `src/lib.rs` (add `pub mod audio;`)

**Interfaces:**
- Produces: `audio::SAMPLE_RATE: u32 = 16_000`, `audio::ms_to_samples(u64) -> usize`, `audio::samples_to_ms(usize) -> u64`, `audio::wav_duration_ms(&Path) -> Result<u64>`, `audio::read_wav_range(&Path, start_ms: u64, end_ms: u64) -> Result<Vec<f32>>`, `audio::normalize_to_wav(src: &Path, dst: &Path) -> Result<()>`, `audio::encode_opus(wav: &Path, dst: &Path) -> Result<()>`, `audio::probe_duration_ms(&Path) -> Result<u64>`, and a test helper `audio::write_test_wav(&Path, seconds: f32, value: i16)` behind `#[cfg(test)]`

- [ ] **Step 1: Write the failing tests**

`src/audio.rs` (tests first):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(path: &Path, n: usize) {
        let spec = hound::WavSpec { channels: 1, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..n {
            w.write_sample((i % 1000) as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    #[test]
    fn conversions() {
        assert_eq!(ms_to_samples(1_000), 16_000);
        assert_eq!(samples_to_ms(8_000), 500);
    }

    #[test]
    fn read_range_returns_exact_samples() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.wav");
        ramp(&p, 32_000); // 2 s
        assert_eq!(wav_duration_ms(&p).unwrap(), 2_000);
        let s = read_wav_range(&p, 510, 610).unwrap();
        assert_eq!(s.len(), 1_600);
        assert_eq!(s[0], (8_160 % 1000) as f32 / 32768.0);
    }

    #[test]
    fn read_range_clamps_past_end() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.wav");
        ramp(&p, 16_000); // 1 s
        assert_eq!(read_wav_range(&p, 500, 5_000).unwrap().len(), 8_000);
        assert!(read_wav_range(&p, 3_000, 5_000).unwrap().is_empty());
        assert!(read_wav_range(&p, 800, 200).unwrap().is_empty());
    }

    #[test]
    fn rejects_wrong_format() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("stereo.wav");
        let spec = hound::WavSpec { channels: 2, sample_rate: 44_100, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        w.write_sample(0i16).unwrap();
        w.write_sample(0i16).unwrap();
        w.finalize().unwrap();
        let err = read_wav_range(&p, 0, 10).unwrap_err();
        assert!(format!("{err:#}").contains("16 kHz mono"), "{err:#}");
    }

    #[test]
    fn ffmpeg_normalize_and_opus_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src.wav");
        let spec = hound::WavSpec { channels: 2, sample_rate: 44_100, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(&src, spec).unwrap();
        for i in 0..(44_100 * 2 * 3) {
            w.write_sample(((i % 200) as i16 - 100) * 50).unwrap();
        }
        w.finalize().unwrap();

        let wav = tmp.path().join("norm.wav");
        normalize_to_wav(&src, &wav).unwrap();
        assert!((wav_duration_ms(&wav).unwrap() as i64 - 3_000).abs() < 50);

        let opus = tmp.path().join("out.opus.tmp");
        encode_opus(&wav, &opus).unwrap();
        assert!((probe_duration_ms(&opus).unwrap() as i64 - 3_000).abs() < 500);
        assert!(std::fs::metadata(&opus).unwrap().len() < std::fs::metadata(&wav).unwrap().len());
    }

    #[test]
    fn ffmpeg_errors_carry_stderr() {
        let tmp = tempfile::tempdir().unwrap();
        let err = normalize_to_wav(&tmp.path().join("missing.wav"), &tmp.path().join("x.wav")).unwrap_err();
        assert!(format!("{err:#}").contains("ffmpeg failed"), "{err:#}");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib audio::`
Expected: compile errors.

- [ ] **Step 3: Implement**

Top of `src/audio.rs`:
```rust
//! Audio files: kute works on 16 kHz mono 16-bit WAV and archives as Ogg Opus.
//! ffmpeg/ffprobe do the format conversions.

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

pub const SAMPLE_RATE: u32 = 16_000;

pub fn ms_to_samples(ms: u64) -> usize {
    (ms * SAMPLE_RATE as u64 / 1_000) as usize
}

pub fn samples_to_ms(n: usize) -> u64 {
    n as u64 * 1_000 / SAMPLE_RATE as u64
}

pub fn wav_duration_ms(path: &Path) -> Result<u64> {
    Ok(samples_to_ms(open_checked(path)?.duration() as usize))
}

/// Samples in `[start_ms, end_ms)` as f32 in [-1, 1]. Ranges past the end of the file are
/// clamped, so a short or empty result is normal.
pub fn read_wav_range(path: &Path, start_ms: u64, end_ms: u64) -> Result<Vec<f32>> {
    let mut reader = open_checked(path)?;
    let total = reader.duration() as usize;
    let start = ms_to_samples(start_ms).min(total);
    let end = ms_to_samples(end_ms).min(total).max(start);
    reader.seek(start as u32).with_context(|| format!("seeking in {}", path.display()))?;
    reader
        .samples::<i16>()
        .take(end - start)
        .map(|s| s.map(|v| v as f32 / 32768.0).map_err(anyhow::Error::from))
        .collect::<Result<Vec<f32>>>()
        .with_context(|| format!("reading {}", path.display()))
}

fn open_checked(path: &Path) -> Result<hound::WavReader<BufReader<File>>> {
    let reader = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        bail!("{} is not 16 kHz mono 16-bit PCM", path.display());
    }
    Ok(reader)
}

/// Converts any audio ffmpeg can read into kute's working WAV format.
pub fn normalize_to_wav(src: &Path, dst: &Path) -> Result<()> {
    let mut cmd = ffmpeg();
    cmd.arg("-i").arg(src).args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav"]).arg(dst);
    run(cmd, "ffmpeg")
}

/// Encodes a WAV as Ogg Opus at speech quality. `dst` may have any extension.
pub fn encode_opus(wav: &Path, dst: &Path) -> Result<()> {
    let mut cmd = ffmpeg();
    cmd.arg("-i")
        .arg(wav)
        .args(["-c:a", "libopus", "-b:a", "24k", "-application", "voip", "-f", "opus"])
        .arg(dst);
    run(cmd, "ffmpeg")
}

pub fn probe_duration_ms(path: &Path) -> Result<u64> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
        .arg(path)
        .output()
        .context("running ffprobe (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!("ffprobe failed on {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let secs: f64 = text.trim().parse().with_context(|| format!("ffprobe printed {text:?}"))?;
    Ok((secs * 1_000.0).round() as u64)
}

fn ffmpeg() -> Command {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y"]);
    cmd
}

fn run(mut cmd: Command, name: &str) -> Result<()> {
    let out = cmd.output().with_context(|| format!("running {name} (is ffmpeg installed?)"))?;
    if !out.status.success() {
        bail!("{name} failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}

/// Test helper: writes `seconds` of a 200 Hz square wave of amplitude `value` (first
/// sample is `+value`) as a working-format WAV. A square wave, not a constant, so it
/// survives Opus' high-pass filter in round trips.
#[cfg(test)]
pub fn write_test_wav(path: &Path, seconds: f32, value: i16) {
    let spec = hound::WavSpec { channels: 1, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..(seconds * SAMPLE_RATE as f32) as usize {
        w.write_sample(if (i / 40) % 2 == 0 { value } else { -value }).unwrap();
    }
    w.finalize().unwrap();
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib audio::`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: add WAV range reading and ffmpeg helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Speech chunking and per-track transcription

**Files:**
- Create: `src/transcribe/mod.rs`
- Modify: `src/lib.rs` (add `pub mod transcribe;`)

**Interfaces:**
- Consumes: `audio::{ms_to_samples, samples_to_ms}`, `store::{Segment, Word}`
- Produces:
  - `transcribe::Span { start_ms: u64, end_ms: u64 }` with `len_ms()`
  - `transcribe::RawSegment { start_ms: u64, end_ms: u64, text: String, words: Vec<Word> }` (times relative to the audio passed in)
  - `trait SpeechDetector { fn speech_regions(&mut self, samples: &[f32]) -> Result<Vec<Span>>; }`
  - `trait Transcriber { fn model_name(&self) -> &str; fn detect_language(&mut self, samples: &[f32]) -> Result<String>; fn transcribe(&mut self, samples: &[f32], language: &str) -> Result<Vec<RawSegment>>; }`
  - constants `CHUNK_JOIN_GAP_MS = 2_000`, `CHUNK_MAX_MS = 120_000`, `CHUNK_PAD_MS = 200`, `MIN_DETECT_MS = 3_000`, `DETECT_WINDOW_MS = 30_000`
  - `transcribe::plan_chunks(regions: &[Span], total_ms: u64) -> Vec<Span>`
  - `transcribe::TrackResult { language: Option<String>, segments: Vec<Segment> }`
  - `transcribe::transcribe_track(samples: &[f32], base_ms: u64, conversation: u32, forced_language: Option<&str>, vad: &mut dyn SpeechDetector, asr: &mut dyn Transcriber) -> Result<TrackResult>`

How it works: VAD finds speech regions; regions are padded by 200 ms, joined when the gap is at most 2 s, and capped at 120 s per chunk (chunks never overlap). Each chunk of at least 3 s gets its own language detection; shorter chunks take the language of the nearest long chunk (the earlier one on a tie); if no chunk is long enough, one detection runs on the first 30 s of all speech joined together. A forced language skips detection. The track's language is the one covering the most chunk time.

- [ ] **Step 1: Write the failing tests**

`src/transcribe/mod.rs` (tests first):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FakeVad(Vec<Span>);
    impl SpeechDetector for FakeVad {
        fn speech_regions(&mut self, _samples: &[f32]) -> Result<Vec<Span>> {
            Ok(self.0.clone())
        }
    }

    #[derive(Default)]
    struct FakeAsr {
        detect: VecDeque<String>,
        detect_calls: Vec<usize>,
        calls: Vec<(usize, String)>,
    }
    impl Transcriber for FakeAsr {
        fn model_name(&self) -> &str {
            "fake"
        }
        fn detect_language(&mut self, samples: &[f32]) -> Result<String> {
            self.detect_calls.push(samples.len());
            Ok(self.detect.pop_front().unwrap_or_else(|| "en".into()))
        }
        fn transcribe(&mut self, samples: &[f32], language: &str) -> Result<Vec<RawSegment>> {
            self.calls.push((samples.len(), language.to_string()));
            Ok(vec![RawSegment {
                start_ms: 100,
                end_ms: 600,
                text: " hi".into(),
                words: vec![Word { start_ms: 100, end_ms: 600, text: " hi".into() }],
            }])
        }
    }

    fn span(a: u64, b: u64) -> Span {
        Span { start_ms: a, end_ms: b }
    }

    fn silence(ms: u64) -> Vec<f32> {
        vec![0.0; crate::audio::ms_to_samples(ms)]
    }

    #[test]
    fn plan_pads_and_clamps() {
        assert_eq!(plan_chunks(&[span(100, 1_000)], 1_100), vec![span(0, 1_100)]);
    }

    #[test]
    fn plan_joins_small_gaps_and_keeps_big_ones() {
        assert_eq!(plan_chunks(&[span(0, 1_000), span(3_000, 4_000)], 10_000), vec![span(0, 4_200)]);
        assert_eq!(
            plan_chunks(&[span(0, 1_000), span(5_000, 6_000)], 10_000),
            vec![span(0, 1_200), span(4_800, 6_200)]
        );
    }

    #[test]
    fn plan_caps_chunk_length_without_overlap() {
        let regions: Vec<Span> = (0..200).map(|i| span(i * 1_000, i * 1_000 + 500)).collect();
        let chunks = plan_chunks(&regions, 300_000);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.len_ms() <= CHUNK_MAX_MS, "{c:?}");
        }
        for pair in chunks.windows(2) {
            assert!(pair[1].start_ms >= pair[0].end_ms, "{pair:?}");
        }
    }

    #[test]
    fn no_speech_means_no_calls() {
        let mut asr = FakeAsr::default();
        let r = transcribe_track(&silence(10_000), 0, 1, None, &mut FakeVad(vec![]), &mut asr).unwrap();
        assert_eq!(r.language, None);
        assert!(r.segments.is_empty());
        assert!(asr.calls.is_empty() && asr.detect_calls.is_empty());
    }

    #[test]
    fn times_are_shifted_into_session_time() {
        let mut asr = FakeAsr::default();
        let r = transcribe_track(&silence(20_000), 60_000, 3, None, &mut FakeVad(vec![span(1_000, 5_000)]), &mut asr)
            .unwrap();
        // chunk starts at 1_000 - 200 pad; segment at +100 inside it
        assert_eq!(r.segments.len(), 1);
        assert_eq!(r.segments[0].conversation, 3);
        assert_eq!(r.segments[0].start_ms, 60_000 + 800 + 100);
        assert_eq!(r.segments[0].words[0].end_ms, 60_000 + 800 + 600);
        assert_eq!(r.language.as_deref(), Some("en"));
    }

    #[test]
    fn forced_language_skips_detection() {
        let mut asr = FakeAsr::default();
        transcribe_track(&silence(10_000), 0, 1, Some("it"), &mut FakeVad(vec![span(0, 4_000)]), &mut asr).unwrap();
        assert!(asr.detect_calls.is_empty());
        assert_eq!(asr.calls[0].1, "it");
    }

    #[test]
    fn short_chunk_takes_nearest_long_chunk_language() {
        let mut asr = FakeAsr { detect: VecDeque::from(vec!["it".to_string()]), ..Default::default() };
        let vad = &mut FakeVad(vec![span(0, 5_000), span(10_000, 11_000)]);
        transcribe_track(&silence(20_000), 0, 1, None, vad, &mut asr).unwrap();
        assert_eq!(asr.detect_calls.len(), 1);
        assert_eq!(asr.calls.iter().map(|c| c.1.as_str()).collect::<Vec<_>>(), vec!["it", "it"]);
    }

    #[test]
    fn all_short_chunks_share_one_detection() {
        let mut asr = FakeAsr { detect: VecDeque::from(vec!["it".to_string()]), ..Default::default() };
        let vad = &mut FakeVad(vec![span(0, 1_000), span(5_000, 6_000)]);
        let r = transcribe_track(&silence(10_000), 0, 1, None, vad, &mut asr).unwrap();
        assert_eq!(asr.detect_calls.len(), 1);
        // [0, 1_200] (padding clamped at 0) + [4_800, 6_200]
        assert_eq!(asr.detect_calls[0], crate::audio::ms_to_samples(1_200 + 1_400));
        assert_eq!(r.language.as_deref(), Some("it"));
    }

    #[test]
    fn track_language_is_the_dominant_one() {
        let mut asr = FakeAsr { detect: VecDeque::from(vec!["en".to_string(), "it".to_string()]), ..Default::default() };
        let vad = &mut FakeVad(vec![span(0, 6_000), span(20_000, 24_000)]);
        let r = transcribe_track(&silence(30_000), 0, 1, None, vad, &mut asr).unwrap();
        assert_eq!(r.language.as_deref(), Some("en"));
        assert_eq!(asr.calls.iter().map(|c| c.1.as_str()).collect::<Vec<_>>(), vec!["en", "it"]);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib transcribe::`
Expected: compile errors.

- [ ] **Step 3: Implement**

Top of `src/transcribe/mod.rs`:
```rust
//! Speech detection, chunking, and per-track transcription. The engines sit behind
//! traits so the orchestration is testable and other backends can be added.

use crate::audio::{ms_to_samples, samples_to_ms};
use crate::store::{Segment, Word};
use anyhow::Result;
use std::collections::BTreeMap;

pub const CHUNK_JOIN_GAP_MS: u64 = 2_000;
pub const CHUNK_MAX_MS: u64 = 120_000;
pub const CHUNK_PAD_MS: u64 = 200;
pub const MIN_DETECT_MS: u64 = 3_000;
pub const DETECT_WINDOW_MS: u64 = 30_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start_ms: u64,
    pub end_ms: u64,
}

impl Span {
    pub fn len_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A Whisper segment with times relative to the audio that was passed in.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub words: Vec<Word>,
}

pub trait SpeechDetector {
    fn speech_regions(&mut self, samples: &[f32]) -> Result<Vec<Span>>;
}

pub trait Transcriber {
    fn model_name(&self) -> &str;
    fn detect_language(&mut self, samples: &[f32]) -> Result<String>;
    fn transcribe(&mut self, samples: &[f32], language: &str) -> Result<Vec<RawSegment>>;
}

/// Pads speech regions, joins close ones, and caps chunk length. Chunks never overlap.
pub fn plan_chunks(regions: &[Span], total_ms: u64) -> Vec<Span> {
    let mut sorted = regions.to_vec();
    sorted.sort_by_key(|r| r.start_ms);
    let mut chunks: Vec<Span> = Vec::new();
    for r in sorted {
        let padded = Span {
            start_ms: r.start_ms.saturating_sub(CHUNK_PAD_MS),
            end_ms: (r.end_ms + CHUNK_PAD_MS).min(total_ms),
        };
        if padded.end_ms <= padded.start_ms {
            continue;
        }
        match chunks.last_mut() {
            Some(c)
                if padded.start_ms <= c.end_ms + CHUNK_JOIN_GAP_MS
                    && padded.end_ms.saturating_sub(c.start_ms) <= CHUNK_MAX_MS =>
            {
                c.end_ms = c.end_ms.max(padded.end_ms);
            }
            Some(c) => {
                let start_ms = padded.start_ms.max(c.end_ms);
                if padded.end_ms > start_ms {
                    chunks.push(Span { start_ms, end_ms: padded.end_ms });
                }
            }
            None => chunks.push(padded),
        }
    }
    chunks
}

pub struct TrackResult {
    pub language: Option<String>,
    pub segments: Vec<Segment>,
}

/// Transcribes one track's audio for one conversation. `base_ms` is the session time of
/// `samples[0]`; returned segments are in session time.
pub fn transcribe_track(
    samples: &[f32],
    base_ms: u64,
    conversation: u32,
    forced_language: Option<&str>,
    vad: &mut dyn SpeechDetector,
    asr: &mut dyn Transcriber,
) -> Result<TrackResult> {
    let total_ms = samples_to_ms(samples.len());
    let chunks = plan_chunks(&vad.speech_regions(samples)?, total_ms);
    let languages = assign_languages(&chunks, samples, forced_language, asr)?;

    let mut segments = Vec::new();
    let mut coverage: BTreeMap<String, u64> = BTreeMap::new();
    for (chunk, language) in chunks.iter().zip(&languages) {
        *coverage.entry(language.clone()).or_default() += chunk.len_ms();
        let offset = base_ms + chunk.start_ms;
        for raw in asr.transcribe(slice(samples, *chunk), language)? {
            segments.push(Segment {
                conversation,
                start_ms: offset + raw.start_ms,
                end_ms: offset + raw.end_ms,
                text: raw.text,
                words: raw
                    .words
                    .into_iter()
                    .map(|w| Word { start_ms: offset + w.start_ms, end_ms: offset + w.end_ms, text: w.text })
                    .collect(),
            });
        }
    }

    let mut language: Option<(String, u64)> = None;
    for (lang, ms) in coverage {
        if language.as_ref().is_none_or(|(_, best)| ms > *best) {
            language = Some((lang, ms));
        }
    }
    Ok(TrackResult { language: language.map(|(l, _)| l), segments })
}

fn assign_languages(
    chunks: &[Span],
    samples: &[f32],
    forced: Option<&str>,
    asr: &mut dyn Transcriber,
) -> Result<Vec<String>> {
    if let Some(lang) = forced {
        return Ok(vec![lang.to_string(); chunks.len()]);
    }
    let mut detected: Vec<Option<String>> = vec![None; chunks.len()];
    for (i, c) in chunks.iter().enumerate() {
        if c.len_ms() >= MIN_DETECT_MS {
            detected[i] = Some(asr.detect_language(slice(samples, *c))?);
        }
    }
    let known: Vec<(usize, String)> =
        detected.iter().enumerate().filter_map(|(i, l)| l.clone().map(|l| (i, l))).collect();
    if known.is_empty() {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let mut joined = Vec::new();
        for c in chunks {
            joined.extend_from_slice(slice(samples, *c));
            if joined.len() >= ms_to_samples(DETECT_WINDOW_MS) {
                break;
            }
        }
        let lang = asr.detect_language(&joined)?;
        return Ok(vec![lang; chunks.len()]);
    }
    Ok((0..chunks.len())
        .map(|i| {
            detected[i].clone().unwrap_or_else(|| {
                let nearest = known.iter().min_by_key(|(j, _)| (i.abs_diff(*j), *j > i));
                nearest.expect("known is not empty").1.clone()
            })
        })
        .collect())
}

fn slice(samples: &[f32], span: Span) -> &[f32] {
    let start = ms_to_samples(span.start_ms).min(samples.len());
    let end = ms_to_samples(span.end_ms).min(samples.len()).max(start);
    &samples[start..end]
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib transcribe::`
Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: chunk speech and transcribe tracks per chunk

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: whisper.cpp and Silero backends, speech fixtures

**Files:**
- Create: `src/transcribe/whisper.rs`, `tests/fixtures/make-fixtures.sh`, `tests/fixtures/me.wav`, `tests/fixtures/others.wav`
- Modify: `src/transcribe/mod.rs` (add `pub mod whisper;` at the top)

**Interfaces:**
- Consumes: `transcribe::{SpeechDetector, Transcriber, RawSegment, Span}`, `store::Word`
- Produces: `whisper::SileroVad::load(&Path) -> Result<SileroVad>` (implements `SpeechDetector`), `whisper::WhisperTranscriber::load(path: &Path, name: &str, use_gpu: bool) -> Result<WhisperTranscriber>` (implements `Transcriber`), `whisper::TokenPiece { bytes: Vec<u8>, t0_cs: i64, t1_cs: i64, special: bool }`, `whisper::tokens_to_words(&[TokenPiece]) -> Vec<Word>`

whisper-rs 0.16 facts this code relies on (checked against the crate source): `WhisperVadContext::segments_from_samples(WhisperVadParams, &[f32])` returns an iterator of `WhisperVadSegment { start, end }` in **centiseconds** (f32); segment `start_timestamp()` / token `token_data().t0` / `.t1` are **centiseconds** (i64); tokens with `id >= ctx.token_eot()` are special; `WhisperState::lang_detect(offset_ms, threads)` needs `pcm_to_mel` first; `whisper_rs::get_lang_str(id)`. Token bytes can split a UTF-8 character, so words are assembled from raw bytes and decoded once at the end.

- [ ] **Step 1: Generate the speech fixtures**

`tests/fixtures/make-fixtures.sh`:
```bash
#!/usr/bin/env bash
# Regenerates the synthetic speech fixtures (no real voices). Needs espeak-ng and ffmpeg.
# me.wav: English, then silence. others.wav: silence, then Italian.
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ff() { ffmpeg -nostdin -hide_banner -loglevel error -y "$@"; }
espeak-ng -v en -s 150 -w "$tmp/en.wav" "Hello everyone. Today we are going to talk about the differential download feature."
espeak-ng -v it -s 150 -w "$tmp/it.wav" "Ciao a tutti. Oggi parliamo della nuova versione del programma."
ff -i "$tmp/en.wav" -ar 16000 -ac 1 "$tmp/en16.wav"
ff -i "$tmp/it.wav" -ar 16000 -ac 1 "$tmp/it16.wav"
ff -f lavfi -t 6 -i anullsrc=r=16000:cl=mono "$tmp/sil6.wav"
ff -f lavfi -t 4 -i anullsrc=r=16000:cl=mono "$tmp/sil4.wav"
ff -i "$tmp/en16.wav" -i "$tmp/sil4.wav" -filter_complex '[0][1]concat=n=2:v=0:a=1' -ar 16000 -ac 1 -c:a pcm_s16le me.wav
ff -i "$tmp/sil6.wav" -i "$tmp/it16.wav" -filter_complex '[0][1]concat=n=2:v=0:a=1' -ar 16000 -ac 1 -c:a pcm_s16le others.wav
```
Run:
```bash
chmod +x tests/fixtures/make-fixtures.sh && tests/fixtures/make-fixtures.sh
ffprobe -v error -show_entries stream=sample_rate,channels -of csv=p=0 tests/fixtures/me.wav   # expect 16000,1
```

- [ ] **Step 2: Write the failing tests**

`src/transcribe/whisper.rs` (tests first):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn piece(bytes: &[u8], t0: i64, t1: i64) -> TokenPiece {
        TokenPiece { bytes: bytes.to_vec(), t0_cs: t0, t1_cs: t1, special: false }
    }

    #[test]
    fn tokens_to_words_groups_on_leading_space() {
        let words = tokens_to_words(&[
            TokenPiece { bytes: b"[_BEG_]".to_vec(), t0_cs: 0, t1_cs: 0, special: true },
            piece(b" Hello", 5, 46),
            piece(b" every", 46, 70),
            piece(b"one", 70, 97),
            piece(b"!", 132, 132),
            TokenPiece { bytes: b"[_TT_500]".to_vec(), t0_cs: 1000, t1_cs: 1000, special: true },
        ]);
        let got: Vec<_> = words.iter().map(|w| (w.start_ms, w.end_ms, w.text.as_str())).collect();
        assert_eq!(got, vec![(50, 460, " Hello"), (460, 1_320, " everyone!")]);
    }

    #[test]
    fn tokens_to_words_rejoins_split_utf8() {
        // "perché" with the é (C3 A9) split across two tokens, then "città".
        let words = tokens_to_words(&[
            piece(b" perch", 0, 10),
            piece(&[0xC3], 10, 11),
            piece(&[0xA9], 11, 12),
            piece(b"?", 12, 12),
            piece(b" citt", 20, 30),
            piece(&[0xC3, 0xA0], 30, 31),
        ]);
        let texts: Vec<_> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec![" perché?", " città"]);
        assert!(words.iter().all(|w| !w.text.contains('\u{FFFD}')));
    }

    #[test]
    fn tokens_to_words_skips_bracket_markers_and_negative_times() {
        let words = tokens_to_words(&[piece(b"[_SOT_]", -1, -1), piece(b" ok", -3, 4)]);
        assert_eq!(words, vec![Word { start_ms: 0, end_ms: 40, text: " ok".into() }]);
    }

    /// Needs KUTE_TEST_MODELS with ggml-tiny.bin and ggml-silero-v6.2.0.bin.
    #[test]
    #[ignore]
    fn real_models_transcribe_the_fixture() {
        let dir = std::path::PathBuf::from(std::env::var("KUTE_TEST_MODELS").expect("set KUTE_TEST_MODELS"));
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/me.wav");
        let samples = crate::audio::read_wav_range(&fixture, 0, 60_000).unwrap();
        let mut vad = SileroVad::load(&dir.join("ggml-silero-v6.2.0.bin")).unwrap();
        let regions = vad.speech_regions(&samples).unwrap();
        assert!(!regions.is_empty());
        assert!(regions[0].start_ms < 1_000, "{regions:?}");
        let mut asr = WhisperTranscriber::load(&dir.join("ggml-tiny.bin"), "tiny", false).unwrap();
        assert_eq!(asr.detect_language(&samples).unwrap(), "en");
        let text: String = asr.transcribe(&samples, "en").unwrap().iter().map(|s| s.text.clone()).collect();
        assert!(text.to_lowercase().contains("differential"), "{text}");
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --lib transcribe::whisper`
Expected: compile errors.

- [ ] **Step 4: Implement**

Top of `src/transcribe/whisper.rs`:
```rust
//! whisper.cpp (via whisper-rs) as the `Transcriber`, and its Silero VAD as the
//! `SpeechDetector`. whisper.cpp's own VAD option is not used: it remaps segment times but
//! not token times, which would break word timestamps.

use super::{RawSegment, Span, SpeechDetector, Transcriber};
use crate::store::Word;
use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadContext,
    WhisperVadContextParams, WhisperVadParams,
};

pub struct SileroVad {
    ctx: WhisperVadContext,
}

impl SileroVad {
    pub fn load(path: &Path) -> Result<SileroVad> {
        let mut params = WhisperVadContextParams::default();
        params.set_use_gpu(false);
        params.set_n_threads(2);
        let path = path.to_str().context("VAD model path is not valid UTF-8")?;
        let ctx = WhisperVadContext::new(path, params).map_err(|e| anyhow!("loading VAD model {path}: {e}"))?;
        Ok(SileroVad { ctx })
    }
}

impl SpeechDetector for SileroVad {
    fn speech_regions(&mut self, samples: &[f32]) -> Result<Vec<Span>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let segments = self
            .ctx
            .segments_from_samples(WhisperVadParams::new(), samples)
            .map_err(|e| anyhow!("running VAD: {e}"))?;
        // VAD timestamps are in centiseconds.
        Ok(segments
            .map(|s| Span { start_ms: (s.start.max(0.0) * 10.0) as u64, end_ms: (s.end.max(0.0) * 10.0) as u64 })
            .collect())
    }
}

pub struct WhisperTranscriber {
    ctx: WhisperContext,
    name: String,
    threads: usize,
}

impl WhisperTranscriber {
    pub fn load(path: &Path, name: &str, use_gpu: bool) -> Result<WhisperTranscriber> {
        let mut params = WhisperContextParameters::default();
        params.use_gpu(use_gpu);
        let path = path.to_str().context("model path is not valid UTF-8")?;
        let ctx = WhisperContext::new_with_params(path, params)
            .map_err(|e| anyhow!("loading Whisper model {path}: {e}"))?;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
        Ok(WhisperTranscriber { ctx, name: name.to_string(), threads })
    }
}

impl Transcriber for WhisperTranscriber {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn detect_language(&mut self, samples: &[f32]) -> Result<String> {
        let mut state = self.ctx.create_state().map_err(|e| anyhow!("creating Whisper state: {e}"))?;
        state.pcm_to_mel(samples, self.threads).map_err(|e| anyhow!("computing mel spectrogram: {e}"))?;
        let (id, _) = state.lang_detect(0, self.threads).map_err(|e| anyhow!("detecting language: {e}"))?;
        match whisper_rs::get_lang_str(id) {
            Some(lang) => Ok(lang.to_string()),
            None => bail!("Whisper returned unknown language id {id}"),
        }
    }

    fn transcribe(&mut self, samples: &[f32], language: &str) -> Result<Vec<RawSegment>> {
        let mut state = self.ctx.create_state().map_err(|e| anyhow!("creating Whisper state: {e}"))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(language));
        params.set_n_threads(self.threads as i32);
        params.set_token_timestamps(true);
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        state.full(params, samples).map_err(|e| anyhow!("Whisper failed: {e}"))?;

        let eot = self.ctx.token_eot();
        let mut out = Vec::new();
        for segment in state.as_iter() {
            let mut pieces = Vec::new();
            for i in 0..segment.n_tokens() {
                let Some(token) = segment.get_token(i) else { continue };
                let data = token.token_data();
                pieces.push(TokenPiece {
                    bytes: token.to_bytes().map(<[u8]>::to_vec).unwrap_or_default(),
                    t0_cs: data.t0,
                    t1_cs: data.t1,
                    special: data.id >= eot,
                });
            }
            out.push(RawSegment {
                start_ms: (segment.start_timestamp().max(0) as u64) * 10,
                end_ms: (segment.end_timestamp().max(0) as u64) * 10,
                text: segment.to_str_lossy().map(|s| s.into_owned()).unwrap_or_default(),
                words: tokens_to_words(&pieces),
            });
        }
        Ok(out)
    }
}

/// One Whisper token: raw bytes (possibly part of a UTF-8 character) and its times in
/// centiseconds.
pub struct TokenPiece {
    pub bytes: Vec<u8>,
    pub t0_cs: i64,
    pub t1_cs: i64,
    pub special: bool,
}

/// Groups tokens into words: a token starting with a space starts a new word, any other
/// token continues the previous one. Bytes are joined before decoding, so characters
/// split across tokens survive.
pub fn tokens_to_words(tokens: &[TokenPiece]) -> Vec<Word> {
    let mut words: Vec<(u64, u64, Vec<u8>)> = Vec::new();
    for t in tokens {
        if t.special || t.bytes.starts_with(b"[_") {
            continue;
        }
        let start = t.t0_cs.max(0) as u64 * 10;
        let end = t.t1_cs.max(0) as u64 * 10;
        match words.last_mut() {
            Some(word) if !t.bytes.starts_with(b" ") => {
                word.1 = word.1.max(end);
                word.2.extend_from_slice(&t.bytes);
            }
            _ => words.push((start, end.max(start), t.bytes.clone())),
        }
    }
    words
        .into_iter()
        .map(|(start_ms, end_ms, bytes)| Word { start_ms, end_ms, text: String::from_utf8_lossy(&bytes).into_owned() })
        .collect()
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib transcribe::whisper`
Expected: 3 PASS, 1 ignored.

Then, with models available (they're downloaded in Task 8; to run this now, fetch them by hand):
```bash
mkdir -p /tmp/kute-models && cd /tmp/kute-models
curl -sSfLO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
curl -sSfLO https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
cd - && KUTE_TEST_MODELS=/tmp/kute-models cargo test --lib transcribe::whisper -- --ignored
```
Expected: `real_models_transcribe_the_fixture` PASS.

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/ tests/fixtures/
git commit -m "feat: add whisper.cpp transcriber and Silero VAD

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Model registry and verified downloads

**Files:**
- Create: `src/models.rs`
- Modify: `src/lib.rs` (add `pub mod models;`)

**Interfaces:**
- Consumes: `config::expand_tilde`
- Produces: `models::ModelInfo { name, file, url, sha256: &'static str, size: u64 }`, `models::WHISPER_MODELS: &[ModelInfo]`, `models::VAD_MODEL: ModelInfo`, `models::find_whisper(&str) -> Option<&'static ModelInfo>`, `models::path_for(&Path, &ModelInfo) -> PathBuf`, `models::resolve_whisper(models_dir: &Path, model: &str) -> Result<(PathBuf, String)>`, `models::resolve_vad(&Path) -> Result<PathBuf>`, `models::download(models_dir: &Path, &ModelInfo, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf>`, `models::sha256_file(&Path) -> Result<String>`

- [ ] **Step 1: Write the failing tests**

`src/models.rs` (tests first):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    const HELLO_SHA256: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn registry_lookup() {
        assert_eq!(find_whisper("large-v3-turbo").unwrap().file, "ggml-large-v3-turbo.bin");
        assert!(find_whisper("huge").is_none());
        assert_eq!(VAD_MODEL.file, "ggml-silero-v6.2.0.bin");
    }

    #[test]
    fn resolve_whisper_by_name_path_and_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let err = resolve_whisper(tmp.path(), "tiny").unwrap_err();
        assert!(format!("{err:#}").contains("kute model download tiny"), "{err:#}");
        assert!(format!("{:#}", resolve_whisper(tmp.path(), "huge").unwrap_err()).contains("unknown"));

        std::fs::write(tmp.path().join("ggml-tiny.bin"), b"x").unwrap();
        let (path, name) = resolve_whisper(tmp.path(), "tiny").unwrap();
        assert_eq!((path, name.as_str()), (tmp.path().join("ggml-tiny.bin"), "tiny"));

        let custom = tmp.path().join("my-finetune.bin");
        std::fs::write(&custom, b"x").unwrap();
        let (path, name) = resolve_whisper(tmp.path(), custom.to_str().unwrap()).unwrap();
        assert_eq!((path, name.as_str()), (custom.clone(), "my-finetune"));
    }

    #[test]
    fn sha256_of_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("f");
        std::fs::write(&p, b"hello").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), HELLO_SHA256);
    }

    #[test]
    fn download_verifies_checksum_and_reports_progress() {
        let mut server = mockito::Server::new();
        server.mock("GET", "/good.bin").with_body("hello").create();
        server.mock("GET", "/bad.bin").with_body("tampered").create();
        let tmp = tempfile::tempdir().unwrap();

        let dest = tmp.path().join("good.bin");
        let mut seen = Vec::new();
        download_from(&format!("{}/good.bin", server.url()), &dest, HELLO_SHA256, 5, &mut |d, t| seen.push((d, t)))
            .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello");
        assert_eq!(seen.last(), Some(&(5, 5)));

        let bad = tmp.path().join("bad.bin");
        let err = download_from(&format!("{}/bad.bin", server.url()), &bad, HELLO_SHA256, 5, &mut |_, _| {})
            .unwrap_err();
        assert!(format!("{err:#}").contains("checksum"), "{err:#}");
        assert!(!bad.exists());
        assert!(!tmp.path().join("bad.bin.part").exists());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib models::`
Expected: compile errors.

- [ ] **Step 3: Implement**

Top of `src/models.rs`:
```rust
//! Known Whisper and VAD models, where they live, and checksum-verified downloads.

use crate::config::expand_tilde;
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct ModelInfo {
    pub name: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

pub const WHISPER_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "tiny",
        file: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        size: 77_691_713,
    },
    ModelInfo {
        name: "base",
        file: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        size: 147_951_465,
    },
    ModelInfo {
        name: "small",
        file: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
        size: 487_601_967,
    },
    ModelInfo {
        name: "medium",
        file: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
        size: 1_533_763_059,
    },
    ModelInfo {
        name: "large-v3",
        file: "ggml-large-v3.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
        size: 3_095_033_483,
    },
    ModelInfo {
        name: "large-v3-turbo",
        file: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        size: 1_624_555_275,
    },
    ModelInfo {
        name: "large-v3-turbo-q5_0",
        file: "ggml-large-v3-turbo-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        size: 574_041_195,
    },
];

pub const VAD_MODEL: ModelInfo = ModelInfo {
    name: "silero-v6.2.0",
    file: "ggml-silero-v6.2.0.bin",
    url: "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
    sha256: "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987",
    size: 885_098,
};

pub fn find_whisper(name: &str) -> Option<&'static ModelInfo> {
    WHISPER_MODELS.iter().find(|m| m.name == name)
}

pub fn path_for(models_dir: &Path, info: &ModelInfo) -> PathBuf {
    models_dir.join(info.file)
}

/// `model` is a known name or a path to a ggml `.bin` file. Returns the file and a name
/// to record in `meta.toml`.
pub fn resolve_whisper(models_dir: &Path, model: &str) -> Result<(PathBuf, String)> {
    let as_path = expand_tilde(Path::new(model));
    if as_path.extension().is_some_and(|e| e == "bin") && as_path.is_file() {
        let name = as_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        return Ok((as_path, name));
    }
    let Some(info) = find_whisper(model) else {
        let known: Vec<_> = WHISPER_MODELS.iter().map(|m| m.name).collect();
        bail!("unknown Whisper model {model:?}; use one of {} or a path to a ggml .bin file", known.join(", "));
    };
    let path = path_for(models_dir, info);
    if !path.is_file() {
        bail!("the Whisper model {model} is not downloaded yet; run `kute model download {model}`");
    }
    Ok((path, model.to_string()))
}

pub fn resolve_vad(models_dir: &Path) -> Result<PathBuf> {
    let path = path_for(models_dir, &VAD_MODEL);
    if !path.is_file() {
        bail!("the VAD model is not downloaded yet; run `kute model download`");
    }
    Ok(path)
}

/// Downloads `info` into `models_dir` unless an intact copy is already there.
pub fn download(models_dir: &Path, info: &ModelInfo, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    let dest = path_for(models_dir, info);
    if dest.is_file() && sha256_file(&dest)? == info.sha256 {
        progress(info.size, info.size);
        return Ok(dest);
    }
    download_from(info.url, &dest, info.sha256, info.size, progress)?;
    Ok(dest)
}

/// Streams `url` to `<dest>.part`, checks the SHA-256, then renames to `dest`.
pub(crate) fn download_from(
    url: &str,
    dest: &Path,
    sha256: &str,
    size: u64,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<()> {
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let response = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut done = 0u64;
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("downloading {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        done += n as u64;
        progress(done, size);
    }
    file.sync_all()?;
    drop(file);
    let got = hex(&hasher.finalize());
    if got != sha256 {
        let _ = fs::remove_file(&part);
        bail!("checksum mismatch for {url}: expected {sha256}, got {got}");
    }
    fs::rename(&part, dest).with_context(|| format!("moving download to {}", dest.display()))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib models::`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: add model registry with checksum-verified downloads

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Search index (SQLite FTS5)

**Files:**
- Create: `src/index.rs`
- Modify: `src/lib.rs` (add `pub mod index;`)

**Interfaces:**
- Consumes: `merge::Line`, `store::Meta`
- Produces: `index::Index::open(&Path) -> Result<Index>`, `Index::open_in_memory() -> Result<Index>`, `.upsert_session(&Meta)`, `.replace_lines(session_id: &str, conversation: u32, &[Line])`, `.replace_summary(session_id: &str, conversation: u32, Option<&str>)`, `.delete_session(&str)`, `.clear()`, `.search(&str, limit: usize) -> Result<Vec<Hit>>` (all writes return `Result<()>`); `index::Hit { kind: HitKind, session_id: String, title: Option<String>, started_at: String, conversation: u32, t_ms: Option<u64>, track: Option<String>, text: String }`; `index::HitKind { Line, Summary }`; `index::fts_query(&str) -> Option<String>`

User input never reaches FTS5 as syntax: every whitespace-separated term is wrapped in double quotes (inner quotes doubled), so `l'ho`, `C++`, `-x` and `AND` are plain words. The tokenizer is `unicode61 remove_diacritics 2`, so `perche` finds `perché`.

- [ ] **Step 1: Write the failing tests**

`src/index.rs` (tests first):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Track, Trigger, TriggerKind};

    fn meta(id: &str) -> Meta {
        let mut m = Meta::new(
            id.into(),
            "2026-10-08T09:00:00+02:00[+02:00]".parse().unwrap(),
            Trigger { kind: TriggerKind::Import, app_name: None, binary: None },
            "auto".into(),
        );
        m.title = Some("Sync".into());
        m
    }

    fn line(start_ms: u64, track: Track, text: &str) -> Line {
        Line { start_ms, track, text: text.into() }
    }

    fn seeded() -> Index {
        let idx = Index::open_in_memory().unwrap();
        idx.upsert_session(&meta("s1")).unwrap();
        idx.replace_lines(
            "s1",
            1,
            &[
                line(1_000, Track::Me, "Perché l'ho fatto in C++?"),
                line(2_000, Track::Others, "the differential download is ready"),
            ],
        )
        .unwrap();
        idx.replace_summary("s1", 1, Some("## Decisions\n\nShip the differential download.")).unwrap();
        idx
    }

    #[test]
    fn fts_query_quotes_every_term() {
        assert_eq!(fts_query("  "), None);
        assert_eq!(fts_query("\" -- ?"), None);
        assert_eq!(fts_query("a \"b\""), Some("\"a\" \"\"\"b\"\"\"".into()));
    }

    #[test]
    fn finds_lines_and_summaries() {
        let idx = seeded();
        let hits = idx.search("differential", 10).unwrap();
        assert_eq!(hits.len(), 2);
        let line_hit = hits.iter().find(|h| h.kind == HitKind::Line).unwrap();
        assert_eq!(line_hit.t_ms, Some(2_000));
        assert_eq!(line_hit.track.as_deref(), Some("others"));
        assert_eq!(line_hit.title.as_deref(), Some("Sync"));
        assert!(hits.iter().any(|h| h.kind == HitKind::Summary && h.text.contains("differential")));
    }

    #[test]
    fn search_handles_fts_syntax_literally() {
        let idx = seeded();
        for q in ["l'ho", "C++", "perche", "\"", "-x", "AND", "differential AND"] {
            let result = idx.search(q, 10);
            assert!(result.is_ok(), "{q:?}: {:?}", result.err());
        }
        assert_eq!(idx.search("perche", 10).unwrap().len(), 1);
        assert_eq!(idx.search("l'ho", 10).unwrap().len(), 1);
        assert_eq!(idx.search("C++", 10).unwrap().len(), 1);
        assert!(idx.search("AND", 10).unwrap().is_empty());
    }

    #[test]
    fn replace_and_delete() {
        let idx = seeded();
        idx.replace_lines("s1", 1, &[line(0, Track::Me, "something else")]).unwrap();
        assert!(idx.search("differential", 10).unwrap().iter().all(|h| h.kind == HitKind::Summary));
        idx.replace_summary("s1", 1, None).unwrap();
        assert!(idx.search("differential", 10).unwrap().is_empty());
        idx.delete_session("s1").unwrap();
        assert!(idx.search("something", 10).unwrap().is_empty());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib index::`
Expected: compile errors.

- [ ] **Step 3: Implement**

Top of `src/index.rs`:
```rust
//! Full-text search over transcripts and summaries. Derived from the session folders and
//! rebuildable at any time.

use crate::merge::Line;
use crate::store::Meta;
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::Path;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    app TEXT NOT NULL,
    conversation_count INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS lines USING fts5(
    text, session_id UNINDEXED, conversation UNINDEXED, t_ms UNINDEXED, track UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS summaries USING fts5(
    text, session_id UNINDEXED, conversation UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
";

pub struct Index {
    conn: Connection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitKind {
    Line,
    Summary,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub kind: HitKind,
    pub session_id: String,
    pub title: Option<String>,
    /// RFC 9557, as stored in `meta.toml`.
    pub started_at: String,
    pub conversation: u32,
    pub t_ms: Option<u64>,
    pub track: Option<String>,
    pub text: String,
}

impl Index {
    pub fn open(path: &Path) -> Result<Index> {
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Index { conn })
    }

    pub fn open_in_memory() -> Result<Index> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Index { conn })
    }

    pub fn upsert_session(&self, meta: &Meta) -> Result<()> {
        self.conn.execute(
            "INSERT INTO sessions (id, title, started_at, ended_at, app, conversation_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET title = ?2, started_at = ?3, ended_at = ?4, app = ?5,
                 conversation_count = ?6",
            params![
                meta.id,
                meta.title,
                meta.started_at.to_string(),
                meta.ended_at.as_ref().map(|z| z.to_string()),
                meta.app_label(),
                meta.conversations.len() as i64,
            ],
        )?;
        Ok(())
    }

    pub fn replace_lines(&self, session_id: &str, conversation: u32, lines: &[Line]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM lines WHERE session_id = ?1 AND conversation = ?2", params![session_id, conversation])?;
        {
            let mut insert = tx.prepare(
                "INSERT INTO lines (text, session_id, conversation, t_ms, track) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for l in lines {
                insert.execute(params![l.text, session_id, conversation, l.start_ms as i64, l.track.as_str()])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn replace_summary(&self, session_id: &str, conversation: u32, text: Option<&str>) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM summaries WHERE session_id = ?1 AND conversation = ?2",
            params![session_id, conversation],
        )?;
        if let Some(text) = text {
            tx.execute(
                "INSERT INTO summaries (text, session_id, conversation) VALUES (?1, ?2, ?3)",
                params![text, session_id, conversation],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM lines WHERE session_id = ?1", params![session_id])?;
        tx.execute("DELETE FROM summaries WHERE session_id = ?1", params![session_id])?;
        tx.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
        tx.commit()?;
        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM lines; DELETE FROM summaries; DELETE FROM sessions;")?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let Some(fts) = fts_query(query) else { return Ok(Vec::new()) };
        let mut stmt = self.conn.prepare(
            "SELECT 'line', lines.session_id, lines.conversation, lines.t_ms, lines.track, lines.text,
                    s.started_at, s.title, bm25(lines) AS rank
               FROM lines JOIN sessions s ON s.id = lines.session_id
              WHERE lines MATCH ?1
             UNION ALL
             SELECT 'summary', summaries.session_id, summaries.conversation, NULL, NULL,
                    snippet(summaries, 0, '[', ']', '…', 12),
                    s.started_at, s.title, bm25(summaries) AS rank
               FROM summaries JOIN sessions s ON s.id = summaries.session_id
              WHERE summaries MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![fts, limit as i64], |r| {
            let kind: String = r.get(0)?;
            Ok(Hit {
                kind: if kind == "line" { HitKind::Line } else { HitKind::Summary },
                session_id: r.get(1)?,
                conversation: r.get::<_, i64>(2)? as u32,
                t_ms: r.get::<_, Option<i64>>(3)?.map(|v| v as u64),
                track: r.get(4)?,
                text: r.get(5)?,
                started_at: r.get(6)?,
                title: r.get(7)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
}

/// Turns free text into an FTS5 query where every term is a quoted literal.
pub fn fts_query(input: &str) -> Option<String> {
    let terms: Vec<String> = input
        .split_whitespace()
        .filter(|t| t.chars().any(char::is_alphanumeric))
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() { None } else { Some(terms.join(" ")) }
}
```

Note: the test asserts `search("AND")` returns nothing because the quoted literal `"AND"` matches no stored word. That is the intended, literal behavior.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib index::`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: add SQLite FTS5 search index

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Summarizer trait, prompt, and the command backend

**Files:**
- Create: `src/summarize/mod.rs`, `src/summarize/prompt.md`, `src/summarize/command.rs`
- Modify: `src/lib.rs` (add `pub mod summarize;`)

**Interfaces:**
- Consumes: `config::{SummaryConfig, SummaryBackend, expand_tilde}`
- Produces: `trait Summarizer { fn name(&self) -> String; fn summarize(&self, prompt: &str) -> Result<String>; }`, `summarize::DEFAULT_PROMPT: &str`, `summarize::PromptContext<'a> { date: String, time_range: String, app: String, title: Option<&'a str>, user_name: &'a str, me_label: &'a str, others_label: &'a str, language: &'a str }`, `summarize::build_prompt(template: &str, &PromptContext, transcript: &str) -> String`, `summarize::language_instruction(&str) -> String`, `summarize::load_template(&SummaryConfig) -> Result<String>`, `summarize::from_config(&SummaryConfig) -> Result<Option<Box<dyn Summarizer>>>`, `command::CommandSummarizer::new(Vec<String>, std::time::Duration)`
- Task 11 adds `openai` and `anthropic` modules; until then `from_config` returns an error for those backends (replaced in Task 11 Step 4).

- [ ] **Step 1: Write the prompt template**

`src/summarize/prompt.md`:
```markdown
You are writing meeting notes from a call transcript.

Rules:
- Keep names, technical terms, identifiers, and ticket or merge-request references exactly as spoken. Never translate them.
- When the exact wording matters, quote it briefly in its original language.
- Use these sections, in this order, and leave out any that would be empty:
  ## Summary
  ## Decisions
  ## Action items (one bullet per item, written as "who: what")
  ## Open questions
- Be concise. Output only the notes, in Markdown, with no preamble.
```

- [ ] **Step 2: Write the failing tests**

At the bottom of `src/summarize/mod.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(user: &'a str, language: &'a str) -> PromptContext<'a> {
        PromptContext {
            date: "2026-10-08".into(),
            time_range: "09:40 to 09:54".into(),
            app: "firefox".into(),
            title: Some("Sync"),
            user_name: user,
            me_label: "Me",
            others_label: "Others",
            language,
        }
    }

    #[test]
    fn prompt_has_template_context_and_transcript() {
        let p = build_prompt("Do notes.\n", &ctx("Alex", "it"), "[09:40:11] Me: Ciao\n");
        assert!(p.starts_with("Do notes.\n\n## Context\n"));
        assert!(p.contains("- Date: 2026-10-08\n- Time: 09:40 to 09:54\n- App: firefox\n- Title: Sync\n"));
        assert!(p.contains("\"Me\" is Alex. \"Others\" is everyone else on the call."));
        assert!(p.contains("Write the notes in Italian."));
        assert!(p.ends_with("## Transcript\n\n[09:40:11] Me: Ciao\n"));
    }

    #[test]
    fn prompt_without_user_name_or_fixed_language() {
        let p = build_prompt(DEFAULT_PROMPT, &ctx("  ", "auto"), "x");
        assert!(p.contains("\"Me\" is the person who recorded the call."));
        assert!(p.contains("dominant language of the transcript"));
        assert_eq!(language_instruction("xx"), "Write the notes in the language with code \"xx\".");
    }

    #[test]
    fn from_config_none_and_bad_command() {
        let mut cfg = SummaryConfig::default();
        assert!(from_config(&cfg).unwrap().is_none());
        cfg.backend = SummaryBackend::Command;
        assert!(format!("{:#}", from_config(&cfg).err().unwrap()).contains("summary.command"));
        cfg.command = vec!["cat".into()];
        assert_eq!(from_config(&cfg).unwrap().unwrap().name(), "command: cat");
    }

    #[test]
    fn custom_template_is_read_from_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("prompt.md");
        std::fs::write(&p, "Custom.").unwrap();
        let cfg = SummaryConfig { prompt: Some(p), ..Default::default() };
        assert_eq!(load_template(&cfg).unwrap(), "Custom.");
        assert_eq!(load_template(&SummaryConfig::default()).unwrap(), DEFAULT_PROMPT);
    }
}
```

At the bottom of `src/summarize/command.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str, timeout_ms: u64) -> CommandSummarizer {
        CommandSummarizer::new(vec!["sh".into(), "-c".into(), script.into()], Duration::from_millis(timeout_ms))
    }

    #[test]
    fn stdout_becomes_the_summary() {
        assert_eq!(sh("cat", 5_000).summarize("  notes  \n").unwrap(), "notes");
    }

    #[test]
    fn large_prompts_do_not_deadlock() {
        let big = "word ".repeat(400_000);
        assert_eq!(sh("wc -w", 10_000).summarize(&big).unwrap(), "400000");
    }

    #[test]
    fn failure_includes_stderr() {
        let err = sh("echo oops >&2; exit 3", 5_000).summarize("x").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("oops") && msg.contains("exit"), "{msg}");
    }

    #[test]
    fn empty_output_is_an_error() {
        assert!(format!("{:#}", sh("cat >/dev/null", 5_000).summarize("x").unwrap_err()).contains("no output"));
    }

    #[test]
    fn hanging_command_times_out() {
        let started = std::time::Instant::now();
        let err = sh("sleep 30", 300).summarize("x").unwrap_err();
        assert!(format!("{err:#}").contains("did not finish"), "{err:#}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn missing_program_is_reported() {
        let s = CommandSummarizer::new(vec!["kute-no-such-program".into()], Duration::from_secs(1));
        assert!(format!("{:#}", s.summarize("x").unwrap_err()).contains("kute-no-such-program"));
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --lib summarize::`
Expected: compile errors.

- [ ] **Step 4: Implement**

Top of `src/summarize/mod.rs`:
```rust
//! Summarization backends and the prompt they get.

pub mod command;

use crate::config::{SummaryBackend, SummaryConfig, expand_tilde};
use anyhow::{Context, Result, bail};

pub trait Summarizer {
    /// Recorded in `meta.toml`, e.g. `command: claude -p`.
    fn name(&self) -> String;
    fn summarize(&self, prompt: &str) -> Result<String>;
}

pub const DEFAULT_PROMPT: &str = include_str!("prompt.md");

pub struct PromptContext<'a> {
    pub date: String,
    pub time_range: String,
    pub app: String,
    pub title: Option<&'a str>,
    pub user_name: &'a str,
    pub me_label: &'a str,
    pub others_label: &'a str,
    pub language: &'a str,
}

/// Template, then a context block kute writes itself, then the transcript.
pub fn build_prompt(template: &str, ctx: &PromptContext, transcript: &str) -> String {
    let mut p = template.trim_end().to_string();
    p.push_str("\n\n## Context\n\n");
    p.push_str(&format!("- Date: {}\n- Time: {}\n- App: {}\n", ctx.date, ctx.time_range, ctx.app));
    if let Some(title) = ctx.title {
        p.push_str(&format!("- Title: {title}\n"));
    }
    let who = match ctx.user_name.trim() {
        "" => "the person who recorded the call".to_string(),
        name => name.to_string(),
    };
    p.push_str(&format!(
        "- \"{}\" is {who}. \"{}\" is everyone else on the call.\n",
        ctx.me_label, ctx.others_label
    ));
    p.push_str(&format!("- {}\n", language_instruction(ctx.language)));
    p.push_str("\n## Transcript\n\n");
    p.push_str(transcript.trim_end());
    p.push('\n');
    p
}

pub fn language_instruction(code: &str) -> String {
    if code == "auto" {
        return "Write the notes in the dominant language of the transcript.".into();
    }
    let name = match code {
        "en" => "English",
        "it" => "Italian",
        "de" => "German",
        "fr" => "French",
        "es" => "Spanish",
        "pt" => "Portuguese",
        "nl" => "Dutch",
        other => return format!("Write the notes in the language with code \"{other}\"."),
    };
    format!("Write the notes in {name}.")
}

pub fn load_template(cfg: &SummaryConfig) -> Result<String> {
    match &cfg.prompt {
        Some(path) => {
            let path = expand_tilde(path);
            std::fs::read_to_string(&path).with_context(|| format!("reading prompt template {}", path.display()))
        }
        None => Ok(DEFAULT_PROMPT.to_string()),
    }
}

pub fn from_config(cfg: &SummaryConfig) -> Result<Option<Box<dyn Summarizer>>> {
    let timeout = cfg.timeout.unsigned_abs();
    Ok(match cfg.backend {
        SummaryBackend::None => None,
        SummaryBackend::Command => {
            if cfg.command.is_empty() {
                bail!("summary.backend = \"command\" needs summary.command, e.g. [\"claude\", \"-p\"]");
            }
            Some(Box::new(command::CommandSummarizer::new(cfg.command.clone(), timeout)))
        }
        SummaryBackend::Openai | SummaryBackend::Anthropic => {
            bail!("this summary backend is added in the next task")
        }
    })
}
```

`src/summarize/command.rs` (above the tests):
```rust
//! Runs any command: the prompt goes to stdin, stdout is the summary.

use super::Summarizer;
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use wait_timeout::ChildExt;

pub struct CommandSummarizer {
    argv: Vec<String>,
    timeout: Duration,
}

impl CommandSummarizer {
    pub fn new(argv: Vec<String>, timeout: Duration) -> CommandSummarizer {
        CommandSummarizer { argv, timeout }
    }
}

impl Summarizer for CommandSummarizer {
    fn name(&self) -> String {
        format!("command: {}", self.argv.join(" "))
    }

    fn summarize(&self, prompt: &str) -> Result<String> {
        let program = &self.argv[0];
        let mut child = Command::new(program)
            .args(&self.argv[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {program:?}"))?;

        // Feed stdin and drain stdout/stderr on their own threads so a large prompt or
        // a chatty command can't fill a pipe and deadlock.
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let input = prompt.to_owned();
        let writer = thread::spawn(move || {
            let _ = stdin.write_all(input.as_bytes());
        });
        let mut stdout = child.stdout.take().expect("stdout is piped");
        let out_reader = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        });
        let mut stderr = child.stderr.take().expect("stderr is piped");
        let err_reader = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });

        let Some(status) = child.wait_timeout(self.timeout)? else {
            let _ = child.kill();
            let _ = child.wait();
            // Don't join the readers: a grandchild may still hold the pipes open.
            bail!("{program} did not finish within {} s", self.timeout.as_secs());
        };
        let _ = writer.join();
        let out = out_reader.join().unwrap_or_default();
        let err = err_reader.join().unwrap_or_default();
        if !status.success() {
            let err = String::from_utf8_lossy(&err);
            let tail: String = err.trim().chars().rev().take(2_000).collect::<Vec<_>>().into_iter().rev().collect();
            bail!("{program} failed ({status}): {tail}");
        }
        let text = String::from_utf8_lossy(&out).trim().to_string();
        if text.is_empty() {
            bail!("{program} produced no output");
        }
        Ok(text)
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib summarize::`
Expected: 10 tests PASS. (`failure_includes_stderr` relies on `ExitStatus`'s display, `exit status: 3`, containing "exit".)

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: add summarizer trait, prompt, and command backend

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: OpenAI-compatible and Anthropic summarizers

**Files:**
- Create: `src/summarize/openai.rs`, `src/summarize/anthropic.rs`
- Modify: `src/summarize/mod.rs` (declare the modules, add `http_agent`, wire `from_config`)

**Interfaces:**
- Consumes: `summarize::Summarizer`, `config::SummaryConfig`
- Produces: `openai::OpenAiSummarizer::new(base_url: String, model: String, api_key: Option<String>, timeout: Duration)`, `openai::DEFAULT_BASE_URL = "http://localhost:11434/v1"`, `anthropic::AnthropicSummarizer::new(base_url: String, model: String, api_key: String, timeout: Duration)`, `anthropic::DEFAULT_BASE_URL = "https://api.anthropic.com"`, `anthropic::DEFAULT_MODEL = "claude-opus-5-5"`

- [ ] **Step 1: Write the failing tests**

At the bottom of `src/summarize/openai.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Matcher;

    #[test]
    fn posts_chat_completion_and_reads_content() {
        let mut server = mockito::Server::new();
        let m = server
            .mock("POST", "/v1/chat/completions")
            .match_header("authorization", "Bearer k")
            .match_body(Matcher::PartialJson(serde_json::json!({
                "model": "qwen3:8b",
                "messages": [{"role": "user", "content": "the prompt"}]
            })))
            .with_body(r#"{"choices":[{"message":{"role":"assistant","content":"  ## Summary\n\nok  "}}]}"#)
            .create();
        let s = OpenAiSummarizer::new(format!("{}/v1/", server.url()), "qwen3:8b".into(), Some("k".into()), Duration::from_secs(5));
        assert_eq!(s.summarize("the prompt").unwrap(), "## Summary\n\nok");
        assert_eq!(s.name(), "openai: qwen3:8b");
        m.assert();
    }

    #[test]
    fn no_key_means_no_auth_header() {
        let mut server = mockito::Server::new();
        let m = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", Matcher::Missing)
            .with_body(r#"{"choices":[{"message":{"content":"x"}}]}"#)
            .create();
        let s = OpenAiSummarizer::new(server.url(), "m".into(), None, Duration::from_secs(5));
        s.summarize("p").unwrap();
        m.assert();
    }

    #[test]
    fn http_errors_include_status_and_body() {
        let mut server = mockito::Server::new();
        server.mock("POST", "/chat/completions").with_status(500).with_body("model not loaded").create();
        let s = OpenAiSummarizer::new(server.url(), "m".into(), None, Duration::from_secs(5));
        let msg = format!("{:#}", s.summarize("p").unwrap_err());
        assert!(msg.contains("500") && msg.contains("model not loaded"), "{msg}");
    }
}
```

At the bottom of `src/summarize/anthropic.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Matcher;

    fn summarizer(server: &mockito::Server) -> AnthropicSummarizer {
        AnthropicSummarizer::new(server.url(), DEFAULT_MODEL.into(), "k".into(), Duration::from_secs(5))
    }

    #[test]
    fn sends_required_headers_and_reads_text_blocks_only() {
        let mut server = mockito::Server::new();
        let m = server
            .mock("POST", "/v1/messages")
            .match_header("x-api-key", "k")
            .match_header("anthropic-version", "2023-06-01")
            .match_header("anthropic-beta", "server-side-fallback-2026-07-01")
            .match_body(Matcher::PartialJson(serde_json::json!({
                "model": "claude-opus-5-5",
                "max_tokens": 16000,
                "fallbacks": "default",
                "messages": [{"role": "user", "content": "the prompt"}]
            })))
            .with_body(
                r###"{"stop_reason":"end_turn","content":[
                    {"type":"thinking","thinking":""},
                    {"type":"text","text":"## Summary\n\n"},
                    {"type":"text","text":"Done."}]}"###,
            )
            .create();
        assert_eq!(summarizer(&server).summarize("the prompt").unwrap(), "## Summary\n\nDone.");
        assert_eq!(summarizer(&server).name(), "anthropic: claude-opus-5-5");
        m.assert();
    }

    #[test]
    fn refusal_is_an_error() {
        let mut server = mockito::Server::new();
        server
            .mock("POST", "/v1/messages")
            .with_body(r#"{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber"},"content":[]}"#)
            .create();
        let msg = format!("{:#}", summarizer(&server).summarize("p").unwrap_err());
        assert!(msg.contains("declined") && msg.contains("cyber"), "{msg}");
    }

    #[test]
    fn truncated_output_is_an_error() {
        let mut server = mockito::Server::new();
        server
            .mock("POST", "/v1/messages")
            .with_body(r#"{"stop_reason":"max_tokens","content":[{"type":"text","text":"half"}]}"#)
            .create();
        assert!(format!("{:#}", summarizer(&server).summarize("p").unwrap_err()).contains("max_tokens"));
    }

    #[test]
    fn http_errors_include_status_and_body() {
        let mut server = mockito::Server::new();
        server.mock("POST", "/v1/messages").with_status(401).with_body(r#"{"error":{"message":"invalid x-api-key"}}"#).create();
        let msg = format!("{:#}", summarizer(&server).summarize("p").unwrap_err());
        assert!(msg.contains("401") && msg.contains("invalid x-api-key"), "{msg}");
    }
}
```

Add to the tests in `src/summarize/mod.rs`:
```rust
    #[test]
    fn from_config_http_backends() {
        let mut cfg = SummaryConfig { backend: SummaryBackend::Openai, ..Default::default() };
        assert!(format!("{:#}", from_config(&cfg).err().unwrap()).contains("summary.model"));
        cfg.model = Some("qwen3:8b".into());
        assert_eq!(from_config(&cfg).unwrap().unwrap().name(), "openai: qwen3:8b");

        let cfg = SummaryConfig {
            backend: SummaryBackend::Anthropic,
            api_key_env: Some("KUTE_TEST_UNSET_KEY_VAR".into()),
            ..Default::default()
        };
        assert!(format!("{:#}", from_config(&cfg).err().unwrap()).contains("KUTE_TEST_UNSET_KEY_VAR"));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib summarize::`
Expected: compile errors (modules not declared).

- [ ] **Step 3: Implement the backends**

`src/summarize/openai.rs` (above the tests):
```rust
//! Any OpenAI-compatible Chat Completions endpoint (Ollama, llama.cpp, vLLM, OpenRouter...).

use super::{Summarizer, http_agent};
use anyhow::{Context, Result, bail};
use std::time::Duration;

pub const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

pub struct OpenAiSummarizer {
    base_url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl OpenAiSummarizer {
    pub fn new(base_url: String, model: String, api_key: Option<String>, timeout: Duration) -> OpenAiSummarizer {
        OpenAiSummarizer { base_url, model, api_key, agent: http_agent(timeout) }
    }
}

impl Summarizer for OpenAiSummarizer {
    fn name(&self) -> String {
        format!("openai: {}", self.model)
    }

    fn summarize(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        });
        let mut request = self.agent.post(&url);
        if let Some(key) = &self.api_key {
            request = request.header("authorization", format!("Bearer {key}"));
        }
        let mut response = request.send_json(&body).with_context(|| format!("POST {url}"))?;
        let status = response.status().as_u16();
        let text = response.body_mut().read_to_string()?;
        if !(200..300).contains(&status) {
            bail!("{url} returned HTTP {status}: {}", text.chars().take(500).collect::<String>());
        }
        let value: serde_json::Value = serde_json::from_str(&text).context("the summarizer returned invalid JSON")?;
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .context("the response has no choices[0].message.content")?
            .trim();
        if content.is_empty() {
            bail!("the summarizer returned an empty answer");
        }
        Ok(content.to_string())
    }
}
```

`src/summarize/anthropic.rs` (above the tests):
```rust
//! Anthropic Messages API over raw HTTP (there is no official Rust SDK).

use super::{Summarizer, http_agent};
use anyhow::{Context, Result, bail};
use std::time::Duration;

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";
const MAX_TOKENS: u32 = 16_000;

pub struct AnthropicSummarizer {
    base_url: String,
    model: String,
    api_key: String,
    agent: ureq::Agent,
}

impl AnthropicSummarizer {
    pub fn new(base_url: String, model: String, api_key: String, timeout: Duration) -> AnthropicSummarizer {
        AnthropicSummarizer { base_url, model, api_key, agent: http_agent(timeout) }
    }
}

impl Summarizer for AnthropicSummarizer {
    fn name(&self) -> String {
        format!("anthropic: {}", self.model)
    }

    fn summarize(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        // `fallbacks: "default"` lets the API retry a declined request on the model
        // Anthropic recommends for that refusal category.
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "fallbacks": "default",
            "messages": [{"role": "user", "content": prompt}],
        });
        let mut response = self
            .agent
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "server-side-fallback-2026-07-01")
            .send_json(&body)
            .with_context(|| format!("POST {url}"))?;
        let status = response.status().as_u16();
        let text = response.body_mut().read_to_string()?;
        if !(200..300).contains(&status) {
            bail!("Anthropic API returned HTTP {status}: {}", text.chars().take(500).collect::<String>());
        }
        let value: serde_json::Value = serde_json::from_str(&text).context("the Anthropic API returned invalid JSON")?;
        match value["stop_reason"].as_str() {
            Some("refusal") => bail!(
                "the model declined to summarize this conversation (category: {})",
                value["stop_details"]["category"].as_str().unwrap_or("unknown")
            ),
            Some("max_tokens") => bail!("the summary was cut off at max_tokens"),
            _ => {}
        }
        let summary: String = value["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|block| block["type"] == "text")
            .filter_map(|block| block["text"].as_str())
            .collect();
        let summary = summary.trim();
        if summary.is_empty() {
            bail!("the Anthropic API returned no text");
        }
        Ok(summary.to_string())
    }
}
```

- [ ] **Step 4: Wire them into `summarize/mod.rs`**

Add below `pub mod command;`:
```rust
pub mod anthropic;
pub mod openai;
```

Add this function to `src/summarize/mod.rs`:
```rust
/// Blocking HTTP client with a whole-request timeout. Non-2xx statuses are returned, not
/// turned into errors, so callers can include the response body in their message.
pub(crate) fn http_agent(timeout: std::time::Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
}

fn api_key(env: &str) -> Result<String> {
    std::env::var(env).with_context(|| format!("environment variable {env} is not set"))
}
```

Replace the `SummaryBackend::Openai | SummaryBackend::Anthropic` arm of `from_config` with:
```rust
        SummaryBackend::Openai => {
            let model = cfg.model.clone().context("summary.backend = \"openai\" needs summary.model")?;
            let base = cfg.base_url.clone().unwrap_or_else(|| openai::DEFAULT_BASE_URL.into());
            let key = cfg.api_key_env.as_deref().map(api_key).transpose()?;
            Some(Box::new(openai::OpenAiSummarizer::new(base, model, key, timeout)))
        }
        SummaryBackend::Anthropic => {
            let model = cfg.model.clone().unwrap_or_else(|| anthropic::DEFAULT_MODEL.into());
            let base = cfg.base_url.clone().unwrap_or_else(|| anthropic::DEFAULT_BASE_URL.into());
            let key = api_key(cfg.api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY"))?;
            Some(Box::new(anthropic::AnthropicSummarizer::new(base, model, key, timeout)))
        }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib summarize::`
Expected: 18 tests PASS.

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: add OpenAI-compatible and Anthropic summarizers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Processing pipeline

**Files:**
- Create: `src/pipeline.rs`
- Modify: `src/lib.rs` (add `pub mod pipeline;`)

**Interfaces:**
- Consumes: everything above: `Store`, `Session`, `Meta`, `Index`, `Config`, `transcribe::{SpeechDetector, Transcriber, transcribe_track}`, `transcribe::whisper::{SileroVad, WhisperTranscriber}`, `models::{resolve_whisper, resolve_vad}`, `summarize::{Summarizer, PromptContext, build_prompt}`, `render::*`, `audio::*`
- Produces:
  - `trait AsrProvider { fn get(&mut self) -> Result<(&mut dyn SpeechDetector, &mut dyn Transcriber)>; fn release(&mut self); }`
  - `pipeline::WhisperProvider::new(&Config, &Store) -> Result<WhisperProvider>` (checks model files exist, loads lazily)
  - `pipeline::Pipeline<'a> { index: &'a Index, config: &'a Config, asr: Option<&'a mut dyn AsrProvider>, summarizer: Option<&'a dyn Summarizer>, template: String, quiet: bool }`
  - `Pipeline::process(&mut self, &Session) -> Result<Report>`, `Pipeline::retranscribe(&mut self, &Session) -> Result<Report>`, `Pipeline::resummarize(&mut self, &Session, only: Option<u32>) -> Result<Report>`
  - `pipeline::Report { transcribed: Vec<u32>, summarized: Vec<u32>, skipped: Vec<u32>, failed: Vec<(u32, String)>, compressed: bool, compression_error: Option<String> }`
  - `pipeline::write_documents(&Session, &Meta, &Labels) -> Result<()>` (rewrites `transcript.md` and `summary.md`)
  - `pipeline::index_session(&Index, &Session) -> Result<()>`, `pipeline::reindex_all(&Store, &Index) -> Result<usize>`

Order inside `process`: (1) transcribe every conversation that needs it, (2) release the ASR models so a local LLM gets the GPU, (3) summarize every conversation that needs it, (4) compress once the session has ended and every conversation is settled. Failures are recorded per conversation and never stop the others. A session in `recording` state is refused.

- [ ] **Step 1: Write the failing tests**

At the bottom of `src/pipeline.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::write_test_wav;
    use crate::store::{Conversation, Trigger, TriggerKind, Word};
    use crate::transcribe::{RawSegment, Span};
    use std::cell::RefCell;

    /// "Speech" is any non-zero sample; the transcript says which level it heard.
    struct LevelVad;
    impl SpeechDetector for LevelVad {
        fn speech_regions(&mut self, samples: &[f32]) -> Result<Vec<Span>> {
            if samples.iter().any(|s| *s != 0.0) {
                Ok(vec![Span { start_ms: 0, end_ms: crate::audio::samples_to_ms(samples.len()) }])
            } else {
                Ok(vec![])
            }
        }
    }
    struct LevelAsr;
    impl Transcriber for LevelAsr {
        fn model_name(&self) -> &str {
            "fake"
        }
        fn detect_language(&mut self, _: &[f32]) -> Result<String> {
            Ok("en".into())
        }
        fn transcribe(&mut self, samples: &[f32], _: &str) -> Result<Vec<RawSegment>> {
            let level = (samples[0] * 100.0).round() as i32;
            let text = format!(" level {level} words here");
            Ok(vec![RawSegment {
                start_ms: 0,
                end_ms: 500,
                text: text.clone(),
                words: vec![Word { start_ms: 0, end_ms: 500, text }],
            }])
        }
    }
    struct FakeProvider {
        vad: LevelVad,
        asr: LevelAsr,
        released: usize,
        fail: bool,
    }
    impl FakeProvider {
        fn new() -> Self {
            FakeProvider { vad: LevelVad, asr: LevelAsr, released: 0, fail: false }
        }
    }
    impl AsrProvider for FakeProvider {
        fn get(&mut self) -> Result<(&mut dyn SpeechDetector, &mut dyn Transcriber)> {
            if self.fail {
                anyhow::bail!("GPU out of memory");
            }
            Ok((&mut self.vad as &mut dyn SpeechDetector, &mut self.asr as &mut dyn Transcriber))
        }
        fn release(&mut self) {
            self.released += 1;
        }
    }
    struct FakeSummarizer {
        prompts: RefCell<Vec<String>>,
        fail: bool,
    }
    impl Summarizer for FakeSummarizer {
        fn name(&self) -> String {
            "fake".into()
        }
        fn summarize(&self, prompt: &str) -> Result<String> {
            self.prompts.borrow_mut().push(prompt.to_string());
            if self.fail {
                anyhow::bail!("rate limited");
            }
            Ok("## Summary\n\nIt went fine.".into())
        }
    }

    struct Fixture {
        _tmp: tempfile::TempDir,
        store: Store,
        index: Index,
        config: Config,
        session: Session,
    }

    /// One session, `me` at level 0.25 and `others` at level 0.5, one conversation over
    /// the first `conv_ms` milliseconds.
    fn fixture(me_secs: f32, others_secs: f32, conv_ms: u64) -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let index = Index::open(&store.index_path()).unwrap();
        let mut config = Config::default();
        config.summary.min_words = 3;
        let start: jiff::Zoned = "2026-10-08T09:00:00+02:00[+02:00]".parse().unwrap();
        let session = store.create_session(&start).unwrap();
        write_test_wav(&session.wav_path(Track::Me), me_secs, 8_192);
        write_test_wav(&session.wav_path(Track::Others), others_secs, 16_384);
        let mut meta = Meta::new(
            session.id.clone(),
            start.clone(),
            Trigger { kind: TriggerKind::Import, app_name: None, binary: None },
            "en".into(),
        );
        meta.state = SessionState::Processing;
        meta.ended_at = Some(crate::time::add_ms(&start, conv_ms));
        meta.conversations.push(Conversation::new(1, 0, conv_ms));
        session.save_meta(&meta).unwrap();
        Fixture { _tmp: tmp, store, index, config, session }
    }

    fn pipeline<'a>(
        f: &'a Fixture,
        asr: &'a mut FakeProvider,
        summarizer: Option<&'a FakeSummarizer>,
    ) -> Pipeline<'a> {
        Pipeline {
            index: &f.index,
            config: &f.config,
            asr: Some(asr),
            summarizer: summarizer.map(|s| s as &dyn Summarizer),
            template: crate::summarize::DEFAULT_PROMPT.to_string(),
            quiet: true,
        }
    }

    fn summarizer(fail: bool) -> FakeSummarizer {
        FakeSummarizer { prompts: RefCell::new(Vec::new()), fail }
    }

    #[test]
    fn full_run_transcribes_summarizes_compresses_and_indexes() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        let sum = summarizer(false);
        let report = pipeline(&f, &mut asr, Some(&sum)).process(&f.session).unwrap();
        assert_eq!(report.transcribed, vec![1]);
        assert_eq!(report.summarized, vec![1]);
        assert!(report.compressed, "{:?}", report.compression_error);
        assert_eq!(asr.released, 1);

        let meta = f.session.load_meta().unwrap();
        assert_eq!(meta.state, SessionState::Done);
        assert_eq!(meta.conversations[0].stage, Stage::Summarized);
        assert_eq!(meta.conversations[0].languages.me.as_deref(), Some("en"));
        assert_eq!(meta.conversations[0].whisper_model.as_deref(), Some("fake"));
        assert_eq!(meta.conversations[0].summarizer.as_deref(), Some("fake"));

        let transcript = std::fs::read_to_string(f.session.transcript_path()).unwrap();
        assert!(transcript.contains("[09:00:00] Me: level 25 words here"), "{transcript}");
        assert!(transcript.contains("[09:00:00] Others: level 50 words here"), "{transcript}");
        assert!(std::fs::read_to_string(f.session.summary_path()).unwrap().contains("It went fine."));
        assert!(sum.prompts.borrow()[0].contains("Me: level 25 words here"));

        for t in Track::BOTH {
            assert!(!f.session.wav_path(t).exists());
            assert!(f.session.opus_path(t).is_file());
        }
        assert_eq!(f.index.search("level", 10).unwrap().len(), 2);
        assert_eq!(f.index.search("fine", 10).unwrap().len(), 1);
    }

    #[test]
    fn silent_track_still_produces_transcript() {
        let f = fixture(2.0, 2.0, 2_000);
        write_test_wav(&f.session.wav_path(Track::Others), 2.0, 0);
        let mut asr = FakeProvider::new();
        pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        let meta = f.session.load_meta().unwrap();
        assert_eq!(meta.conversations[0].languages.others, None);
        assert_eq!(meta.conversations[0].stage, Stage::Skipped);
        let transcript = std::fs::read_to_string(f.session.transcript_path()).unwrap();
        assert!(transcript.contains("Me: level 25") && !transcript.contains("Others:"), "{transcript}");
    }

    #[test]
    fn track_offset_and_short_track_are_handled() {
        let f = fixture(3.0, 1.0, 3_000);
        let mut meta = f.session.load_meta().unwrap();
        meta.tracks.others.offset_ms = 1_500;
        f.session.save_meta(&meta).unwrap();
        let mut asr = FakeProvider::new();
        pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        let others = f.session.load_track(Track::Others).unwrap();
        assert_eq!(others.segments[0].start_ms, 1_500);
        let transcript = std::fs::read_to_string(f.session.transcript_path()).unwrap();
        assert!(transcript.contains("[09:00:01] Others: level 50"), "{transcript}");
    }

    #[test]
    fn transcription_failure_is_recorded_and_resumable() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider { fail: true, ..FakeProvider::new() };
        let report = pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        assert_eq!(report.failed.len(), 1);
        let meta = f.session.load_meta().unwrap();
        let c = &meta.conversations[0];
        assert_eq!((c.stage, c.failed_step, c.attempts), (Stage::Failed, Some(Step::Transcribe), 1));
        assert!(c.error.as_deref().unwrap().contains("GPU out of memory"));
        // A failed conversation counts as settled, so the audio was compressed; the retry
        // decodes the Opus files.
        let mut asr = FakeProvider::new();
        let report = pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        assert_eq!(report.transcribed, vec![1]);
        assert!(f.session.load_meta().unwrap().conversations[0].error.is_none());
    }

    #[test]
    fn summary_failure_keeps_transcript_and_retries() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        let bad = summarizer(true);
        let report = pipeline(&f, &mut asr, Some(&bad)).process(&f.session).unwrap();
        assert_eq!(report.failed[0].0, 1);
        assert!(f.session.transcript_path().is_file());
        let good = summarizer(false);
        let report = pipeline(&f, &mut asr, Some(&good)).process(&f.session).unwrap();
        assert_eq!(report.summarized, vec![1]);
    }

    #[test]
    fn failed_compression_keeps_wavs_and_can_resume() {
        let f = fixture(2.0, 2.0, 2_000);
        // A directory where ffmpeg wants to write its temp file makes encoding fail.
        std::fs::create_dir(f.session.dir.join(".me.opus.tmp")).unwrap();
        let mut asr = FakeProvider::new();
        let report = pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        assert!(!report.compressed && report.compression_error.is_some());
        let meta = f.session.load_meta().unwrap();
        assert_eq!(meta.compression.stage, CompressionStage::Failed);
        assert!(f.session.wav_path(Track::Me).is_file() && f.session.wav_path(Track::Others).is_file());

        std::fs::remove_dir(f.session.dir.join(".me.opus.tmp")).unwrap();
        let report = pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        assert!(report.compressed);
        assert!(!f.session.wav_path(Track::Me).exists());
    }

    #[test]
    fn retranscribe_reads_opus_after_compression() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        let report = pipeline(&f, &mut asr, None).retranscribe(&f.session).unwrap();
        assert_eq!(report.transcribed, vec![1]);
        assert!(std::fs::read_to_string(f.session.transcript_path()).unwrap().contains("Me: level"));
        // Decoded temp files are cleaned up.
        assert!(!f.session.dir.join(".decoded-me.wav").exists());
    }

    #[test]
    fn resummarize_only_one_conversation_and_recording_is_refused() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        let sum = summarizer(false);
        pipeline(&f, &mut asr, Some(&sum)).process(&f.session).unwrap();
        let report = pipeline(&f, &mut asr, Some(&sum)).resummarize(&f.session, Some(1)).unwrap();
        assert_eq!(report.summarized, vec![1]);
        assert!(pipeline(&f, &mut asr, Some(&sum)).resummarize(&f.session, Some(7)).is_err());

        let mut meta = f.session.load_meta().unwrap();
        meta.state = SessionState::Recording;
        f.session.save_meta(&meta).unwrap();
        assert!(pipeline(&f, &mut asr, None).process(&f.session).is_err());
    }

    #[test]
    fn reindex_rebuilds_from_files() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        f.index.clear().unwrap();
        assert!(f.index.search("level", 10).unwrap().is_empty());
        assert_eq!(reindex_all(&f.store, &f.index).unwrap(), 1);
        assert_eq!(f.index.search("level", 10).unwrap().len(), 2);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib pipeline::`
Expected: compile errors.

- [ ] **Step 3: Implement**

Top of `src/pipeline.rs`:
```rust
//! Takes a recorded session from WAVs to transcript, summaries, Opus and index entries.
//! All progress lives in `meta.toml`, so any run can be interrupted and resumed.

use crate::audio;
use crate::config::{Config, Labels};
use crate::index::Index;
use crate::models;
use crate::render;
use crate::store::{
    CompressionStage, Meta, Session, SessionState, Stage, Step, Store, Track, write_atomic,
};
use crate::summarize::{PromptContext, Summarizer, build_prompt};
use crate::transcribe::whisper::{SileroVad, WhisperTranscriber};
use crate::transcribe::{SpeechDetector, Transcriber, transcribe_track};
use anyhow::{Context, Result, anyhow, bail};
use std::fs;
use std::path::{Path, PathBuf};

pub trait AsrProvider {
    fn get(&mut self) -> Result<(&mut dyn SpeechDetector, &mut dyn Transcriber)>;
    /// Frees the models (and their VRAM) until the next `get`.
    fn release(&mut self);
}

pub struct WhisperProvider {
    model_path: PathBuf,
    model_name: String,
    vad_path: PathBuf,
    gpu: bool,
    loaded: Option<(SileroVad, WhisperTranscriber)>,
}

impl WhisperProvider {
    /// Checks that the configured model files exist; loading waits until first use.
    pub fn new(config: &Config, store: &Store) -> Result<WhisperProvider> {
        let (model_path, model_name) = models::resolve_whisper(&store.models_dir(), &config.transcribe.model)?;
        let vad_path = models::resolve_vad(&store.models_dir())?;
        Ok(WhisperProvider { model_path, model_name, vad_path, gpu: config.transcribe.gpu, loaded: None })
    }
}

impl AsrProvider for WhisperProvider {
    fn get(&mut self) -> Result<(&mut dyn SpeechDetector, &mut dyn Transcriber)> {
        if self.loaded.is_none() {
            let vad = SileroVad::load(&self.vad_path)?;
            let asr = WhisperTranscriber::load(&self.model_path, &self.model_name, self.gpu)?;
            self.loaded = Some((vad, asr));
        }
        let (vad, asr) = self.loaded.as_mut().expect("loaded above");
        Ok((vad as &mut dyn SpeechDetector, asr as &mut dyn Transcriber))
    }

    fn release(&mut self) {
        self.loaded = None;
    }
}

#[derive(Debug, Default)]
pub struct Report {
    pub transcribed: Vec<u32>,
    pub summarized: Vec<u32>,
    pub skipped: Vec<u32>,
    pub failed: Vec<(u32, String)>,
    pub compressed: bool,
    pub compression_error: Option<String>,
}

pub struct Pipeline<'a> {
    pub index: &'a Index,
    pub config: &'a Config,
    /// `None` means this run doesn't transcribe (e.g. `kute summarize`).
    pub asr: Option<&'a mut dyn AsrProvider>,
    pub summarizer: Option<&'a dyn Summarizer>,
    pub template: String,
    pub quiet: bool,
}

impl Pipeline<'_> {
    pub fn process(&mut self, session: &Session) -> Result<Report> {
        let mut meta = session.load_meta()?;
        if meta.state == SessionState::Recording {
            bail!("session {} is still recording", meta.id);
        }
        let mut report = Report::default();
        let ns: Vec<u32> = meta.conversations.iter().map(|c| c.n).collect();

        if self.asr.is_some() {
            for &n in &ns {
                if !meta.conversation(n).is_some_and(|c| c.needs_transcription()) {
                    continue;
                }
                note(self.quiet, &format!("{}: transcribing conversation {n}", meta.id));
                match self.transcribe_conversation(session, &mut meta, n) {
                    Ok(()) => report.transcribed.push(n),
                    Err(e) => {
                        meta.conversation_mut(n).expect("n comes from meta").fail(Step::Transcribe, &e);
                        session.save_meta(&meta)?;
                        report.failed.push((n, format!("{e:#}")));
                    }
                }
            }
            if let Some(asr) = self.asr.as_deref_mut() {
                asr.release();
            }
        }

        for &n in &ns {
            if !meta.conversation(n).is_some_and(|c| c.needs_summary()) {
                continue;
            }
            match self.summarize_conversation(session, &mut meta, n) {
                Ok(true) => report.summarized.push(n),
                Ok(false) => report.skipped.push(n),
                Err(e) => {
                    meta.conversation_mut(n).expect("n comes from meta").fail(Step::Summarize, &e);
                    session.save_meta(&meta)?;
                    report.failed.push((n, format!("{e:#}")));
                }
            }
        }

        self.compress_if_ready(session, &mut meta, &mut report)?;
        write_documents(session, &meta, &self.config.labels)?;
        index_session(self.index, session)?;
        Ok(report)
    }

    /// Forgets all transcripts and summaries, then processes the session again from the
    /// stored audio (Opus after compression).
    pub fn retranscribe(&mut self, session: &Session) -> Result<Report> {
        let mut meta = session.load_meta()?;
        if meta.state == SessionState::Recording {
            bail!("session {} is still recording", meta.id);
        }
        for c in &mut meta.conversations {
            let (n, start_ms, end_ms) = (c.n, c.start_ms, c.end_ms);
            *c = crate::store::Conversation::new(n, start_ms, end_ms);
        }
        for track in Track::BOTH {
            session.save_track(track, &Default::default())?;
        }
        let dir = session.summaries_dir();
        if dir.is_dir() {
            fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
        }
        session.save_meta(&meta)?;
        self.process(session)
    }

    /// Summarizes again: every transcribed conversation, or only `only`.
    pub fn resummarize(&mut self, session: &Session, only: Option<u32>) -> Result<Report> {
        let mut meta = session.load_meta()?;
        if let Some(n) = only {
            let c = meta.conversation(n).with_context(|| format!("session {} has no conversation {n}", meta.id))?;
            if c.needs_transcription() {
                bail!("conversation {n} has not been transcribed yet; run `kute process {}`", meta.id);
            }
        }
        for c in &mut meta.conversations {
            if only.is_some_and(|n| n != c.n) || c.needs_transcription() {
                continue;
            }
            c.stage = Stage::Transcribed;
            c.clear_error();
        }
        session.save_meta(&meta)?;
        self.process(session)
    }

    fn transcribe_conversation(&mut self, session: &Session, meta: &mut Meta, n: u32) -> Result<()> {
        let conv = meta.conversation(n).expect("n comes from meta").clone();
        let forced = match self.config.transcribe.language.as_str() {
            "auto" => None,
            lang => Some(lang.to_string()),
        };
        let provider = self.asr.as_deref_mut().context("no transcriber available")?;
        let (vad, asr) = provider.get()?;
        let model = asr.model_name().to_string();
        let mut results = Vec::new();
        for track in Track::BOTH {
            let offset = meta.tracks.get(track).offset_ms;
            let base_ms = conv.start_ms.max(offset);
            let samples = if conv.end_ms > base_ms {
                let source = AudioSource::open(session, track)?;
                audio::read_wav_range(source.path(), base_ms - offset, conv.end_ms - offset)?
            } else {
                Vec::new()
            };
            let result = transcribe_track(&samples, base_ms, n, forced.as_deref(), &mut *vad, &mut *asr)
                .with_context(|| format!("transcribing the {} track", track.as_str()))?;
            results.push((track, result));
        }
        let c = meta.conversation_mut(n).expect("n comes from meta");
        for (track, result) in results {
            let mut transcript = session.load_track(track)?;
            transcript.replace_conversation(n, result.segments);
            session.save_track(track, &transcript)?;
            c.languages.set(track, result.language);
        }
        c.stage = Stage::Transcribed;
        c.whisper_model = Some(model);
        c.clear_error();
        session.save_meta(meta)?;
        write_documents(session, meta, &self.config.labels)?;
        index_session(self.index, session)
    }

    /// Returns `Ok(false)` when the conversation is skipped (no summarizer, or too short).
    fn summarize_conversation(&mut self, session: &Session, meta: &mut Meta, n: u32) -> Result<bool> {
        let conv = meta.conversation(n).expect("n comes from meta").clone();
        let me = session.load_track(Track::Me)?;
        let others = session.load_track(Track::Others)?;
        let lines = render::conversation_lines(&conv, &me, &others);
        let summary_path = session.conversation_summary_path(n);

        let summarizer = match self.summarizer {
            Some(s) if render::word_count(&lines) >= self.config.summary.min_words => s,
            _ => {
                if summary_path.is_file() {
                    fs::remove_file(&summary_path)?;
                }
                let c = meta.conversation_mut(n).expect("n comes from meta");
                c.stage = Stage::Skipped;
                c.clear_error();
                session.save_meta(meta)?;
                return Ok(false);
            }
        };

        note(self.quiet, &format!("{}: summarizing conversation {n}", meta.id));
        let labels = &self.config.labels;
        let ctx = PromptContext {
            date: meta.started_at.strftime("%Y-%m-%d").to_string(),
            time_range: render::time_range(meta, &conv),
            app: meta.app_label(),
            title: meta.title.as_deref(),
            user_name: &self.config.user.name,
            me_label: &labels.me,
            others_label: &labels.others,
            language: &meta.summary.language,
        };
        let prompt = build_prompt(&self.template, &ctx, &render::render_lines(meta, labels, &lines));
        let text = summarizer.summarize(&prompt)?;

        fs::create_dir_all(session.summaries_dir())?;
        write_atomic(&summary_path, format!("{}\n", text.trim()).as_bytes())?;
        let c = meta.conversation_mut(n).expect("n comes from meta");
        c.stage = Stage::Summarized;
        c.summarizer = Some(summarizer.name());
        c.clear_error();
        session.save_meta(meta)?;
        write_documents(session, meta, labels)?;
        index_session(self.index, session)?;
        Ok(true)
    }

    fn compress_if_ready(&mut self, session: &Session, meta: &mut Meta, report: &mut Report) -> Result<()> {
        if meta.compression.stage == CompressionStage::Done
            || meta.ended_at.is_none()
            || !meta.conversations.iter().all(|c| c.is_settled())
        {
            return Ok(());
        }
        note(self.quiet, &format!("{}: compressing audio", meta.id));
        match compress_tracks(session) {
            Ok(()) => {
                meta.compression.stage = CompressionStage::Done;
                meta.compression.error = None;
                meta.state = SessionState::Done;
                report.compressed = true;
            }
            Err(e) => {
                meta.compression.stage = CompressionStage::Failed;
                meta.compression.error = Some(format!("{e:#}"));
                report.compression_error = Some(format!("{e:#}"));
            }
        }
        session.save_meta(meta)
    }
}

/// Encodes both WAVs to Opus and checks durations; deletes the WAVs only when both passed.
fn compress_tracks(session: &Session) -> Result<()> {
    for track in Track::BOTH {
        let wav = session.wav_path(track);
        if !wav.is_file() {
            continue;
        }
        let tmp = session.dir.join(format!(".{}.opus.tmp", track.as_str()));
        audio::encode_opus(&wav, &tmp)?;
        let wav_ms = audio::wav_duration_ms(&wav)?;
        let opus_ms = audio::probe_duration_ms(&tmp)?;
        if wav_ms.abs_diff(opus_ms) > 500 {
            let _ = fs::remove_file(&tmp);
            bail!("{} track: Opus is {opus_ms} ms but the WAV is {wav_ms} ms", track.as_str());
        }
        fs::rename(&tmp, session.opus_path(track))?;
    }
    for track in Track::BOTH {
        let wav = session.wav_path(track);
        if wav.is_file() && session.opus_path(track).is_file() {
            fs::remove_file(&wav).with_context(|| format!("removing {}", wav.display()))?;
        }
    }
    Ok(())
}

/// The WAV if it still exists, otherwise a temporary WAV decoded from the Opus file.
struct AudioSource {
    path: PathBuf,
    temporary: bool,
}

impl AudioSource {
    fn open(session: &Session, track: Track) -> Result<AudioSource> {
        let wav = session.wav_path(track);
        if wav.is_file() {
            return Ok(AudioSource { path: wav, temporary: false });
        }
        let opus = session.opus_path(track);
        if opus.is_file() {
            let path = session.dir.join(format!(".decoded-{}.wav", track.as_str()));
            audio::normalize_to_wav(&opus, &path)?;
            return Ok(AudioSource { path, temporary: true });
        }
        Err(anyhow!("session {} has no audio for the {} track", session.id, track.as_str()))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for AudioSource {
    fn drop(&mut self) {
        if self.temporary {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Rewrites `transcript.md` and `summary.md` from the session's current state.
pub fn write_documents(session: &Session, meta: &Meta, labels: &Labels) -> Result<()> {
    let me = session.load_track(Track::Me)?;
    let others = session.load_track(Track::Others)?;
    write_atomic(&session.transcript_path(), render::render_transcript(meta, labels, &me, &others).as_bytes())?;

    let mut summaries = Vec::new();
    for c in meta.conversations.iter().filter(|c| c.stage == Stage::Summarized) {
        if let Ok(text) = fs::read_to_string(session.conversation_summary_path(c.n)) {
            summaries.push((c, text));
        }
    }
    let path = session.summary_path();
    if summaries.is_empty() {
        if path.is_file() {
            fs::remove_file(&path)?;
        }
    } else {
        write_atomic(&path, render::render_summary(meta, &summaries).as_bytes())?;
    }
    Ok(())
}

/// Replaces everything the index knows about one session with what is on disk.
pub fn index_session(index: &Index, session: &Session) -> Result<()> {
    let meta = session.load_meta()?;
    index.delete_session(&meta.id)?;
    index.upsert_session(&meta)?;
    let me = session.load_track(Track::Me)?;
    let others = session.load_track(Track::Others)?;
    for c in meta.conversations.iter().filter(|c| !c.needs_transcription()) {
        index.replace_lines(&meta.id, c.n, &render::conversation_lines(c, &me, &others))?;
        if c.stage == Stage::Summarized {
            if let Ok(text) = fs::read_to_string(session.conversation_summary_path(c.n)) {
                index.replace_summary(&meta.id, c.n, Some(&text))?;
            }
        }
    }
    Ok(())
}

/// Rebuilds the index from every session folder. A broken session is reported on stderr
/// and skipped, so one bad folder doesn't block search for the rest.
pub fn reindex_all(store: &Store, index: &Index) -> Result<usize> {
    index.clear()?;
    let mut count = 0;
    for session in store.list()? {
        match index_session(index, &session) {
            Ok(()) => count += 1,
            Err(e) => eprintln!("kute: skipping {}: {e:#}", session.id),
        }
    }
    Ok(count)
}

fn note(quiet: bool, message: &str) {
    if !quiet {
        eprintln!("{message}");
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib pipeline::`
Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: add the processing pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Importing recordings

**Files:**
- Create: `src/import.rs`
- Modify: `src/lib.rs` (add `pub mod import;`)

**Interfaces:**
- Consumes: `Store`, `Session`, `Meta`, `Conversation`, `Trigger`, `audio::{normalize_to_wav, wav_duration_ms}`, `time::{parse_start, add_ms, truncate_ms}`, `Config`
- Produces: `import::ImportRequest { me: PathBuf, others: PathBuf, start: Option<String>, title: Option<String> }`, `import::import(&Store, &Config, &ImportRequest) -> Result<Session>`, `import::resolve_start(&ImportRequest) -> Result<Zoned>`, `import::start_file_for(&Path) -> Option<PathBuf>`

The start time comes from `--start`, else a sibling `<prefix>.start` file (the `record-call` layout writes a naive local time like `2026-10-08T15:00:12`), else the file's birth time. Modification time is never used: it marks when the recording ended. Until plan 3 adds conversation splitting, an import is one conversation covering the whole recording.

- [ ] **Step 1: Write the failing tests**

`src/import.rs` (tests first):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::write_test_wav;
    use crate::store::{SessionState, TriggerKind};

    fn req(dir: &Path, start: Option<&str>) -> ImportRequest {
        ImportRequest {
            me: dir.join("2026-10-08_1500_sync-me.wav"),
            others: dir.join("2026-10-08_1500_sync-others.wav"),
            start: start.map(String::from),
            title: Some("Sync".into()),
        }
    }

    #[test]
    fn start_file_naming() {
        assert_eq!(
            start_file_for(Path::new("/r/2026-10-08_1500_sync-me.wav")),
            Some(PathBuf::from("/r/2026-10-08_1500_sync.start"))
        );
        assert_eq!(start_file_for(Path::new("/r/me.wav")), None);
    }

    #[test]
    fn start_from_flag_then_start_file() {
        let tmp = tempfile::tempdir().unwrap();
        let r = req(tmp.path(), Some("2026-10-08T15:00:00+02:00[+02:00]"));
        assert_eq!(resolve_start(&r).unwrap().strftime("%H:%M").to_string(), "15:00");

        std::fs::write(tmp.path().join("2026-10-08_1500_sync.start"), "2026-10-08T15:00:12\n").unwrap();
        let r = req(tmp.path(), None);
        assert_eq!(resolve_start(&r).unwrap().datetime().to_string(), "2026-10-08T15:00:12");
    }

    #[test]
    fn import_creates_a_processing_session() {
        let tmp = tempfile::tempdir().unwrap();
        let r = req(tmp.path(), Some("2026-10-08T15:00:00+02:00[+02:00]"));
        write_test_wav(&r.me, 3.0, 1_000);
        write_test_wav(&r.others, 2.0, 1_000);
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let session = import(&store, &Config::default(), &r).unwrap();

        assert_eq!(session.id, "2026-10-08T150000");
        let meta = session.load_meta().unwrap();
        assert_eq!(meta.state, SessionState::Processing);
        assert_eq!(meta.trigger.kind, TriggerKind::Import);
        assert_eq!(meta.title.as_deref(), Some("Sync"));
        assert_eq!(meta.summary.language, "auto");
        assert_eq!(meta.duration_ms(), Some(3_000));
        assert_eq!(meta.conversations.len(), 1);
        assert_eq!((meta.conversations[0].start_ms, meta.conversations[0].end_ms), (0, 3_000));
        assert!(session.wav_path(Track::Me).is_file() && session.wav_path(Track::Others).is_file());
    }

    #[test]
    fn failed_import_leaves_no_session_behind() {
        let tmp = tempfile::tempdir().unwrap();
        let r = req(tmp.path(), Some("2026-10-08T15:00:00+02:00[+02:00]"));
        write_test_wav(&r.me, 1.0, 1_000);
        std::fs::write(&r.others, b"not audio").unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        assert!(import(&store, &Config::default(), &r).is_err());
        assert!(store.list().unwrap().is_empty());
        assert_eq!(std::fs::read_dir(store.sessions_dir()).unwrap().count(), 0);
    }

    #[test]
    fn missing_input_is_reported_before_anything_is_created() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let err = import(&store, &Config::default(), &req(tmp.path(), Some("2026-10-08T15:00:00"))).unwrap_err();
        assert!(format!("{err:#}").contains("does not exist"), "{err:#}");
        assert_eq!(std::fs::read_dir(store.sessions_dir()).unwrap().count(), 0);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib import::`
Expected: compile errors.

- [ ] **Step 3: Implement**

Top of `src/import.rs`:
```rust
//! `kute import`: brings in two recordings made elsewhere (for example by `record-call`).

use crate::audio;
use crate::config::Config;
use crate::store::{Conversation, Meta, Session, SessionState, Store, Track, Trigger, TriggerKind};
use crate::time;
use anyhow::{Context, Result, bail};
use jiff::{Timestamp, Zoned, tz::TimeZone};
use std::fs;
use std::path::{Path, PathBuf};

pub struct ImportRequest {
    pub me: PathBuf,
    pub others: PathBuf,
    pub start: Option<String>,
    pub title: Option<String>,
}

pub fn import(store: &Store, config: &Config, req: &ImportRequest) -> Result<Session> {
    for path in [&req.me, &req.others] {
        if !path.is_file() {
            bail!("{} does not exist", path.display());
        }
    }
    let started_at = resolve_start(req)?;
    let session = store.create_session(&started_at)?;
    match fill_session(&session, config, req, started_at) {
        Ok(()) => Ok(session),
        Err(e) => {
            let _ = fs::remove_dir_all(&session.dir);
            Err(e)
        }
    }
}

fn fill_session(session: &Session, config: &Config, req: &ImportRequest, started_at: Zoned) -> Result<()> {
    audio::normalize_to_wav(&req.me, &session.wav_path(Track::Me))
        .with_context(|| format!("converting {}", req.me.display()))?;
    audio::normalize_to_wav(&req.others, &session.wav_path(Track::Others))
        .with_context(|| format!("converting {}", req.others.display()))?;
    let total = audio::wav_duration_ms(&session.wav_path(Track::Me))?
        .max(audio::wav_duration_ms(&session.wav_path(Track::Others))?);

    let mut meta = Meta::new(
        session.id.clone(),
        started_at.clone(),
        Trigger { kind: TriggerKind::Import, app_name: None, binary: None },
        config.summary.language.clone(),
    );
    meta.title = req.title.clone();
    meta.ended_at = Some(time::add_ms(&started_at, total));
    meta.state = SessionState::Processing;
    if total > 0 {
        meta.conversations.push(Conversation::new(1, 0, total));
    }
    session.save_meta(&meta)
}

/// `--start`, else `<prefix>.start` next to the "me" file, else the file's birth time.
pub fn resolve_start(req: &ImportRequest) -> Result<Zoned> {
    if let Some(s) = &req.start {
        return time::parse_start(s);
    }
    if let Some(path) = start_file_for(&req.me).filter(|p| p.is_file()) {
        let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        return time::parse_start(&text).with_context(|| format!("parsing {}", path.display()));
    }
    let created = fs::metadata(&req.me)
        .and_then(|m| m.created())
        .with_context(|| format!("no start time for {}; pass --start", req.me.display()))?;
    Ok(time::truncate_ms(&Timestamp::try_from(created)?.to_zoned(TimeZone::system())))
}

/// `<prefix>-me.wav` -> `<prefix>.start`.
pub fn start_file_for(me: &Path) -> Option<PathBuf> {
    let name = me.file_name()?.to_str()?;
    let prefix = name.strip_suffix("-me.wav")?;
    Some(me.with_file_name(format!("{prefix}.start")))
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib import::`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/
git commit -m "feat: import recordings made outside kute

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: The `kute` command line

**Files:**
- Create: `src/cli.rs`, `src/main.rs`
- Modify: `src/lib.rs` (add `pub mod cli;`)

**Interfaces:**
- Consumes: all modules above
- Produces: the `kute` binary with `import`, `process`, `list`, `show`, `search`, `title`, `open`, `summarize`, `retranscribe`, `reindex`, `model download`, `model list`, and a global `--config <FILE>`; `cli::run() -> Result<()>`

- [ ] **Step 1: Write the failing test**

`tests/cli.rs`:
```rust
//! Black-box checks of the CLI that need no models.

use std::process::Command;

fn kute(config: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kute")).arg("--config").arg(config).args(args).output().unwrap()
}

fn setup() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp.path().join("config.toml");
    std::fs::write(&config, format!("[storage]\ndir = {:?}\n", tmp.path().join("data"))).unwrap();
    (tmp, config)
}

#[test]
fn help_lists_commands() {
    let out = Command::new(env!("CARGO_BIN_EXE_kute")).arg("--help").output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    for cmd in ["import", "process", "list", "show", "search", "title", "summarize", "retranscribe", "reindex", "model"] {
        assert!(text.contains(cmd), "missing {cmd} in:\n{text}");
    }
}

#[test]
fn empty_archive() {
    let (_tmp, config) = setup();
    let out = kute(&config, &["list"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("no sessions"));
    let out = kute(&config, &["search", "anything"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("no matches"));
    assert!(!kute(&config, &["show", "last"]).status.success());
}

#[test]
fn model_list_marks_missing_models() {
    let (_tmp, config) = setup();
    let out = kute(&config, &["model", "list"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("large-v3-turbo") && text.contains("(default)") && text.contains("silero-v6.2.0"), "{text}");
}

#[test]
fn import_without_models_explains_what_to_do() {
    let (tmp, config) = setup();
    let me = tmp.path().join("me.wav");
    std::fs::copy(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/me.wav"), &me).unwrap();
    let out = kute(&config, &["import", me.to_str().unwrap(), me.to_str().unwrap(), "--start", "2026-10-08T15:00:00"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("kute model download"));
    // Nothing was imported, so nothing is half-processed.
    assert!(String::from_utf8_lossy(&kute(&config, &["list"]).stdout).contains("no sessions"));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --test cli`
Expected: compile error (no `kute` binary yet).

- [ ] **Step 3: Implement**

`src/main.rs`:
```rust
use std::process::ExitCode;

fn main() -> ExitCode {
    match kute::cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kute: {e:#}");
            ExitCode::FAILURE
        }
    }
}
```

`src/cli.rs`:
```rust
//! The `kute` command line.

use crate::config::{Config, SummaryBackend};
use crate::import::{self, ImportRequest};
use crate::index::{HitKind, Index};
use crate::models;
use crate::pipeline::{self, AsrProvider, Pipeline, Report, WhisperProvider};
use crate::render;
use crate::store::{Session, Store, Track};
use crate::summarize;
use crate::time;
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kute", version, about = "Records, transcribes and summarizes your calls, locally")]
struct Cli {
    /// Config file (default: ~/.config/kute/config.toml)
    #[arg(long, global = true, value_name = "FILE")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Import two recordings (your mic, everyone else) and process them
    Import {
        me: PathBuf,
        others: PathBuf,
        /// Start time, e.g. 2026-10-08T15:00:00 (default: <prefix>.start, then file birth time)
        #[arg(long)]
        start: Option<String>,
        #[arg(long)]
        title: Option<String>,
        /// Only import; run `kute process` later
        #[arg(long)]
        no_process: bool,
    },
    /// Finish any outstanding transcription, summaries and compression for a session
    Process { id: String },
    /// List sessions
    List {
        /// Only sessions newer than this, e.g. 7d or 12h
        #[arg(long)]
        since: Option<String>,
    },
    /// Print a session's transcript (or summary)
    Show {
        id: String,
        #[arg(long)]
        summary: bool,
    },
    /// Search all transcripts and summaries
    Search {
        #[arg(required = true)]
        query: Vec<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Set a session's title
    Title {
        id: String,
        #[arg(required = true)]
        title: Vec<String>,
    },
    /// Open a session's folder
    Open { id: String },
    /// Summarize again (all conversations, or one)
    Summarize {
        id: String,
        #[arg(long)]
        conversation: Option<u32>,
        /// Summary language for this session, e.g. it (saved in meta.toml)
        #[arg(long)]
        lang: Option<String>,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Transcribe again from the stored audio
    Retranscribe {
        id: String,
        #[arg(long)]
        model: Option<String>,
    },
    /// Rebuild the search index from the session folders
    Reindex,
    /// Manage Whisper and VAD models
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Download a Whisper model (default: the configured one) and the VAD model
    Download { name: Option<String> },
    /// List known models
    List,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    None,
    Openai,
    Anthropic,
    Command,
}

impl From<BackendArg> for SummaryBackend {
    fn from(b: BackendArg) -> Self {
        match b {
            BackendArg::None => SummaryBackend::None,
            BackendArg::Openai => SummaryBackend::Openai,
            BackendArg::Anthropic => SummaryBackend::Anthropic,
            BackendArg::Command => SummaryBackend::Command,
        }
    }
}

pub fn run() -> Result<()> {
    whisper_rs::install_logging_hooks();
    let cli = Cli::parse();
    let mut config = Config::load(cli.config.as_deref())?;
    let store = Store::open(config.data_dir()?)?;

    match cli.command {
        Command::Import { me, others, start, title, no_process } => {
            // Check the models first, so a missing model doesn't leave a half-done import.
            let provider = if no_process { None } else { Some(WhisperProvider::new(&config, &store)?) };
            let session = import::import(&store, &config, &ImportRequest { me, others, start, title })?;
            println!("imported {}", session.id);
            if let Some(mut provider) = provider {
                run_pipeline(&store, &config, &session, Some(&mut provider), Work::Process)?;
            }
        }
        Command::Process { id } => {
            let session = store.resolve(&id)?;
            let mut provider = WhisperProvider::new(&config, &store)?;
            run_pipeline(&store, &config, &session, Some(&mut provider), Work::Process)?;
        }
        Command::List { since } => list(&store, since.as_deref())?,
        Command::Show { id, summary } => {
            let session = store.resolve(&id)?;
            let (path, what) =
                if summary { (session.summary_path(), "summary") } else { (session.transcript_path(), "transcript") };
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("{} has no {what} yet", session.id))?;
            print!("{text}");
        }
        Command::Search { query, limit } => search(&store, &config, &query.join(" "), limit)?,
        Command::Title { id, title } => {
            let session = store.resolve(&id)?;
            let mut meta = session.load_meta()?;
            meta.title = Some(title.join(" "));
            session.save_meta(&meta)?;
            pipeline::write_documents(&session, &meta, &config.labels)?;
            pipeline::index_session(&Index::open(&store.index_path())?, &session)?;
        }
        Command::Open { id } => {
            let session = store.resolve(&id)?;
            std::process::Command::new("xdg-open").arg(&session.dir).spawn().context("running xdg-open")?;
        }
        Command::Summarize { id, conversation, lang, backend } => {
            let session = store.resolve(&id)?;
            if let Some(lang) = lang {
                let mut meta = session.load_meta()?;
                meta.summary.language = lang;
                session.save_meta(&meta)?;
            }
            if let Some(b) = backend {
                config.summary.backend = b.into();
            }
            run_pipeline(&store, &config, &session, None, Work::Summarize(conversation))?;
        }
        Command::Retranscribe { id, model } => {
            let session = store.resolve(&id)?;
            if let Some(model) = model {
                config.transcribe.model = model;
            }
            let mut provider = WhisperProvider::new(&config, &store)?;
            run_pipeline(&store, &config, &session, Some(&mut provider), Work::Retranscribe)?;
        }
        Command::Reindex => {
            let n = pipeline::reindex_all(&store, &Index::open(&store.index_path())?)?;
            println!("indexed {n} sessions");
        }
        Command::Model { command: ModelCommand::List } => model_list(&store, &config),
        Command::Model { command: ModelCommand::Download { name } } => {
            let name = name.unwrap_or_else(|| config.transcribe.model.clone());
            let info = models::find_whisper(&name).with_context(|| format!("unknown Whisper model {name:?}"))?;
            for info in [info, &models::VAD_MODEL] {
                let path = models::download(&store.models_dir(), info, &mut |done, total| {
                    eprint!("\r{}: {} / {} MB", info.name, done / 1_000_000, total / 1_000_000);
                    let _ = std::io::stderr().flush();
                })?;
                eprintln!();
                println!("{}", path.display());
            }
        }
    }
    Ok(())
}

enum Work {
    Process,
    Retranscribe,
    Summarize(Option<u32>),
}

fn run_pipeline(
    store: &Store,
    config: &Config,
    session: &Session,
    asr: Option<&mut WhisperProvider>,
    work: Work,
) -> Result<()> {
    let index = Index::open(&store.index_path())?;
    let summarizer = summarize::from_config(&config.summary)?;
    let mut pipeline = Pipeline {
        index: &index,
        config,
        asr: asr.map(|p| p as &mut dyn AsrProvider),
        summarizer: summarizer.as_deref(),
        template: summarize::load_template(&config.summary)?,
        quiet: false,
    };
    let report = match work {
        Work::Process => pipeline.process(session)?,
        Work::Retranscribe => pipeline.retranscribe(session)?,
        Work::Summarize(only) => pipeline.resummarize(session, only)?,
    };
    print_report(session, &report)
}

fn print_report(session: &Session, report: &Report) -> Result<()> {
    if session.transcript_path().is_file() {
        println!("transcript: {}", session.transcript_path().display());
    }
    if session.summary_path().is_file() {
        println!("summary:    {}", session.summary_path().display());
    }
    if let Some(e) = &report.compression_error {
        eprintln!("kute: compression failed (audio kept as WAV): {e}");
    }
    if !report.failed.is_empty() {
        for (n, e) in &report.failed {
            eprintln!("kute: conversation {n} failed: {e}");
        }
        bail!("{} conversation(s) failed; fix the cause and run `kute process {}`", report.failed.len(), session.id);
    }
    Ok(())
}

fn list(store: &Store, since: Option<&str>) -> Result<()> {
    let cutoff = match since {
        Some(s) => {
            let span: jiff::Span = s.parse().with_context(|| format!("cannot read {s:?} as a duration (try 7d)"))?;
            Some(time::now().checked_sub(span)?)
        }
        None => None,
    };
    let mut shown = 0;
    for session in store.list()? {
        let meta = match session.load_meta() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("kute: skipping {}: {e:#}", session.id);
                continue;
            }
        };
        if cutoff.as_ref().is_some_and(|c| meta.started_at < *c) {
            continue;
        }
        let state = format!("{:?}", meta.state).to_lowercase();
        println!(
            "{:<20} {:<28} {:>11} {:<10} {:>3} conv  {state}",
            meta.id,
            meta.title.as_deref().unwrap_or("-"),
            meta.duration_ms().map(time::format_duration).unwrap_or_else(|| "-".into()),
            meta.app_label(),
            meta.conversations.len(),
        );
        shown += 1;
    }
    if shown == 0 {
        println!("no sessions");
    }
    Ok(())
}

fn search(store: &Store, config: &Config, query: &str, limit: usize) -> Result<()> {
    let hits = Index::open(&store.index_path())?.search(query, limit)?;
    if hits.is_empty() {
        println!("no matches");
    }
    for hit in hits {
        match hit.kind {
            HitKind::Line => {
                let at = hit
                    .started_at
                    .parse::<jiff::Zoned>()
                    .map(|z| time::add_ms(&z, hit.t_ms.unwrap_or(0)).strftime("%H:%M:%S").to_string())
                    .unwrap_or_default();
                let track = if hit.track.as_deref() == Some("me") { Track::Me } else { Track::Others };
                println!("{}  {at}  {}: {}", hit.session_id, render::label(&config.labels, track), hit.text);
            }
            HitKind::Summary => {
                println!("{}  conversation {} summary: {}", hit.session_id, hit.conversation, hit.text.replace('\n', " "));
            }
        }
    }
    Ok(())
}

fn model_list(store: &Store, config: &Config) {
    for info in models::WHISPER_MODELS.iter().chain([&models::VAD_MODEL]) {
        let present = if models::path_for(&store.models_dir(), info).is_file() { "downloaded" } else { "-" };
        let default = if info.name == config.transcribe.model { " (default)" } else { "" };
        println!("{:<22} {:>6} MB  {present}{default}", info.name, info.size / 1_000_000);
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --test cli && cargo test`
Expected: 4 CLI tests PASS; the whole suite PASSES (ignored tests stay ignored).

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/ tests/cli.rs
git commit -m "feat: add the kute command line

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: End-to-end run with real models

**Files:**
- Create: `tests/e2e.rs`

**Interfaces:**
- Consumes: the `kute` binary, `tests/fixtures/{me,others}.wav`

- [ ] **Step 1: Write the test**

`tests/e2e.rs`:
```rust
//! Full import -> transcript -> summary -> Opus -> search with the tiny Whisper model.
//! Needs KUTE_TEST_MODELS with ggml-tiny.bin and ggml-silero-v6.2.0.bin:
//!   KUTE_TEST_MODELS=/path/to/models cargo test --test e2e -- --ignored

use std::path::Path;
use std::process::Command;

#[test]
#[ignore]
fn import_transcribes_summarizes_and_compresses() {
    let models = std::env::var("KUTE_TEST_MODELS").expect("set KUTE_TEST_MODELS");
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(data.join("models")).unwrap();
    for file in ["ggml-tiny.bin", "ggml-silero-v6.2.0.bin"] {
        std::os::unix::fs::symlink(Path::new(&models).join(file), data.join("models").join(file)).unwrap();
    }
    let config = tmp.path().join("config.toml");
    std::fs::write(
        &config,
        format!(
            r###"
[storage]
dir = {data:?}
[transcribe]
model = "tiny"
gpu = false
[summary]
backend = "command"
command = ["sh", "-c", 'cat > /dev/null; printf "## Summary\n\nFixture summary.\n"']
min_words = 1
"###
        ),
    )
    .unwrap();

    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let kute = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_kute")).arg("--config").arg(&config).args(args).output().unwrap();
        assert!(out.status.success(), "kute {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    kute(&[
        "import",
        fixtures.join("me.wav").to_str().unwrap(),
        fixtures.join("others.wav").to_str().unwrap(),
        "--start",
        "2026-10-08T15:00:00",
        "--title",
        "Fixture call",
    ]);

    let transcript = kute(&["show", "last"]);
    assert!(transcript.starts_with("# Fixture call. 2026-10-08 15:00"), "{transcript}");
    assert!(transcript.to_lowercase().contains("differential"), "{transcript}");
    assert!(transcript.contains("Others: ") && transcript.to_lowercase().contains("ciao"), "{transcript}");
    let me_line = transcript.lines().position(|l| l.contains("Me: ")).unwrap();
    let others_line = transcript.lines().position(|l| l.contains("Others: ")).unwrap();
    assert!(me_line < others_line, "English (0-5 s) should come before Italian (6 s+)");

    assert!(kute(&["show", "last", "--summary"]).contains("Fixture summary."));
    assert!(kute(&["search", "differential"]).contains("2026-10-08T150000"));
    assert!(kute(&["list"]).contains("Fixture call"));

    let session = data.join("sessions/2026-10-08T150000");
    assert!(session.join("me.opus").is_file() && !session.join("me.wav").exists());
    let meta = std::fs::read_to_string(session.join("meta.toml")).unwrap();
    assert!(meta.contains("state = \"done\""), "{meta}");
    assert!(meta.contains("me = \"en\"") && meta.contains("others = \"it\""), "{meta}");

    // Re-transcribing from Opus works and keeps the result.
    kute(&["retranscribe", "last"]);
    assert!(kute(&["show", "last"]).to_lowercase().contains("differential"));
}
```

- [ ] **Step 2: Run it**

```bash
KUTE_TEST_MODELS=/tmp/kute-models cargo test --test e2e -- --ignored
```
Expected: PASS. (Models were fetched in Task 7 Step 5; `kute model download tiny` into a scratch data dir also works.)

If the tiny model mishears "differential" or "ciao" on the synthetic voice, check the transcript it printed; do not loosen the assertion to something that would pass on an empty transcript. Re-run with `ggml-base.bin` in place of `ggml-tiny.bin` (same `tiny` name in the config is fine for the symlink) before changing any wording.

- [ ] **Step 3: Try it by hand on a real-model setup**

```bash
cargo run --release -- model download
cargo run --release -- import tests/fixtures/me.wav tests/fixtures/others.wav --start 2026-10-08T15:00:00 --title "Smoke test"
cargo run --release -- show last
cargo run --release -- list
```
Expected: the transcript prints with `[15:00:0x] Me: Hello everyone...` and `[15:00:0x] Others: Ciao a tutti...` lines (large-v3-turbo, Vulkan).

- [ ] **Step 4: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add tests/e2e.rs
git commit -m "test: add end-to-end import test with the tiny model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
