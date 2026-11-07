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
use crate::transcribe::{SpeechDetector, Transcriber, session_languages, transcribe_track};
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
        let (model_path, model_name) =
            models::resolve_whisper(&store.models_dir(), &config.transcribe.model)?;
        let vad_path = models::resolve_vad(&store.models_dir())?;
        Ok(WhisperProvider {
            model_path,
            model_name,
            vad_path,
            gpu: config.transcribe.gpu,
            loaded: None,
        })
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
    /// Works on the conversations in `meta.toml` when it starts. A session that is still
    /// recording is fine: its listed conversations are closed (the open one is only
    /// `open_conversation_ms`), and the live session may append more while this runs.
    pub fn process(&mut self, session: &Session) -> Result<Report> {
        let mut meta = session.load_meta()?;
        remove_temp_files(session)?;
        let mut report = Report::default();
        let ns: Vec<u32> = meta.conversations.iter().map(|c| c.n).collect();

        if self.asr.is_some() {
            for &n in &ns {
                if !meta
                    .conversation(n)
                    .is_some_and(|c| c.needs_transcription())
                {
                    continue;
                }
                note(
                    self.quiet,
                    &format!("{}: transcribing conversation {n}", meta.id),
                );
                match self.transcribe_conversation(session, &mut meta, n) {
                    Ok(()) => report.transcribed.push(n),
                    Err(e) => {
                        meta.conversation_mut(n)
                            .expect("n comes from meta")
                            .fail(Step::Transcribe, &e);
                        meta = save_conversation(session, &meta, n)?;
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
                    meta.conversation_mut(n)
                        .expect("n comes from meta")
                        .fail(Step::Summarize, &e);
                    meta = save_conversation(session, &meta, n)?;
                    report.failed.push((n, format!("{e:#}")));
                }
            }
        }

        self.compress_if_ready(session, &mut meta, &mut report)?;
        write_documents(session, &meta, &self.config.labels)?;
        index_session(self.index, session)?;
        Ok(report)
    }

    /// Transcribes and summarizes every conversation again from the stored audio (Opus
    /// after compression).
    pub fn retranscribe(&mut self, session: &Session) -> Result<Report> {
        let meta = session.load_meta()?;
        if meta.state == SessionState::Recording {
            bail!("session {} is still recording", meta.id);
        }
        // Load the models first: if that fails, nothing has been touched. Transcripts and
        // summaries are then replaced conversation by conversation, never wiped up front.
        if let Some(asr) = self.asr.as_deref_mut() {
            asr.get().context("loading the transcription models")?;
        }
        session.update_meta(|m| {
            for c in &mut m.conversations {
                let (n, start_ms, end_ms) = (c.n, c.start_ms, c.end_ms);
                *c = crate::store::Conversation::new(n, start_ms, end_ms);
            }
        })?;
        self.process(session)
    }

    /// Summarizes again: every transcribed conversation, or only `only`.
    pub fn resummarize(&mut self, session: &Session, only: Option<u32>) -> Result<Report> {
        if self.summarizer.is_none() {
            bail!(
                "no summarizer is configured (summary.backend = \"none\"); set one in the config or pass --backend"
            );
        }
        let meta = session.load_meta()?;
        if let Some(n) = only {
            let c = meta
                .conversation(n)
                .with_context(|| format!("session {} has no conversation {n}", meta.id))?;
            if c.needs_transcription() {
                bail!(
                    "conversation {n} has not been transcribed yet; run `kute process {}`",
                    meta.id
                );
            }
        }
        session.update_meta(|m| {
            for c in &mut m.conversations {
                if only.is_some_and(|n| n != c.n) || c.needs_transcription() {
                    continue;
                }
                c.stage = Stage::Transcribed;
                c.clear_error();
            }
        })?;
        self.process(session)
    }

    fn transcribe_conversation(
        &mut self,
        session: &Session,
        meta: &mut Meta,
        n: u32,
    ) -> Result<()> {
        let conv = meta.conversation(n).expect("n comes from meta").clone();
        let provider = self
            .asr
            .as_deref_mut()
            .context("no transcriber available")?;
        let (vad, asr) = provider.get()?;
        let model = asr.model_name().to_string();
        let mut results = Vec::new();
        for track in Track::BOTH {
            let offset = meta.tracks.get(track).offset_ms;
            let base_ms = conv.start_ms.max(offset);
            // Track time of the first sample we need, and how much audio there is from there.
            let first = base_ms - offset;
            let source = if conv.end_ms > base_ms {
                Some(AudioSource::open(session, track)?)
            } else {
                None
            };
            let total_ms = match &source {
                Some(s) => (conv.end_ms - offset)
                    .min(audio::wav_duration_ms(s.path())?)
                    .saturating_sub(first),
                None => 0,
            };
            let languages = session_languages(
                meta.transcribe.for_track(track),
                &self.config.transcribe.language,
            )?;
            let mut read = |a: u64, b: u64| match &source {
                Some(s) => audio::read_wav_range(s.path(), first + a, first + b),
                None => Ok(Vec::new()),
            };
            let result = transcribe_track(
                &mut read, total_ms, base_ms, n, &languages, &mut *vad, &mut *asr,
            )
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
        *meta = save_conversation(session, meta, n)?;
        write_documents(session, meta, &self.config.labels)?;
        index_session(self.index, session)
    }

    /// Returns `Ok(false)` when the conversation is skipped (no summarizer, or too short).
    fn summarize_conversation(
        &mut self,
        session: &Session,
        meta: &mut Meta,
        n: u32,
    ) -> Result<bool> {
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
                *meta = save_conversation(session, meta, n)?;
                return Ok(false);
            }
        };

        note(
            self.quiet,
            &format!("{}: summarizing conversation {n}", meta.id),
        );
        let labels = &meta.labels(&self.config.labels);
        let ctx = PromptContext {
            date: meta.started_at.strftime("%Y-%m-%d").to_string(),
            time_range: render::time_range(meta, &conv),
            app: meta.app_label(),
            title: meta.title.as_deref(),
            notes: meta.notes.as_deref(),
            user_name: &self.config.user.name,
            me_label: &labels.me,
            others_label: &labels.others,
            language: &meta.summary.language,
        };
        let prompt = build_prompt(
            &self.template,
            &ctx,
            &render::render_lines(meta, labels, &lines),
        );
        let text = summarizer.summarize(&prompt)?;

        fs::create_dir_all(session.summaries_dir())?;
        write_atomic(&summary_path, format!("{}\n", text.trim()).as_bytes())?;
        let c = meta.conversation_mut(n).expect("n comes from meta");
        c.stage = Stage::Summarized;
        c.summarizer = Some(summarizer.name());
        c.clear_error();
        *meta = save_conversation(session, meta, n)?;
        write_documents(session, meta, labels)?;
        index_session(self.index, session)?;
        Ok(true)
    }

    fn compress_if_ready(
        &mut self,
        session: &Session,
        meta: &mut Meta,
        report: &mut Report,
    ) -> Result<()> {
        // The recording may have ended, or added conversations, since `meta` was read.
        *meta = session.load_meta()?;
        if meta.compression.stage == CompressionStage::Done
            || meta.ended_at.is_none()
            || !meta.conversations.iter().all(|c| c.is_settled())
        {
            return Ok(());
        }
        note(self.quiet, &format!("{}: compressing audio", meta.id));
        let result = compress_tracks(session);
        *meta = session.update_meta(|m| match &result {
            Ok(()) => {
                m.compression.stage = CompressionStage::Done;
                m.compression.error = None;
                m.state = SessionState::Done;
            }
            Err(e) => {
                m.compression.stage = CompressionStage::Failed;
                m.compression.error = Some(format!("{e:#}"));
            }
        })?;
        match result {
            Ok(()) => report.compressed = true,
            Err(e) => report.compression_error = Some(format!("{e:#}")),
        }
        Ok(())
    }
}

/// Writes conversation `n` from `meta` into `meta.toml` and leaves everything else as it is on
/// disk, so conversations the live session appended meanwhile survive. Returns the saved meta.
fn save_conversation(session: &Session, meta: &Meta, n: u32) -> Result<Meta> {
    let conv = meta.conversation(n).expect("n comes from meta").clone();
    session.update_meta(|m| {
        if let Some(c) = m.conversation_mut(n) {
            *c = conv;
        }
    })
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
            bail!(
                "{} track: Opus is {opus_ms} ms but the WAV is {wav_ms} ms",
                track.as_str()
            );
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
            return Ok(AudioSource {
                path: wav,
                temporary: false,
            });
        }
        let opus = session.opus_path(track);
        if opus.is_file() {
            let path = session.dir.join(format!(".decoded-{}.wav", track.as_str()));
            audio::normalize_to_wav(&opus, &path)?;
            return Ok(AudioSource {
                path,
                temporary: true,
            });
        }
        Err(anyhow!(
            "session {} has no audio for the {} track",
            session.id,
            track.as_str()
        ))
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
    let labels = &meta.labels(labels);
    let me = session.load_track(Track::Me)?;
    let others = session.load_track(Track::Others)?;
    write_atomic(
        &session.transcript_path(),
        render::render_transcript(meta, labels, &me, &others).as_bytes(),
    )?;

    let mut summaries = Vec::new();
    for c in meta
        .conversations
        .iter()
        .filter(|c| c.stage == Stage::Summarized)
    {
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
    for c in meta
        .conversations
        .iter()
        .filter(|c| !c.needs_transcription())
    {
        index.replace_lines(&meta.id, c.n, &render::conversation_lines(c, &me, &others))?;
        if c.stage == Stage::Summarized
            && let Ok(text) = fs::read_to_string(session.conversation_summary_path(c.n))
        {
            index.replace_summary(&meta.id, c.n, Some(&text))?;
        }
    }
    Ok(())
}

/// Rebuilds the index from every session folder. A broken session is reported on stderr
/// and skipped, so one bad folder doesn't block search for the rest.
/// Removes the temp files an interrupted run leaves in the session folder: decoded audio
/// (`AudioSource`, can be ~1 GB) and half-written Opus (`compress_tracks`).
fn remove_temp_files(session: &Session) -> Result<()> {
    for entry in fs::read_dir(&session.dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let temp = name.starts_with(".decoded-") && name.ends_with(".wav")
            || name.starts_with('.') && name.ends_with(".opus.tmp");
        if temp && entry.file_type()?.is_file() {
            fs::remove_file(entry.path())
                .with_context(|| format!("removing {}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// Opens the search index, rebuilding it from the session folders when it was missing or made
/// by another version of kute.
pub fn open_index(store: &Store) -> Result<Index> {
    let index = Index::open(&store.index_path())?;
    if index.recreated() {
        reindex_all(store, &index)?;
    }
    Ok(index)
}

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
                Ok(vec![Span {
                    start_ms: 0,
                    end_ms: crate::audio::samples_to_ms(samples.len()),
                }])
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
        fn detect_language(&mut self, _: &[f32], _: &[String]) -> Result<String> {
            Ok("en".into())
        }
        fn transcribe(&mut self, samples: &[f32], _: &str) -> Result<Vec<RawSegment>> {
            let level = (samples[0] * 100.0).round() as i32;
            let text = format!(" level {level} words here");
            Ok(vec![RawSegment {
                start_ms: 0,
                end_ms: 500,
                text: text.clone(),
                words: vec![Word {
                    start_ms: 0,
                    end_ms: 500,
                    text,
                }],
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
            FakeProvider {
                vad: LevelVad,
                asr: LevelAsr,
                released: 0,
                fail: false,
            }
        }
    }
    impl AsrProvider for FakeProvider {
        fn get(&mut self) -> Result<(&mut dyn SpeechDetector, &mut dyn Transcriber)> {
            if self.fail {
                anyhow::bail!("GPU out of memory");
            }
            Ok((
                &mut self.vad as &mut dyn SpeechDetector,
                &mut self.asr as &mut dyn Transcriber,
            ))
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
            Trigger {
                kind: TriggerKind::Import,
                app_name: None,
                binary: None,
            },
            "en".into(),
        );
        meta.state = SessionState::Processing;
        meta.ended_at = Some(crate::time::add_ms(&start, conv_ms));
        meta.conversations.push(Conversation::new(1, 0, conv_ms));
        session.save_meta(&meta).unwrap();
        Fixture {
            _tmp: tmp,
            store,
            index,
            config,
            session,
        }
    }

    fn pipeline<'a>(
        f: &'a Fixture,
        asr: &'a mut FakeProvider,
        summarizer: Option<&'a dyn Summarizer>,
    ) -> Pipeline<'a> {
        Pipeline {
            index: &f.index,
            config: &f.config,
            asr: Some(asr),
            summarizer,
            template: crate::summarize::DEFAULT_PROMPT.to_string(),
            quiet: true,
        }
    }

    fn summarizer(fail: bool) -> FakeSummarizer {
        FakeSummarizer {
            prompts: RefCell::new(Vec::new()),
            fail,
        }
    }

    #[test]
    fn each_track_uses_its_own_languages() {
        let f = fixture(2.0, 2.0, 2_000);
        f.session
            .update_meta(|m| m.transcribe.me = vec!["ru".into()])
            .unwrap();
        let mut asr = FakeProvider::new();
        pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        let languages = &f.session.load_meta().unwrap().conversations[0].languages;
        // "ru" is forced on the me track; the others track is detected ("en" from the fake).
        assert_eq!(languages.me.as_deref(), Some("ru"));
        assert_eq!(languages.others.as_deref(), Some("en"));
    }

    #[test]
    fn full_run_transcribes_summarizes_compresses_and_indexes() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        let sum = summarizer(false);
        let report = pipeline(&f, &mut asr, Some(&sum))
            .process(&f.session)
            .unwrap();
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
        assert!(
            transcript.contains("[09:00:00] Me: level 25 words here"),
            "{transcript}"
        );
        assert!(
            transcript.contains("[09:00:00] Others: level 50 words here"),
            "{transcript}"
        );
        assert!(
            std::fs::read_to_string(f.session.summary_path())
                .unwrap()
                .contains("It went fine.")
        );
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
        assert!(
            transcript.contains("Me: level 25") && !transcript.contains("Others:"),
            "{transcript}"
        );
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
        assert!(
            transcript.contains("[09:00:01] Others: level 50"),
            "{transcript}"
        );
    }

    #[test]
    fn transcription_failure_is_recorded_and_resumable() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider {
            fail: true,
            ..FakeProvider::new()
        };
        let report = pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        assert_eq!(report.failed.len(), 1);
        let meta = f.session.load_meta().unwrap();
        let c = &meta.conversations[0];
        assert_eq!(
            (c.stage, c.failed_step, c.attempts),
            (Stage::Failed, Some(Step::Transcribe), 1)
        );
        assert!(c.error.as_deref().unwrap().contains("GPU out of memory"));
        // The WAVs were kept for this retry (see transcription_failure_keeps_wavs_for_the_retry).
        let mut asr = FakeProvider::new();
        let report = pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        assert_eq!(report.transcribed, vec![1]);
        assert!(
            f.session.load_meta().unwrap().conversations[0]
                .error
                .is_none()
        );
    }

    #[test]
    fn summary_failure_keeps_transcript_and_retries() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        let bad = summarizer(true);
        let report = pipeline(&f, &mut asr, Some(&bad))
            .process(&f.session)
            .unwrap();
        assert_eq!(report.failed[0].0, 1);
        assert!(f.session.transcript_path().is_file());
        let good = summarizer(false);
        let report = pipeline(&f, &mut asr, Some(&good))
            .process(&f.session)
            .unwrap();
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
        assert!(
            f.session.wav_path(Track::Me).is_file() && f.session.wav_path(Track::Others).is_file()
        );

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
        let report = pipeline(&f, &mut asr, None)
            .retranscribe(&f.session)
            .unwrap();
        assert_eq!(report.transcribed, vec![1]);
        assert!(
            std::fs::read_to_string(f.session.transcript_path())
                .unwrap()
                .contains("Me: level")
        );
        // Decoded temp files are cleaned up.
        assert!(!f.session.dir.join(".decoded-me.wav").exists());
    }

    #[test]
    fn resummarize_only_one_conversation_and_recording_is_refused() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        let sum = summarizer(false);
        pipeline(&f, &mut asr, Some(&sum))
            .process(&f.session)
            .unwrap();
        let report = pipeline(&f, &mut asr, Some(&sum))
            .resummarize(&f.session, Some(1))
            .unwrap();
        assert_eq!(report.summarized, vec![1]);
        assert!(
            pipeline(&f, &mut asr, Some(&sum))
                .resummarize(&f.session, Some(7))
                .is_err()
        );

        f.session
            .update_meta(|m| m.state = SessionState::Recording)
            .unwrap();
        assert!(
            pipeline(&f, &mut asr, None)
                .retranscribe(&f.session)
                .is_err()
        );
        assert!(pipeline(&f, &mut asr, None).process(&f.session).is_ok());
    }

    #[test]
    fn process_handles_closed_conversations_while_recording() {
        let f = fixture(2.0, 2.0, 2_000);
        f.session
            .update_meta(|m| {
                m.state = SessionState::Recording;
                m.ended_at = None;
                m.open_conversation_ms = Some(1_500);
            })
            .unwrap();
        let mut asr = FakeProvider::new();
        let sum = summarizer(false);
        let report = pipeline(&f, &mut asr, Some(&sum))
            .process(&f.session)
            .unwrap();
        assert_eq!(report.transcribed, vec![1]);
        assert!(!report.compressed);
        let meta = f.session.load_meta().unwrap();
        assert_eq!(meta.state, SessionState::Recording);
        assert_eq!(meta.open_conversation_ms, Some(1_500));
        assert!(f.session.wav_path(Track::Me).is_file());
    }

    /// Appends conversation 2 while summarizing, the way the live session does.
    struct AppendingSummarizer<'a> {
        session: &'a Session,
    }
    impl Summarizer for AppendingSummarizer<'_> {
        fn name(&self) -> String {
            "appending".into()
        }
        fn summarize(&self, _: &str) -> Result<String> {
            self.session
                .update_meta(|m| m.conversations.push(Conversation::new(2, 1_000, 2_000)))?;
            Ok("It went fine.".into())
        }
    }

    #[test]
    fn process_keeps_conversations_added_while_it_runs() {
        let f = fixture(2.0, 2.0, 2_000);
        let sum = AppendingSummarizer {
            session: &f.session,
        };
        let mut asr = FakeProvider::new();
        let report = pipeline(&f, &mut asr, Some(&sum))
            .process(&f.session)
            .unwrap();
        // Conversation 2 still needs work, so the WAVs stay.
        assert!(!report.compressed);
        assert!(f.session.wav_path(Track::Me).is_file());
        let meta = f.session.load_meta().unwrap();
        assert_eq!(meta.conversations.len(), 2);
        assert_eq!(meta.conversation(1).unwrap().stage, Stage::Summarized);
        assert_eq!(meta.conversation(2).unwrap().stage, Stage::Pending);
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

    #[test]
    fn resummarize_without_summarizer_refuses_and_keeps_summaries() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        let sum = summarizer(false);
        pipeline(&f, &mut asr, Some(&sum))
            .process(&f.session)
            .unwrap();
        let err = pipeline(&f, &mut asr, None)
            .resummarize(&f.session, None)
            .unwrap_err();
        assert!(format!("{err:#}").contains("no summarizer"), "{err:#}");
        assert!(f.session.conversation_summary_path(1).is_file());
        assert!(f.session.summary_path().is_file());
        assert_eq!(
            f.session.load_meta().unwrap().conversations[0].stage,
            Stage::Summarized
        );
    }

    #[test]
    fn transcription_failure_keeps_wavs_for_the_retry() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider {
            fail: true,
            ..FakeProvider::new()
        };
        let report = pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        assert!(!report.compressed);
        let meta = f.session.load_meta().unwrap();
        assert_eq!(meta.compression.stage, CompressionStage::Pending);
        for t in Track::BOTH {
            assert!(f.session.wav_path(t).is_file());
            assert!(!f.session.opus_path(t).exists());
        }
    }

    #[test]
    fn retranscribe_with_failing_models_keeps_everything() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        let sum = summarizer(false);
        pipeline(&f, &mut asr, Some(&sum))
            .process(&f.session)
            .unwrap();
        let before = std::fs::read_to_string(f.session.transcript_path()).unwrap();
        let mut broken = FakeProvider {
            fail: true,
            ..FakeProvider::new()
        };
        assert!(
            pipeline(&f, &mut broken, Some(&sum))
                .retranscribe(&f.session)
                .is_err()
        );
        let meta = f.session.load_meta().unwrap();
        assert_eq!(meta.conversations[0].stage, Stage::Summarized);
        assert_eq!(
            std::fs::read_to_string(f.session.transcript_path()).unwrap(),
            before
        );
        assert!(f.session.conversation_summary_path(1).is_file());
        assert!(!f.session.load_track(Track::Me).unwrap().segments.is_empty());
    }

    #[test]
    fn process_removes_temp_files_left_by_an_interrupted_run() {
        let f = fixture(2.0, 2.0, 2_000);
        let stale = [".decoded-me.wav", ".others.opus.tmp"];
        for name in stale {
            std::fs::write(f.session.dir.join(name), b"partial").unwrap();
        }
        let mut asr = FakeProvider::new();
        pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        for name in stale {
            assert!(!f.session.dir.join(name).exists(), "{name} survived");
        }
    }

    #[test]
    fn open_index_rebuilds_a_missing_or_outdated_index() {
        let f = fixture(2.0, 2.0, 2_000);
        let mut asr = FakeProvider::new();
        pipeline(&f, &mut asr, None).process(&f.session).unwrap();
        drop(f.index);
        std::fs::remove_file(f.store.index_path()).unwrap();

        let index = open_index(&f.store).unwrap();
        assert_eq!(index.search("level", 10).unwrap().len(), 2);
    }
}
