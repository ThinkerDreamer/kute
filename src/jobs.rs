//! Jobs: the processing work the CLI runs in the foreground and the daemon's worker runs
//! in the background.

use crate::config::{Config, SummaryBackend};
use crate::import::{self, ImportRequest};
use crate::index::Index;
use crate::pipeline::{self, AsrProvider, Pipeline, Report, WhisperProvider};
use crate::store::{LanguageChange, Meta, Session, Stage, Step, Store};
use crate::summarize;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "job", rename_all = "snake_case")]
pub enum Job {
    Process {
        id: String,
    },
    Retranscribe {
        id: String,
        model: Option<String>,
        /// Applied to the session's languages first.
        #[serde(default)]
        languages: LanguageChange,
    },
    Summarize {
        id: String,
        conversation: Option<u32>,
        lang: Option<String>,
        backend: Option<SummaryBackend>,
    },
    /// `me` and `others` are used as given; the CLI makes them absolute.
    Import {
        me: PathBuf,
        others: PathBuf,
        start: Option<String>,
        title: Option<String>,
        #[serde(default)]
        languages: LanguageChange,
        process: bool,
    },
    Title {
        id: String,
        title: String,
    },
    /// `None` removes the notes.
    Notes {
        id: String,
        notes: Option<String>,
    },
    /// The languages spoken in a session.
    Language {
        id: String,
        languages: LanguageChange,
    },
    Reindex,
}

impl Job {
    /// The session the job works on; `None` for an import (its session doesn't exist yet)
    /// and a reindex.
    pub fn session(&self) -> Option<&str> {
        match self {
            Job::Process { id }
            | Job::Retranscribe { id, .. }
            | Job::Summarize { id, .. }
            | Job::Title { id, .. }
            | Job::Notes { id, .. }
            | Job::Language { id, .. } => Some(id),
            Job::Import { .. } | Job::Reindex => None,
        }
    }
}

/// A short description for logs and `kute status`, e.g. "process 2026-10-08T090312".
impl std::fmt::Display for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Job::Process { id } => write!(f, "process {id}"),
            Job::Retranscribe { id, .. } => write!(f, "retranscribe {id}"),
            Job::Summarize { id, .. } => write!(f, "summarize {id}"),
            Job::Import { me, .. } => write!(f, "import {}", me.display()),
            Job::Title { id, .. } => write!(f, "title {id}"),
            Job::Notes { id, .. } => write!(f, "notes {id}"),
            Job::Language { id, .. } => write!(f, "language {id}"),
            Job::Reindex => f.write_str("reindex"),
        }
    }
}

#[derive(Debug, Default)]
pub struct Outcome {
    /// What the CLI prints on stdout; may be empty.
    pub message: String,
    pub session: Option<String>,
    /// Conversations this job wrote a summary for (the daemon announces them).
    pub summarized: Vec<u32>,
}

/// The error a job returns when the pipeline left conversations failed. It carries the
/// session, so the worker can schedule a transcription retry even for an `Import`.
#[derive(Debug)]
pub struct ConversationsFailed {
    pub session: String,
    pub count: usize,
    /// Where the transcript and summary of the rest are, as `kute process` prints them on
    /// success. Left to the caller, since the daemon's stdout goes nowhere.
    pub written: Vec<String>,
}

impl std::fmt::Display for ConversationsFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} conversation(s) failed; fix the cause and run `kute process {}`",
            self.count, self.session
        )
    }
}

impl std::error::Error for ConversationsFailed {}

/// Runs one job. `asr` holds the Whisper provider between jobs; it is built here on the first
/// job that transcribes with the configured model.
pub fn run(
    store: &Store,
    config: &Config,
    asr: &mut Option<WhisperProvider>,
    job: &Job,
) -> Result<Outcome> {
    match job {
        Job::Process { id } => {
            let session = store.resolve(id)?;
            let provider = provider(asr, config, store)?;
            let done = run_pipeline(store, config, &session, Some(provider), Work::Process)?;
            Ok(outcome(done, &session))
        }
        Job::Retranscribe {
            id,
            model,
            languages,
        } => {
            let session = store.resolve(id)?;
            if !languages.is_empty() {
                session.update_meta(|m| languages.apply(&mut m.transcribe))?;
            }
            let done = match model {
                Some(model) => {
                    // Another model for this job only, so it doesn't replace the kept provider.
                    let mut config = config.clone();
                    config.transcribe.model = model.clone();
                    let mut provider = WhisperProvider::new(&config, store)?;
                    run_pipeline(
                        store,
                        &config,
                        &session,
                        Some(&mut provider),
                        Work::Retranscribe,
                    )?
                }
                None => {
                    let provider = provider(asr, config, store)?;
                    run_pipeline(store, config, &session, Some(provider), Work::Retranscribe)?
                }
            };
            Ok(outcome(done, &session))
        }
        Job::Summarize {
            id,
            conversation,
            lang,
            backend,
        } => {
            let session = store.resolve(id)?;
            if let Some(lang) = lang {
                session.update_meta(|m| m.summary.language = lang.clone())?;
            }
            let mut config = config.clone();
            if let Some(b) = backend {
                config.summary.backend = *b;
            }
            let done = run_pipeline(
                store,
                &config,
                &session,
                None,
                Work::Summarize(*conversation),
            )?;
            Ok(outcome(done, &session))
        }
        Job::Import {
            me,
            others,
            start,
            title,
            languages,
            process,
        } => {
            // Check the models first, so a missing model doesn't leave a half-done import.
            if *process {
                provider(asr, config, store)?;
            }
            let session = import::import(
                store,
                config,
                &ImportRequest {
                    me: me.clone(),
                    others: others.clone(),
                    start: start.clone(),
                    title: title.clone(),
                },
            )?;
            if !languages.is_empty() {
                session.update_meta(|m| languages.apply(&mut m.transcribe))?;
            }
            let mut done = Done {
                message: format!("imported {}", session.id),
                summarized: Vec::new(),
            };
            if *process {
                let provider = provider(asr, config, store)?;
                // The session exists now, so a processing error must still say which one.
                let processed =
                    run_pipeline(store, config, &session, Some(provider), Work::Process)
                        .with_context(|| format!("imported {}", session.id))?;
                if !processed.message.is_empty() {
                    done.message = format!("{}\n{}", done.message, processed.message);
                }
                done.summarized = processed.summarized;
            }
            Ok(outcome(done, &session))
        }
        Job::Title { id, title } => {
            let session = store.resolve(id)?;
            let meta = session.update_meta(|m| m.title = Some(title.clone()))?;
            pipeline::write_documents(&session, &meta, &config.labels)?;
            pipeline::index_session(&pipeline::open_index(store)?, &session)?;
            Ok(outcome(Done::default(), &session))
        }
        Job::Notes { id, notes } => {
            // Only the summarizer reads the notes, so there are no documents to rewrite.
            let session = store.resolve(id)?;
            session.update_meta(|m| m.notes = notes.clone())?;
            Ok(outcome(Done::default(), &session))
        }
        Job::Language { id, languages } => {
            // Conversations transcribed from now on use them; earlier ones keep their text
            // until they are transcribed again.
            let session = store.resolve(id)?;
            let meta = session.update_meta(|m| languages.apply(&mut m.transcribe))?;
            let transcribed = meta
                .conversations
                .iter()
                .filter(|c| !c.needs_transcription())
                .count();
            let message = if transcribed > 0 {
                format!(
                    "{transcribed} conversation(s) already transcribed; run `kute retranscribe {id}` to redo them"
                )
            } else {
                String::new()
            };
            Ok(outcome(
                Done {
                    message,
                    ..Done::default()
                },
                &session,
            ))
        }
        Job::Reindex => {
            let n = pipeline::reindex_all(store, &Index::open(&store.index_path())?)?;
            Ok(Outcome {
                message: format!("indexed {n} sessions"),
                ..Outcome::default()
            })
        }
    }
}

/// How long to wait before transcribing a session again: 1, 5 or 30 minutes after a
/// conversation's 1st, 2nd or 3rd failed transcription (the shortest wins). `None` when no
/// conversation is due a retry; after the 4th failure a conversation is left failed.
pub fn retry_delay(meta: &Meta) -> Option<Duration> {
    meta.conversations
        .iter()
        .filter(|c| c.stage == Stage::Failed && c.failed_step == Some(Step::Transcribe))
        .filter_map(|c| match c.attempts {
            1 => Some(60),
            2 => Some(5 * 60),
            3 => Some(30 * 60),
            _ => None,
        })
        .min()
        .map(Duration::from_secs)
}

fn provider<'a>(
    asr: &'a mut Option<WhisperProvider>,
    config: &Config,
    store: &Store,
) -> Result<&'a mut WhisperProvider> {
    if asr.is_none() {
        *asr = Some(WhisperProvider::new(config, store)?);
    }
    Ok(asr.as_mut().expect("set above"))
}

/// What a pipeline run left behind: the lines to print and the conversations it summarized.
#[derive(Default)]
struct Done {
    message: String,
    summarized: Vec<u32>,
}

fn outcome(done: Done, session: &Session) -> Outcome {
    Outcome {
        message: done.message,
        session: Some(session.id.clone()),
        summarized: done.summarized,
    }
}

enum Work {
    Process,
    Retranscribe,
    Summarize(Option<u32>),
}

/// Runs the pipeline and returns the lines to print (where the transcript and summary are)
/// and the conversations it summarized.
fn run_pipeline(
    store: &Store,
    config: &Config,
    session: &Session,
    asr: Option<&mut WhisperProvider>,
    work: Work,
) -> Result<Done> {
    let index = pipeline::open_index(store)?;
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
    Ok(Done {
        message: check_report(session, &report)?,
        summarized: report.summarized,
    })
}

fn check_report(session: &Session, report: &Report) -> Result<String> {
    let mut lines = Vec::new();
    if session.transcript_path().is_file() {
        lines.push(format!(
            "transcript: {}",
            session.transcript_path().display()
        ));
    }
    if session.summary_path().is_file() {
        lines.push(format!("summary:    {}", session.summary_path().display()));
    }
    if let Some(e) = &report.compression_error {
        eprintln!("kute: compression failed (audio kept as WAV): {e}");
    }
    if !report.failed.is_empty() {
        for (n, e) in &report.failed {
            eprintln!("kute: conversation {n} failed: {e}");
        }
        return Err(ConversationsFailed {
            session: session.id.clone(),
            count: report.failed.len(),
            written: lines,
        }
        .into());
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Conversation, Meta, Stage, Step, Track, Trigger, TriggerKind};

    fn meta(conversations: Vec<Conversation>) -> Meta {
        let mut meta = Meta::new(
            "2026-10-19T100000".into(),
            "2026-10-19T10:00:00+02:00[+02:00]".parse().unwrap(),
            Trigger {
                kind: TriggerKind::Import,
                app_name: None,
                binary: None,
            },
            "auto".into(),
        );
        meta.conversations = conversations;
        meta
    }

    #[test]
    fn a_summarize_job_reports_what_it_summarized() {
        // The daemon pops a notification for these, so they must come back in the Outcome.
        use crate::config::SummaryBackend;
        use crate::store::{Segment, TrackTranscript};
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let mut m = meta(vec![Conversation::new(1, 0, 5_000)]);
        m.conversations[0].stage = Stage::Transcribed;
        let session = store.create_session(&m.started_at).unwrap();
        m.id = session.id.clone();
        session.save_meta(&m).unwrap();
        let segment = Segment {
            conversation: 1,
            start_ms: 0,
            end_ms: 2_000,
            text: " Let's ship the daemon.".into(),
            words: Vec::new(),
        };
        session
            .save_track(
                Track::Me,
                &TrackTranscript {
                    segments: vec![segment],
                },
            )
            .unwrap();
        let mut config = Config::default();
        config.summary.backend = SummaryBackend::Command;
        config.summary.command = vec![
            "sh".into(),
            "-c".into(),
            "cat > /dev/null; echo Shipped.".into(),
        ];
        config.summary.min_words = 1;
        let job = Job::Summarize {
            id: session.id.clone(),
            conversation: None,
            lang: None,
            backend: None,
        };
        let outcome = run(&store, &config, &mut None, &job).unwrap();
        assert_eq!(outcome.summarized, vec![1]);
    }

    #[test]
    fn a_notes_job_sets_and_clears_the_notes() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let mut m = meta(Vec::new());
        let session = store.create_session(&m.started_at).unwrap();
        m.id = session.id.clone();
        session.save_meta(&m).unwrap();
        let notes = |notes: Option<&str>| {
            let job = Job::Notes {
                id: session.id.clone(),
                notes: notes.map(Into::into),
            };
            run(&store, &Config::default(), &mut None, &job).unwrap();
            session.load_meta().unwrap().notes
        };
        assert_eq!(
            notes(Some("Jordan is the PM")).as_deref(),
            Some("Jordan is the PM")
        );
        assert_eq!(notes(None), None);
    }

    #[test]
    fn a_language_job_sets_the_languages_and_points_at_retranscribe() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let mut done = Conversation::new(1, 0, 1_000);
        done.stage = Stage::Summarized;
        let mut m = meta(vec![done, Conversation::new(2, 2_000, 3_000)]);
        let session = store.create_session(&m.started_at).unwrap();
        m.id = session.id.clone();
        session.save_meta(&m).unwrap();
        let set = |languages: LanguageChange| {
            let job = Job::Language {
                id: session.id.clone(),
                languages,
            };
            let outcome = run(&store, &Config::default(), &mut None, &job).unwrap();
            (outcome, session.load_meta().unwrap().transcribe)
        };
        let codes = |c: &[&str]| Some(c.iter().map(|c| c.to_string()).collect::<Vec<_>>());

        let (_, t) = set(LanguageChange {
            me: codes(&["ru"]),
            others: codes(&["ru", "en"]),
            ..Default::default()
        });
        assert_eq!(t.for_track(Track::Me), ["ru"]);
        assert_eq!(t.for_track(Track::Others), ["ru", "en"]);
        // Both tracks at once replaces the per-track languages.
        let (outcome, t) = set(LanguageChange {
            all: codes(&["it"]),
            ..Default::default()
        });
        assert_eq!(t.for_track(Track::Me), ["it"]);
        assert_eq!(t.for_track(Track::Others), ["it"]);
        assert!(
            outcome
                .message
                .starts_with("1 conversation(s) already transcribed"),
            "{}",
            outcome.message
        );
    }

    fn failed(n: u32, step: Step, attempts: u32) -> Conversation {
        let mut c = Conversation::new(n, 0, 1_000);
        c.stage = Stage::Failed;
        c.failed_step = Some(step);
        c.attempts = attempts;
        c
    }

    #[test]
    fn retry_delay_follows_the_backoff() {
        let delay = |c: Vec<Conversation>| retry_delay(&meta(c)).map(|d| d.as_secs());
        assert_eq!(delay(vec![failed(1, Step::Transcribe, 1)]), Some(60));
        assert_eq!(delay(vec![failed(1, Step::Transcribe, 2)]), Some(300));
        assert_eq!(delay(vec![failed(1, Step::Transcribe, 3)]), Some(1_800));
        assert_eq!(delay(vec![failed(1, Step::Transcribe, 4)]), None);
        assert_eq!(delay(vec![failed(1, Step::Summarize, 1)]), None);
        assert_eq!(
            delay(vec![
                failed(1, Step::Transcribe, 3),
                failed(2, Step::Transcribe, 1)
            ]),
            Some(60)
        );
        assert_eq!(delay(vec![Conversation::new(1, 0, 1_000)]), None);
    }

    #[test]
    fn job_json_round_trips() {
        let job = Job::Summarize {
            id: "x".into(),
            conversation: Some(2),
            lang: Some("it".into()),
            backend: None,
        };
        let value = serde_json::to_value(&job).unwrap();
        assert_eq!(value["job"], "summarize");
        assert_eq!(value["conversation"], 2);
        assert_eq!(serde_json::from_value::<Job>(value).unwrap(), job);
    }

    #[test]
    fn a_failed_import_processing_names_the_session() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        // Model files only need to exist: the prompt template fails before they are loaded.
        let model = tmp.path().join("fake.bin");
        std::fs::write(&model, b"").unwrap();
        let vad = crate::models::path_for(&store.models_dir(), &crate::models::VAD_MODEL);
        std::fs::create_dir_all(vad.parent().unwrap()).unwrap();
        std::fs::write(&vad, b"").unwrap();
        let mut config = Config::default();
        config.transcribe.model = model.to_string_lossy().into_owned();
        config.summary.prompt = Some(tmp.path().join("missing-prompt.md"));
        let wav = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/me.wav"
        ));
        let job = Job::Import {
            me: wav.clone(),
            others: wav,
            start: Some("2026-10-08T15:00:00".into()),
            title: None,
            languages: LanguageChange::default(),
            process: true,
        };

        let err = run(&store, &config, &mut None, &job).unwrap_err();
        let id = &store.list().unwrap()[0].id;
        let text = format!("{err:#}");
        assert!(text.starts_with(&format!("imported {id}: ")), "{text}");
        assert!(text.contains("prompt template"), "{text}");
    }

    #[test]
    fn what_was_written_before_a_failure_travels_in_the_error() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let session = store
            .create_session(&"2026-10-19T10:00:00+02:00[+02:00]".parse().unwrap())
            .unwrap();
        std::fs::write(session.transcript_path(), "# Call").unwrap();
        let report = Report {
            failed: vec![(2, "no model".into())],
            ..Report::default()
        };
        let err = check_report(&session, &report).unwrap_err();
        let failed = err.downcast_ref::<ConversationsFailed>().unwrap();
        assert_eq!(
            failed.written,
            [format!(
                "transcript: {}",
                session.transcript_path().display()
            )]
        );
    }

    #[test]
    fn conversations_failed_survives_the_import_context() {
        let err = anyhow::Error::from(ConversationsFailed {
            session: "s".into(),
            count: 1,
            written: Vec::new(),
        })
        .context("imported s");
        assert_eq!(
            err.downcast_ref::<ConversationsFailed>().unwrap().session,
            "s"
        );
        assert_eq!(
            format!("{err:#}"),
            "imported s: 1 conversation(s) failed; fix the cause and run `kute process s`"
        );
    }
}
