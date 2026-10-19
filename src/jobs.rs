//! Jobs: the processing work the CLI runs in the foreground and the daemon's worker runs
//! in the background.

use crate::config::{Config, SummaryBackend};
use crate::import::{self, ImportRequest};
use crate::index::Index;
use crate::pipeline::{self, AsrProvider, Pipeline, Report, WhisperProvider};
use crate::store::{Meta, Session, Stage, Step, Store};
use crate::summarize;
use anyhow::Result;
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
    },
    Summarize {
        id: String,
        conversation: Option<u32>,
        lang: Option<String>,
        backend: Option<SummaryBackend>,
    },
    /// `me` and `others` are used as given; the CLI makes them absolute before sending.
    Import {
        me: PathBuf,
        others: PathBuf,
        start: Option<String>,
        title: Option<String>,
        process: bool,
    },
    Title {
        id: String,
        title: String,
    },
    Reindex,
}

#[derive(Debug)]
pub struct Outcome {
    /// What the CLI prints on stdout; may be empty.
    pub message: String,
    pub session: Option<String>,
}

/// The error a job returns when the pipeline left conversations failed. It carries the
/// session, so the worker can schedule a transcription retry even for an `Import`.
#[derive(Debug)]
pub struct ConversationsFailed {
    pub session: String,
    pub count: usize,
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
            let message = run_pipeline(store, config, &session, Some(provider), Work::Process)?;
            Ok(outcome(message, &session))
        }
        Job::Retranscribe { id, model } => {
            let session = store.resolve(id)?;
            let message = match model {
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
            Ok(outcome(message, &session))
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
            let message = run_pipeline(
                store,
                &config,
                &session,
                None,
                Work::Summarize(*conversation),
            )?;
            Ok(outcome(message, &session))
        }
        Job::Import {
            me,
            others,
            start,
            title,
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
            let mut message = format!("imported {}", session.id);
            if *process {
                let provider = provider(asr, config, store)?;
                let report = run_pipeline(store, config, &session, Some(provider), Work::Process)?;
                if !report.is_empty() {
                    message = format!("{message}\n{report}");
                }
            }
            Ok(outcome(message, &session))
        }
        Job::Title { id, title } => {
            let session = store.resolve(id)?;
            let meta = session.update_meta(|m| m.title = Some(title.clone()))?;
            pipeline::write_documents(&session, &meta, &config.labels)?;
            pipeline::index_session(&pipeline::open_index(store)?, &session)?;
            Ok(outcome(String::new(), &session))
        }
        Job::Reindex => {
            let n = pipeline::reindex_all(store, &Index::open(&store.index_path())?)?;
            Ok(Outcome {
                message: format!("indexed {n} sessions"),
                session: None,
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

fn outcome(message: String, session: &Session) -> Outcome {
    Outcome {
        message,
        session: Some(session.id.clone()),
    }
}

enum Work {
    Process,
    Retranscribe,
    Summarize(Option<u32>),
}

/// Runs the pipeline and returns the lines to print: where the transcript and summary are.
fn run_pipeline(
    store: &Store,
    config: &Config,
    session: &Session,
    asr: Option<&mut WhisperProvider>,
    work: Work,
) -> Result<String> {
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
    check_report(session, &report)
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
        }
        .into());
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Conversation, Meta, Stage, Step, Trigger, TriggerKind};

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
}
