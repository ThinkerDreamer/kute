//! The daemon's single background worker: runs one job at a time, so transcription and
//! summarization never share the GPU.

use crate::config::Config;
use crate::jobs::{self, ConversationsFailed, Job, Outcome};
use crate::pipeline::WhisperProvider;
use crate::store::{SessionState, Store};
use anyhow::Result;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

pub enum WorkerMsg {
    Run(Job),
    Shutdown,
}

#[derive(Debug)]
pub struct WorkerDone {
    pub job: Job,
    pub result: Result<Outcome, String>,
    /// When to run the session's processing again, after a failed transcription.
    pub retry_after: Option<Duration>,
}

/// Starts the worker. It runs jobs until `Shutdown`, or until either channel closes. The
/// Whisper provider is kept between jobs; the pipeline frees its models after each
/// transcription run, so nothing stays in VRAM while the worker waits.
pub fn spawn(
    store: Store,
    config: Config,
    rx: Receiver<WorkerMsg>,
    done: Sender<WorkerDone>,
) -> JoinHandle<()> {
    spawn_with(store, config, rx, done, jobs::run)
}

type RunJob = fn(&Store, &Config, &mut Option<WhisperProvider>, &Job) -> Result<Outcome>;

/// `spawn` with the function that runs each job, so tests can make one panic.
fn spawn_with(
    store: Store,
    config: Config,
    rx: Receiver<WorkerMsg>,
    done: Sender<WorkerDone>,
    run: RunJob,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("kute-worker".into())
        .spawn(move || {
            let mut asr = None;
            while let Ok(WorkerMsg::Run(job)) = rx.recv() {
                let ran = catch_unwind(AssertUnwindSafe(|| run(&store, &config, &mut asr, &job)));
                let result = match ran {
                    Ok(result) => result,
                    Err(panic) => {
                        // The provider may be half-loaded; build a fresh one next time.
                        asr = None;
                        Err(anyhow::anyhow!("job panicked: {}", panic_message(&*panic)))
                    }
                };
                let retry_after = retry_after(&store, &job, &result);
                let done_msg = WorkerDone {
                    job,
                    result: result.map_err(|e| format!("{e:#}")),
                    retry_after,
                };
                if done.send(done_msg).is_err() {
                    break;
                }
            }
        })
        .expect("spawning the worker thread")
}

/// The jobs to queue at daemon start: one `Process` for every session left in `processing`
/// state, oldest first.
pub fn startup_jobs(store: &Store) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for session in store.list()? {
        match session.load_meta() {
            Ok(meta) if meta.state == SessionState::Processing => {
                jobs.push(Job::Process { id: session.id })
            }
            Ok(_) => {}
            Err(e) => eprintln!("kute: skipping {}: {e:#}", session.id),
        }
    }
    Ok(jobs)
}

/// For a job that transcribes, the backoff before its session's failed transcriptions are
/// tried again.
fn retry_after(store: &Store, job: &Job, result: &Result<Outcome>) -> Option<Duration> {
    let transcribes = matches!(job, Job::Process { .. } | Job::Import { process: true, .. });
    if !transcribes {
        return None;
    }
    let id = match result {
        Ok(outcome) => outcome.session.clone()?,
        Err(e) => e.downcast_ref::<ConversationsFailed>()?.session.clone(),
    };
    let meta = store.session(&id).and_then(|s| s.load_meta());
    match meta {
        Ok(meta) => jobs::retry_delay(&meta),
        Err(e) => {
            eprintln!("kute: cannot schedule a retry for {id}: {e:#}");
            None
        }
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = panic.downcast_ref::<&str>() {
        s
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s
    } else {
        "unknown panic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::store::{Meta, SessionState, Store, Trigger, TriggerKind};
    use std::sync::mpsc;

    fn store() -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        (tmp, store)
    }

    fn add_session(store: &Store, start: &str, state: SessionState) -> String {
        let start: jiff::Zoned = start.parse().unwrap();
        let session = store.create_session(&start).unwrap();
        let trigger = Trigger {
            kind: TriggerKind::Import,
            app_name: None,
            binary: None,
        };
        let mut meta = Meta::new(session.id.clone(), start, trigger, "auto".into());
        meta.state = state;
        session.save_meta(&meta).unwrap();
        session.id
    }

    #[test]
    fn startup_jobs_lists_processing_sessions_in_order() {
        let (_tmp, store) = store();
        let newer = add_session(
            &store,
            "2026-10-19T11:00:00+02:00[+02:00]",
            SessionState::Processing,
        );
        add_session(
            &store,
            "2026-10-19T09:00:00+02:00[+02:00]",
            SessionState::Done,
        );
        let older = add_session(
            &store,
            "2026-10-19T10:00:00+02:00[+02:00]",
            SessionState::Processing,
        );
        assert_eq!(
            startup_jobs(&store).unwrap(),
            vec![Job::Process { id: older }, Job::Process { id: newer }]
        );
    }

    #[test]
    fn worker_survives_a_failing_job() {
        let (_tmp, store) = store();
        let (tx, rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let handle = spawn(store, Config::default(), rx, done_tx);
        tx.send(WorkerMsg::Run(Job::Process {
            id: "missing".into(),
        }))
        .unwrap();
        tx.send(WorkerMsg::Run(Job::Reindex)).unwrap();

        let first = done_rx.recv().unwrap();
        assert_eq!(
            first.job,
            Job::Process {
                id: "missing".into()
            }
        );
        assert!(first.result.is_err());
        assert_eq!(first.retry_after, None);
        let second = done_rx.recv().unwrap();
        assert_eq!(second.job, Job::Reindex);
        assert_eq!(second.result.unwrap().message, "indexed 0 sessions");

        tx.send(WorkerMsg::Shutdown).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn worker_reports_a_panicking_job_and_keeps_going() {
        fn run(
            store: &Store,
            config: &Config,
            asr: &mut Option<WhisperProvider>,
            job: &Job,
        ) -> Result<Outcome> {
            if matches!(job, Job::Title { .. }) {
                panic!("boom");
            }
            jobs::run(store, config, asr, job)
        }
        let (_tmp, store) = store();
        let (tx, rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let handle = spawn_with(store, Config::default(), rx, done_tx, run);
        let title = Job::Title {
            id: "x".into(),
            title: "Planning".into(),
        };
        tx.send(WorkerMsg::Run(title)).unwrap();
        tx.send(WorkerMsg::Run(Job::Reindex)).unwrap();

        let err = done_rx.recv().unwrap().result.unwrap_err();
        assert!(err.contains("panicked: boom"), "{err}");
        assert!(done_rx.recv().unwrap().result.is_ok());
        drop(tx);
        handle.join().unwrap();
    }
}
