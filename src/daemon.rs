//! `kute daemon`: detects calls, records them, and queues their processing on the worker.
//!
//! `Daemon::handle` is the only place state changes. `run` feeds it events from the
//! PipeWire monitor, the control socket, the worker and a 500 ms tick, with the time
//! since the daemon started; tests feed it directly with a fake clock.

use crate::capture::{CaptureSource, PwRecord};
use crate::config::Config;
use crate::detect::{Action, DetectError, Detector, Phase};
use crate::ipc::{self, Incoming, Request, Response, Status};
use crate::jobs::Job;
use crate::live::{self, LiveSession, StartRequest};
use crate::models;
use crate::pipeline;
use crate::pw::{self, App, MonitorEvent, Snapshot};
use crate::recorder::VadFactory;
use crate::store::Store;
use crate::transcribe::SpeechDetector;
use crate::transcribe::whisper::SileroVad;
use crate::worker::{self, WorkerDone, WorkerMsg};
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// How often the detector and the live session are looked at between graph changes.
const TICK: Duration = Duration::from_millis(500);

pub enum Event {
    Monitor(MonitorEvent),
    Ipc(Incoming),
    Worker(WorkerDone),
    Tick,
}

pub struct Daemon {
    store: Store,
    config: Config,
    detector: Detector,
    live: Option<LiveSession>,
    /// The last graph snapshot: default devices and sources for the next session.
    snapshot: Snapshot,
    queue: VecDeque<Job>,
    running: Option<Job>,
    /// Processing to run again once its time (daemon ms) comes, after failed transcriptions.
    retries: Vec<(u64, Job)>,
    /// Sessions discarded while a job was running: that job may index them again before
    /// it notices the folder is gone, so they leave the index once more when it ends.
    discarded: Vec<String>,
    worker: Sender<WorkerMsg>,
    source: Box<dyn CaptureSource>,
    vad: VadFactory,
}

impl Daemon {
    pub fn new(
        store: Store,
        config: Config,
        source: Box<dyn CaptureSource>,
        vad: VadFactory,
        worker: Sender<WorkerMsg>,
    ) -> Daemon {
        Daemon {
            detector: Detector::new(&config.detect, std::process::id()),
            store,
            config,
            live: None,
            snapshot: Snapshot::default(),
            queue: VecDeque::new(),
            running: None,
            retries: Vec::new(),
            discarded: Vec::new(),
            worker,
            source,
            vad,
        }
    }

    pub fn handle(&mut self, event: Event, now_ms: u64) -> Result<()> {
        match event {
            Event::Monitor(MonitorEvent::Snapshot(snapshot)) => {
                self.snapshot = snapshot;
                let actions = self.detector.streams(&self.snapshot.streams, now_ms);
                self.apply(actions)
            }
            Event::Monitor(MonitorEvent::Disconnected(reason)) => self.disconnected(&reason),
            Event::Tick => self.tick(now_ms),
            Event::Ipc(Incoming { request, reply }) => {
                let response = self.request(request, now_ms);
                // The client may have given up waiting.
                let _ = reply.send(response);
                Ok(())
            }
            Event::Worker(done) => self.done(done, now_ms),
        }
    }

    pub fn status(&self) -> Status {
        Status {
            phase: self.detector.phase(),
            paused: self.detector.paused(),
            session: self.live.as_ref().map(|l| l.id().to_owned()),
            app: self.detector.trigger().and_then(app_label),
            queued: self.queue.len(),
            running_job: self.running.as_ref().map(ToString::to_string),
        }
    }

    /// Spec: "close the current session as if it ended". A Tentative one was never shown,
    /// so it goes.
    fn disconnected(&mut self, reason: &str) -> Result<()> {
        eprintln!("kute: lost the PipeWire graph: {reason}");
        let ended = match self.detector.phase() {
            Phase::Idle => return Ok(()),
            Phase::Tentative => self.detector.discard(),
            Phase::Recording | Phase::Grace => self.detector.stop(),
        };
        self.apply(ended.unwrap_or_default())
    }

    fn tick(&mut self, now_ms: u64) -> Result<()> {
        if let Some(live) = &mut self.live {
            let id = live.id().to_owned();
            let pumped = live.pump();
            let mut failed = live.failed().map(str::to_owned);
            match pumped {
                Ok(closed) if !closed.is_empty() => {
                    self.enqueue(Job::Process { id: id.clone() })?
                }
                Ok(_) => {}
                // Saving a conversation failed (disk full): keep what was written.
                Err(e) => failed = failed.or(Some(format!("{e:#}"))),
            }
            if let Some(reason) = failed {
                eprintln!("kute: {id}: ending the session: {reason}");
                let ended = self.detector.stop().unwrap_or_default();
                self.apply(ended)?;
            }
        }
        let actions = self.detector.tick(now_ms);
        self.apply(actions)?;

        let (due, later) = std::mem::take(&mut self.retries)
            .into_iter()
            .partition(|(at, _)| *at <= now_ms);
        self.retries = later;
        for (_, job) in due {
            self.enqueue(job)?;
        }
        Ok(())
    }

    /// Answers a control request. Errors become the reply; the ones that are not just a
    /// request that doesn't apply are logged too.
    fn request(&mut self, request: Request, now_ms: u64) -> Response {
        self.control(request, now_ms).unwrap_or_else(|e| {
            if e.downcast_ref::<DetectError>().is_none() {
                eprintln!("kute: {e:#}");
            }
            Response::Error {
                message: format!("{e:#}"),
            }
        })
    }

    fn control(&mut self, request: Request, now_ms: u64) -> Result<Response> {
        let message = match request {
            Request::Status => {
                return Ok(Response::Status {
                    status: self.status(),
                });
            }
            Request::Record { title } => {
                let actions = match self.detector.record(now_ms) {
                    Err(DetectError::AlreadyRecording) => {
                        return Ok(Response::Error {
                            message: format!("already recording {}", self.live_id()),
                        });
                    }
                    other => other?,
                };
                self.apply(actions)?;
                let id = self.live_id();
                if let Some(title) = title {
                    self.store
                        .session(&id)?
                        .update_meta(|m| m.title = Some(title))?;
                }
                format!("recording {id}")
            }
            Request::Stop => {
                let id = self.live_id();
                let actions = self.detector.stop()?;
                self.apply(actions)?;
                format!("stopped {id}")
            }
            Request::Discard => {
                let id = self.live_id();
                let actions = self.detector.discard()?;
                self.apply(actions)?;
                format!("discarded {id}")
            }
            Request::Pause => {
                let actions = self.detector.pause();
                self.apply(actions)?;
                "paused".into()
            }
            Request::Resume => {
                self.detector.resume();
                // A call already in progress shows up in the last snapshot, which may be
                // the last one for a while.
                let actions = self.detector.streams(&self.snapshot.streams, now_ms);
                self.apply(actions)?;
                "resumed".into()
            }
            Request::Job { job } => {
                let message = format!("queued {job}");
                self.enqueue(job)?;
                message
            }
        };
        Ok(Response::Ok { message })
    }

    fn done(&mut self, done: WorkerDone, now_ms: u64) -> Result<()> {
        self.running = None;
        match &done.result {
            Ok(outcome) if !outcome.message.is_empty() => {
                eprintln!(
                    "kute: {}: {}",
                    done.job,
                    outcome.message.replace('\n', "; ")
                )
            }
            Ok(_) => eprintln!("kute: {}: done", done.job),
            Err(e) => eprintln!("kute: {} failed: {e}", done.job),
        }
        let discarded = std::mem::take(&mut self.discarded);
        for id in &discarded {
            unindex(&self.store, id);
        }
        // Always `Process`, even after an `Import`: the session exists now.
        if let (Some(delay), Some(id)) = (done.retry_after, done.session)
            && !discarded.contains(&id)
        {
            eprintln!("kute: transcribing {id} again in {} s", delay.as_secs());
            let at = now_ms.saturating_add(delay.as_millis() as u64);
            self.retries.push((at, Job::Process { id }));
        }
        self.start_next()
    }

    /// Queues a job and starts it if the worker is free. A `Process` already waiting
    /// covers a second one; one that is running doesn't, since it may have started before
    /// the conversations that prompted the second.
    fn enqueue(&mut self, job: Job) -> Result<()> {
        let covered = matches!(job, Job::Process { .. }) && self.queue.contains(&job);
        if !covered {
            self.queue.push_back(job);
        }
        self.start_next()
    }

    fn start_next(&mut self) -> Result<()> {
        if self.running.is_some() {
            return Ok(());
        }
        let Some(job) = self.queue.pop_front() else {
            return Ok(());
        };
        if self.worker.send(WorkerMsg::Run(job.clone())).is_err() {
            self.queue.push_front(job);
            anyhow::bail!("the worker has stopped");
        }
        self.running = Some(job);
        Ok(())
    }

    fn apply(&mut self, actions: Vec<Action>) -> Result<()> {
        for action in actions {
            match action {
                Action::Start { trigger } => {
                    if let Err(e) = self.start(trigger) {
                        // Back to Idle: there is no session for the detector to track.
                        let _ = self.detector.discard();
                        return Err(e.context("could not start recording"));
                    }
                }
                Action::Commit => {
                    let app = self.status().app.unwrap_or_else(|| "manual".into());
                    eprintln!("kute: recording {} ({app})", self.live_id());
                }
                Action::Finish => {
                    let Some(live) = self.live.take() else {
                        continue;
                    };
                    let id = live.id().to_owned();
                    let session = live.finish().with_context(|| format!("finishing {id}"))?;
                    eprintln!("kute: finished {id}");
                    self.enqueue(Job::Process { id: session.id })?;
                }
                Action::Discard => {
                    let Some(live) = self.live.take() else {
                        continue;
                    };
                    let id = live.id().to_owned();
                    live.discard().with_context(|| format!("discarding {id}"))?;
                    eprintln!("kute: discarded {id}");
                    self.forget(&id);
                }
            }
        }
        Ok(())
    }

    /// Drops what is left of a discarded session: the work queued for it and its rows in
    /// the search index (its closed conversations may already be transcribed).
    fn forget(&mut self, id: &str) {
        self.queue.retain(|job| job.session() != Some(id));
        self.retries.retain(|(_, job)| job.session() != Some(id));
        if self.running.is_some() {
            self.discarded.push(id.to_owned());
        }
        unindex(&self.store, id);
    }

    fn start(&mut self, trigger: Option<App>) -> Result<()> {
        let mic_target = self.config.capture.mic.as_deref().and_then(|mic| {
            let found = self.snapshot.find_source(mic);
            if found.is_none() {
                eprintln!("kute: capture.mic {mic:?} matches no source; using the default");
            }
            found
        });
        let devices = (
            mic_target
                .clone()
                .or_else(|| self.snapshot.default_source.clone()),
            self.snapshot
                .default_sink
                .as_ref()
                .map(|sink| format!("{sink}.monitor")),
        );
        let app = trigger.as_ref().and_then(app_label);
        let req = StartRequest {
            trigger,
            title: None,
            mic_target,
            devices,
        };
        let live = LiveSession::start(
            &self.store,
            &self.config,
            req,
            self.source.as_ref(),
            self.vad.clone(),
        )?;
        eprintln!(
            "kute: started {} ({})",
            live.id(),
            app.as_deref().unwrap_or("manual")
        );
        self.live = Some(live);
        Ok(())
    }

    /// The live session's ID; empty when there is none.
    fn live_id(&self) -> String {
        self.live
            .as_ref()
            .map(|l| l.id().to_owned())
            .unwrap_or_default()
    }
}

/// Removes a session from the search index. A failure is only logged: the session is gone
/// either way, and `kute reindex` rebuilds the index from what is on disk.
fn unindex(store: &Store, id: &str) {
    let removed = pipeline::open_index(store).and_then(|index| index.delete_session(id));
    if let Err(e) = removed {
        eprintln!("kute: removing {id} from the search index: {e:#}");
    }
}

fn app_label(app: &App) -> Option<String> {
    app.name.clone().or_else(|| app.binary.clone())
}

/// The daemon itself: wires PipeWire, the socket, the worker and the clock into `Daemon`.
pub fn run(store: Store, config: Config) -> Result<()> {
    let socket = ipc::socket_path()?;
    let (events_tx, events) = mpsc::channel();
    let (ipc_tx, ipc_rx) = mpsc::channel();
    // Bind first: it fails while another daemon runs, before recovery could take that
    // daemon's live sessions for crashed ones.
    ipc::serve(&socket, ipc_tx)?;
    let _socket = RemoveOnDrop(socket);
    forward(ipc_rx, &events_tx, Event::Ipc);

    live::recover(&store)?;
    let startup = worker::startup_jobs(&store)?;

    let (work_tx, work_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    worker::spawn(store.clone(), config.clone(), work_rx, done_tx);
    forward(done_rx, &events_tx, Event::Worker);

    let (pw_tx, pw_rx) = mpsc::channel();
    pw::spawn(pw_tx);
    forward(pw_rx, &events_tx, Event::Monitor);

    let tick = events_tx.clone();
    thread::spawn(move || {
        while tick.send(Event::Tick).is_ok() {
            thread::sleep(TICK);
        }
    });
    drop(events_tx);

    let vad = vad_factory(&store);
    let mut daemon = Daemon::new(store, config, Box::new(PwRecord), vad, work_tx);
    for job in startup {
        daemon.enqueue(job)?;
    }
    eprintln!("kute: daemon started");
    let clock = Instant::now();
    for event in events {
        let now_ms = clock.elapsed().as_millis() as u64;
        if let Err(e) = daemon.handle(event, now_ms) {
            eprintln!("kute: {e:#}");
        }
    }
    let _ = daemon.worker.send(WorkerMsg::Shutdown);
    Ok(())
}

/// Moves everything from `rx` into the event channel, on its own thread.
fn forward<T: Send + 'static>(rx: Receiver<T>, tx: &Sender<Event>, wrap: fn(T) -> Event) {
    let tx = tx.clone();
    thread::spawn(move || {
        for item in rx {
            if tx.send(wrap(item)).is_err() {
                break;
            }
        }
    });
}

/// Silero for live conversation splitting, when its model is downloaded. Checked per
/// session, so a model downloaded while the daemon runs is picked up.
fn vad_factory(store: &Store) -> VadFactory {
    let path = models::path_for(&store.models_dir(), &models::VAD_MODEL);
    if !path.is_file() {
        eprintln!(
            "kute: the VAD model is not downloaded (`kute model download`); until it is, \
             each recording becomes one conversation"
        );
    }
    Arc::new(move || {
        if !path.is_file() {
            return None;
        }
        match SileroVad::load(&path) {
            Ok(vad) => Some(Box::new(vad) as Box<dyn SpeechDetector>),
            Err(e) => {
                eprintln!("kute: {e:#}");
                None
            }
        }
    })
}

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
