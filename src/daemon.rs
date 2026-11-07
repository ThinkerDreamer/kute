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
use crate::notify::{self, Click, Desktop, Notifier, Silent};
use crate::pipeline;
use crate::pw::{self, App, MonitorEvent, Snapshot};
use crate::recorder::VadFactory;
use crate::store::Store;
use crate::transcribe::SpeechDetector;
use crate::transcribe::whisper::SileroVad;
use crate::tray::{Tray, TrayAction};
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
/// A capture that dies this soon after its session started points at pw-record itself.
const BROKEN_CAPTURE_MS: u64 = 5_000;

pub enum Event {
    Monitor(MonitorEvent),
    Ipc(Incoming),
    Worker(WorkerDone),
    /// A notification's Stop or Discard button.
    Click(Click),
    /// The tray icon's menu.
    Tray(TrayAction),
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
    notifier: Box<dyn Notifier>,
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
            notifier: Box::new(Silent),
        }
    }

    /// Where announcements go (recording, a summary is ready, a failure). Silent unless set.
    pub fn with_notifier(mut self, notifier: Box<dyn Notifier>) -> Daemon {
        self.notifier = notifier;
        self
    }

    pub fn handle(&mut self, event: Event, now_ms: u64) -> Result<()> {
        match event {
            Event::Monitor(MonitorEvent::Snapshot(snapshot)) => {
                self.snapshot = snapshot;
                let actions = self.detector.streams(&self.snapshot.streams, now_ms);
                self.apply(actions)
            }
            Event::Monitor(MonitorEvent::Disconnected(reason)) => {
                self.disconnected(&reason, now_ms)
            }
            Event::Tick => self.tick(now_ms),
            Event::Ipc(Incoming { request, reply }) => {
                let response = self.request(request, now_ms);
                // The client may have given up waiting.
                let _ = reply.send(response);
                Ok(())
            }
            Event::Worker(done) => self.done(done, now_ms),
            Event::Click(click) => {
                self.click(click, now_ms);
                Ok(())
            }
            Event::Tray(action) => {
                self.tray(action, now_ms);
                Ok(())
            }
        }
    }

    pub fn status(&self) -> Status {
        Status {
            phase: self.detector.phase(),
            paused: self.detector.paused(),
            session: self.live.as_ref().map(|l| l.id().to_owned()),
            app: self.detector.trigger().and_then(app_label),
            queued: self.queue.len(),
            retrying: self.retries.len(),
            running_job: self.running.as_ref().map(ToString::to_string),
        }
    }

    /// Spec: "close the current session as if it ended". The graph is unknown until the
    /// monitor reconnects, so the detector sees no streams: an automatic session ends the
    /// way it would if the call had ended, unless the call is back in time; a manual one
    /// carries on. Not the user's stop, so nothing blocks the call from being picked up
    /// again (R24); a block the user's stop already set stays until a snapshot after the
    /// reconnect shows the mic gone (R26).
    fn disconnected(&mut self, reason: &str, now_ms: u64) -> Result<()> {
        eprintln!("kute: lost the PipeWire graph: {reason}");
        self.snapshot = Snapshot::default();
        let actions = self.detector.graph_lost(now_ms);
        self.apply(actions)
    }

    fn tick(&mut self, now_ms: u64) -> Result<()> {
        if let Some(live) = &mut self.live {
            let id = live.id().to_owned();
            let pumped = live.pump();
            let mut failed = live.failed().map(str::to_owned);
            let has_audio = live.has_audio();
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
                self.end_failed(&id, &reason, has_audio, now_ms)?;
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

    /// Ends a session whose recording failed. That is not the user's stop, so a call still
    /// in progress may start a new session (R24), unless the capture died within
    /// `BROKEN_CAPTURE_MS` of starting: pw-record itself is then likely broken, and would
    /// fail the same way every tick. A Tentative session with no audio is discarded.
    fn end_failed(&mut self, id: &str, reason: &str, has_audio: bool, now_ms: u64) -> Result<()> {
        let discard = self.detector.phase() == Phase::Tentative && !has_audio;
        let app = self.status().app;
        self.notifier.notify(notify::recording_failed_note(
            id,
            app.as_deref(),
            reason,
            !discard,
        ));
        let early = self
            .detector
            .started_ms()
            .is_some_and(|started| now_ms.saturating_sub(started) < BROKEN_CAPTURE_MS);
        let ended = match (discard, early) {
            (false, false) => self.detector.stop_unblocked(),
            (false, true) => self.detector.stop(),
            (true, false) => self.detector.discard_unblocked(),
            (true, true) => self.detector.discard(),
        };
        // The detector already counts the session as over: end it here too, so the two
        // agree and this doesn't come back every tick.
        let actions = ended.unwrap_or_else(|_| {
            vec![if discard {
                Action::Discard
            } else {
                Action::Finish
            }]
        });
        self.apply(actions)
    }

    /// A notification's Stop or Discard, for the session it was shown for: by the time it is
    /// clicked, that one may be over and another recording.
    fn click(&mut self, click: Click, now_ms: u64) {
        let (request, id) = match click {
            Click::Stop(id) => (Request::Stop, id),
            Click::Discard(id) => (Request::Discard, id),
            Click::OpenSummary(_) | Click::OpenFolder(_) => return,
        };
        self.act(request, Some(&id), "the notification", now_ms);
    }

    /// The tray menu's choice. Stop and Discard name the session the menu showed, like a
    /// notification's buttons.
    fn tray(&mut self, action: TrayAction, now_ms: u64) {
        let (request, id) = match action {
            TrayAction::Record => (Request::Record { title: None }, None),
            TrayAction::Stop(id) => (Request::Stop, Some(id)),
            TrayAction::Discard(id) => (Request::Discard, Some(id)),
            TrayAction::Pause => (Request::Pause, None),
            TrayAction::Resume => (Request::Resume, None),
        };
        self.act(request, id.as_deref(), "the tray", now_ms);
    }

    /// Runs a request from a button, and logs what came of it. `for_id` is the session the
    /// button was shown for; if another one is recording by now, nothing happens.
    fn act(&mut self, request: Request, for_id: Option<&str>, from: &str, now_ms: u64) {
        if let Some(id) = for_id
            && self.live_id() != id
        {
            eprintln!("kute: {id} is no longer recording; ignoring {from}");
            return;
        }
        match self.request(request, now_ms) {
            Response::Ok { message } => eprintln!("kute: {message} (from {from})"),
            Response::Error { message } => eprintln!("kute: {message}"),
            Response::Status { .. } => {}
        }
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
        if let Ok(outcome) = &done.result {
            self.announce_summaries(outcome);
        }
        self.announce_problems(&done);
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

    /// A notification for the summaries a job just wrote, if any. A session that's gone
    /// (discarded meanwhile) gets none.
    fn announce_summaries(&self, outcome: &crate::jobs::Outcome) {
        let Some(id) = &outcome.session else { return };
        if outcome.summarized.is_empty() {
            return;
        }
        let Ok(session) = self.store.session(id) else {
            return;
        };
        let Ok(meta) = session.load_meta() else {
            return;
        };
        if let Some(note) = notify::summary_note(&session, &meta, &outcome.summarized) {
            self.notifier.notify(note);
        }
    }

    /// A notification for what a job left failed, once nothing more will happen on its own
    /// (a transcription with retries left waits for them), or for a job that failed outright.
    fn announce_problems(&self, done: &WorkerDone) {
        if done.problems.is_empty() {
            if let Err(e) = &done.result {
                self.notifier
                    .notify(notify::job_failed_note(&done.job.to_string(), e));
            }
            return;
        }
        let Some(id) = &done.session else { return };
        let Ok(session) = self.store.session(id) else {
            return;
        };
        let Ok(meta) = session.load_meta() else {
            return;
        };
        if let Some(note) = notify::problems_note(&session, &meta, &done.problems) {
            self.notifier.notify(note);
        }
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
                    let app = self.status().app;
                    let id = self.live_id();
                    eprintln!(
                        "kute: recording {id} ({})",
                        app.as_deref().unwrap_or("manual")
                    );
                    self.notifier
                        .notify(notify::recording_note(&id, app.as_deref()));
                }
                Action::Finish => {
                    self.notifier.withdraw(notify::RECORDING);
                    let Some(live) = self.live.take() else {
                        continue;
                    };
                    let id = live.id().to_owned();
                    let session = live.finish().with_context(|| format!("finishing {id}"))?;
                    eprintln!("kute: finished {id}");
                    self.enqueue(Job::Process { id: session.id })?;
                }
                Action::Discard => {
                    self.notifier.withdraw(notify::RECORDING);
                    let Some(live) = self.live.take() else {
                        continue;
                    };
                    let id = live.id().to_owned();
                    let closed = live.has_closed_conversation();
                    live.discard().with_context(|| format!("discarding {id}"))?;
                    eprintln!("kute: discarded {id}");
                    self.forget(&id, closed);
                }
            }
        }
        Ok(())
    }

    /// Drops what is left of a discarded session: the work queued for it and its rows in
    /// the search index, when it may have some: a conversation `closed`, or a job for it
    /// was queued, running or retrying. A mic check never gets that far, and skips opening
    /// the index here, which can wait on the worker's writes.
    fn forget(&mut self, id: &str, closed: bool) {
        let ours = |job: &Job| job.session() == Some(id);
        let indexed = closed
            || self.queue.iter().any(ours)
            || self.running.as_ref().is_some_and(ours)
            || self.retries.iter().any(|(_, job)| ours(job));
        self.queue.retain(|job| !ours(job));
        self.retries.retain(|(_, job)| !ours(job));
        if self.running.is_some() {
            self.discarded.push(id.to_owned());
        }
        if indexed {
            unindex(&self.store, id);
        }
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
    let (click_tx, click_rx) = mpsc::channel();
    forward(click_rx, &events_tx, Event::Click);

    let tray = if config.tray.enabled {
        let (tray_tx, tray_rx) = mpsc::channel();
        forward(tray_rx, &events_tx, Event::Tray);
        Tray::spawn(tray_tx, store.sessions_dir(), config.notify.open.clone())
            .inspect_err(|e| eprintln!("kute: no tray icon: {e:#}"))
            .ok()
    } else {
        None
    };
    drop(events_tx);

    let vad = vad_factory(&store);
    let desktop = Box::new(Desktop::new(&config.notify, store.clone(), click_tx));
    let mut daemon =
        Daemon::new(store, config, Box::new(PwRecord), vad, work_tx).with_notifier(desktop);
    for job in startup {
        daemon.enqueue(job)?;
    }
    eprintln!("kute: daemon started");
    let clock = Instant::now();
    let mut shown = None;
    for event in events {
        let now_ms = clock.elapsed().as_millis() as u64;
        if let Err(e) = daemon.handle(event, now_ms) {
            eprintln!("kute: {e:#}");
        }
        if let Some(tray) = &tray {
            let status = daemon.status();
            if shown.as_ref() != Some(&status) {
                tray.show(&status);
                shown = Some(status);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{SessionState, Track};
    use crate::testing::{FakeSource, run_vad};

    #[test]
    fn a_failed_capture_ends_the_session_even_when_the_detector_is_idle() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let source = FakeSource::default();
        let (tx, worker) = mpsc::channel();
        let mut d = Daemon::new(
            store.clone(),
            Config::default(),
            Box::new(source.clone()),
            run_vad(),
            tx,
        );
        d.control(Request::Record { title: None }, 0).unwrap();
        let id = d.live_id();
        // However the two came apart, the detector has no session.
        d.detector = Detector::new(&d.config.detect, 1);
        source.close(Track::Me);
        let deadline = Instant::now() + Duration::from_secs(30);
        while d.live.is_some() {
            assert!(Instant::now() < deadline, "the session never ended");
            thread::sleep(Duration::from_millis(10));
            d.handle(Event::Tick, 10_000).unwrap();
        }
        let meta = store.session(&id).unwrap().load_meta().unwrap();
        assert_eq!(meta.state, SessionState::Processing);
        assert!(matches!(
            worker.try_recv(),
            Ok(WorkerMsg::Run(Job::Process { .. }))
        ));
    }
}
