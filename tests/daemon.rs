//! The daemon end to end, driven through `Daemon::handle` with a fake clock: no PipeWire,
//! no models. Captures are `FakeSource` pipes and speech detection is `RunVad`.

use anyhow::{Result, bail};
use kute::capture::{Capture, CaptureSource};
use kute::config::Config;
use kute::daemon::{Daemon, Event};
use kute::detect::Phase;
use kute::index::Index;
use kute::ipc::{Incoming, Request, Response};
use kute::jobs::{Job, Outcome};
use kute::merge::Line;
use kute::pw::{App, MonitorEvent, Snapshot, Source, Stream, StreamKind};
use kute::store::{Meta, SessionState, Store, Track, Trigger, TriggerKind};
use kute::testing::{FakeSource, RecordingNotifier, run_vad};
use kute::worker::{WorkerDone, WorkerMsg};
use std::io::Write;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Harness {
    _tmp: tempfile::TempDir,
    store: Store,
    source: FakeSource,
    daemon: Daemon,
    worker: Receiver<WorkerMsg>,
    notes: RecordingNotifier,
    now: u64,
}

impl Harness {
    fn new() -> Harness {
        Harness::with(Config::default(), |source| Box::new(source))
    }

    fn with(config: Config, source: impl FnOnce(FakeSource) -> Box<dyn CaptureSource>) -> Harness {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let fake = FakeSource::default();
        let (tx, worker) = mpsc::channel();
        let notes = RecordingNotifier::default();
        let daemon = Daemon::new(store.clone(), config, source(fake.clone()), run_vad(), tx)
            .with_notifier(Box::new(notes.clone()));
        Harness {
            _tmp: tmp,
            store,
            source: fake,
            daemon,
            worker,
            notes,
            now: 0,
        }
    }

    fn snapshot(&mut self, streams: Vec<Stream>) -> Result<()> {
        let snap = Snapshot {
            streams,
            default_source: Some("alsa_input.builtin".into()),
            default_sink: Some("alsa_output.speakers".into()),
            sources: vec![
                source("alsa_input.builtin", "Built-in Audio"),
                source("alsa_input.usb-headset", "USB Headset"),
            ],
        };
        self.daemon
            .handle(Event::Monitor(MonitorEvent::Snapshot(snap)), self.now)
    }

    /// Ticks every 500 ms of fake time up to `until_ms`.
    fn tick_until(&mut self, until_ms: u64) {
        while self.now < until_ms {
            self.now += 500;
            self.daemon.handle(Event::Tick, self.now).unwrap();
        }
    }

    fn request(&mut self, request: Request) -> Response {
        let (reply, answer) = mpsc::channel();
        self.daemon
            .handle(Event::Ipc(Incoming { request, reply }), self.now)
            .unwrap();
        answer.try_recv().expect("no reply")
    }

    fn ok(&mut self, request: Request) -> String {
        match self.request(request) {
            Response::Ok { message } => message,
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    fn done(&mut self, job: Job, session: Option<&str>, retry_after: Option<Duration>) {
        let outcome = Outcome {
            session: session.map(Into::into),
            ..Outcome::default()
        };
        self.finished(job, Ok(outcome), retry_after);
    }

    fn finished(
        &mut self,
        job: Job,
        result: Result<Outcome, String>,
        retry_after: Option<Duration>,
    ) {
        let session = match &result {
            Ok(outcome) => outcome.session.clone(),
            Err(_) => None,
        };
        let done = WorkerDone {
            job,
            result,
            retry_after,
            session,
        };
        self.daemon.handle(Event::Worker(done), self.now).unwrap();
    }

    /// The jobs sent to the worker since the last call.
    fn sent(&self) -> Vec<Job> {
        self.worker
            .try_iter()
            .filter_map(|m| match m {
                WorkerMsg::Run(job) => Some(job),
                WorkerMsg::Shutdown => None,
            })
            .collect()
    }

    fn phase(&self) -> Phase {
        self.daemon.status().phase
    }

    fn disconnect(&mut self) {
        let event = Event::Monitor(MonitorEvent::Disconnected("gone".into()));
        self.daemon.handle(event, self.now).unwrap();
    }

    /// Ticks without moving the fake clock until the session ends, waiting for the track
    /// threads in real time.
    /// Ticks (in real time, so the capture threads can report) until session `id` is over.
    /// The detector may already be in a new session by then.
    fn tick_until_ended(&mut self, id: &str) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while self.daemon.status().session.as_deref() == Some(id) {
            assert!(Instant::now() < deadline, "the session never ended");
            std::thread::sleep(Duration::from_millis(10));
            self.daemon.handle(Event::Tick, self.now).unwrap();
        }
    }
}

fn source(name: &str, description: &str) -> Source {
    Source {
        name: name.into(),
        description: Some(description.into()),
    }
}

fn jitsi(serial: u64, kind: StreamKind, running: bool) -> Stream {
    Stream {
        serial,
        kind,
        app: App {
            name: Some("Jitsi Meet".into()),
            binary: Some("electron".into()),
        },
        node_name: None,
        pid: Some(4242),
        monitor: false,
        running,
    }
}

/// `seconds` of S16LE PCM, non-zero (speech for `RunVad`) inside `speech` (ms).
fn pcm(seconds: u64, speech: std::ops::Range<u64>) -> Vec<u8> {
    (0..seconds * 16_000)
        .flat_map(|i| {
            let v: i16 = if speech.contains(&(i / 16)) { 1_000 } else { 0 };
            v.to_le_bytes()
        })
        .collect()
}

#[test]
fn a_detected_call_is_recorded_split_and_queued() {
    let mut h = Harness::new();
    h.snapshot(vec![
        jitsi(100, StreamKind::Mic, true),
        jitsi(101, StreamKind::Output, true),
    ])
    .unwrap();
    assert_eq!(h.phase(), Phase::Tentative);

    // Pipes hold 64 KiB, so write from threads; the track threads drain them.
    let writers = [
        (Track::Me, pcm(70, 1_000..3_000)),
        (Track::Others, pcm(70, 0..0)),
    ]
    .map(|(track, bytes)| {
        let mut w = h.source.writer(track);
        std::thread::spawn(move || w.write_all(&bytes).unwrap())
    });
    for w in writers {
        w.join().unwrap();
    }

    h.tick_until(19_500);
    assert_eq!(h.phase(), Phase::Tentative);
    h.tick_until(20_000);
    let status = h.daemon.status();
    assert_eq!(status.phase, Phase::Recording);
    assert_eq!(status.app.as_deref(), Some("Jitsi Meet"));
    let id = status.session.clone().expect("a session");

    h.tick_until(60_000);
    // The call ends: the mic goes, and the app's output stops playing.
    h.snapshot(vec![jitsi(101, StreamKind::Output, false)])
        .unwrap();
    h.tick_until(70_000);
    assert_eq!(h.phase(), Phase::Grace);
    h.tick_until(78_000);
    assert_eq!(h.phase(), Phase::Idle);
    assert_eq!(h.daemon.status().session, None);

    let sessions = h.store.list().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, id);
    let meta = sessions[0].load_meta().unwrap();
    assert_eq!(meta.state, SessionState::Processing);
    assert_eq!(meta.trigger.kind, TriggerKind::App);
    assert_eq!(meta.trigger.app_name.as_deref(), Some("Jitsi Meet"));
    assert_eq!(meta.tracks.me.device.as_deref(), Some("alsa_input.builtin"));
    assert_eq!(
        meta.tracks.others.device.as_deref(),
        Some("alsa_output.speakers.monitor")
    );
    assert_eq!(meta.conversations.len(), 1, "{:?}", meta.conversations);
    assert_eq!(meta.conversations[0].start_ms, 0);
    assert!(meta.conversations[0].end_ms.abs_diff(5_000) <= 50);
    for track in Track::BOTH {
        let wav = hound::WavReader::open(sessions[0].wav_path(track)).unwrap();
        assert_eq!(wav.len(), 70 * 16_000, "{track:?}");
    }
    assert_eq!(h.sent(), vec![Job::Process { id }]);
}

#[test]
fn a_mic_check_leaves_nothing_behind() {
    let mut h = Harness::new();
    h.snapshot(vec![jitsi(100, StreamKind::Mic, true)]).unwrap();
    assert_eq!(h.phase(), Phase::Tentative);
    assert_eq!(h.store.list().unwrap().len(), 1);
    h.tick_until(5_000);
    h.snapshot(Vec::new()).unwrap();
    h.tick_until(10_000);
    assert_eq!(h.phase(), Phase::Idle);
    assert!(h.store.list().unwrap().is_empty());
    assert!(h.sent().is_empty());
    // Nothing of it can be indexed, so the daemon doesn't open the index.
    assert!(!h.store.index_path().exists());
}

#[test]
fn manual_record_and_stop_over_the_daemon_api() {
    let mut h = Harness::new();
    let message = h.ok(Request::Record {
        title: Some("Standup".into()),
    });
    let status = h.daemon.status();
    assert_eq!(status.phase, Phase::Recording);
    assert_eq!(status.app, None);
    let id = status.session.expect("a session");
    assert_eq!(message, format!("recording {id}"));

    // A manual session ignores the graph: no mic does not end it.
    h.snapshot(Vec::new()).unwrap();
    h.tick_until(60_000);
    assert_eq!(h.phase(), Phase::Recording);

    assert_eq!(h.ok(Request::Stop), format!("stopped {id}"));
    assert_eq!(h.phase(), Phase::Idle);
    let meta = h.store.session(&id).unwrap().load_meta().unwrap();
    assert_eq!(meta.title.as_deref(), Some("Standup"));
    assert_eq!(meta.trigger.kind, TriggerKind::Manual);
    assert_eq!(meta.state, SessionState::Processing);
    assert_eq!(h.sent(), vec![Job::Process { id }]);
}

#[test]
fn record_names_a_tentative_session_and_keeps_its_app() {
    let mut h = Harness::new();
    h.snapshot(vec![jitsi(100, StreamKind::Mic, true)]).unwrap();
    let id = h.daemon.status().session.unwrap();
    h.ok(Request::Record {
        title: Some("Planning".into()),
    });
    assert_eq!(h.phase(), Phase::Recording);
    assert_eq!(h.daemon.status().session.as_deref(), Some(id.as_str()));
    let meta = h.store.session(&id).unwrap().load_meta().unwrap();
    assert_eq!(meta.title.as_deref(), Some("Planning"));
    assert_eq!(meta.trigger.kind, TriggerKind::App);
}

#[test]
fn control_requests_that_do_not_apply_are_errors() {
    let mut h = Harness::new();
    assert_eq!(
        h.request(Request::Stop),
        Response::Error {
            message: "nothing is being recorded".into()
        }
    );
    h.ok(Request::Record { title: None });
    let id = h.daemon.status().session.unwrap();
    assert_eq!(
        h.request(Request::Record { title: None }),
        Response::Error {
            message: format!("already recording {id}")
        }
    );
    assert_eq!(h.ok(Request::Discard), format!("discarded {id}"));
    assert!(h.store.list().unwrap().is_empty());
    assert!(h.sent().is_empty());
    assert!(!h.store.index_path().exists());
}

#[test]
fn status_reports_the_queue() {
    let mut h = Harness::new();
    let status = match h.request(Request::Status) {
        Response::Status { status } => status,
        other => panic!("{other:?}"),
    };
    assert_eq!(status, h.daemon.status());
    assert_eq!(
        (status.phase, status.paused, status.queued),
        (Phase::Idle, false, 0)
    );
    assert_eq!(status.running_job, None);

    let title = Job::Title {
        id: "s1".into(),
        title: "Planning".into(),
    };
    assert_eq!(h.ok(Request::Job { job: Job::Reindex }), "queued reindex");
    h.ok(Request::Job { job: title.clone() });
    let status = h.daemon.status();
    assert_eq!(status.running_job.as_deref(), Some("reindex"));
    assert_eq!(status.queued, 1);
    assert_eq!(h.sent(), vec![Job::Reindex]);

    h.done(Job::Reindex, None, None);
    assert_eq!(h.sent(), vec![title.clone()]);
    assert_eq!(h.daemon.status().running_job.as_deref(), Some("title s1"));
    h.done(title, Some("s1"), None);
    assert_eq!(h.daemon.status().running_job, None);
    assert!(h.sent().is_empty());
}

#[test]
fn a_process_job_is_queued_once() {
    let mut h = Harness::new();
    h.ok(Request::Job { job: Job::Reindex });
    for _ in 0..2 {
        h.ok(Request::Job {
            job: Job::Process { id: "s1".into() },
        });
    }
    assert_eq!(h.daemon.status().queued, 1);
}

#[test]
fn a_failed_transcription_is_retried_after_the_backoff() {
    let mut h = Harness::new();
    let import = Job::Import {
        me: "/tmp/me.wav".into(),
        others: "/tmp/others.wav".into(),
        start: None,
        title: None,
        process: true,
    };
    h.ok(Request::Job {
        job: import.clone(),
    });
    assert_eq!(
        h.daemon.status().running_job.as_deref(),
        Some("import /tmp/me.wav")
    );
    h.sent();
    h.now = 1_000;
    h.done(import, Some("s1"), Some(Duration::from_secs(60)));
    assert_eq!(h.daemon.status().retrying, 1);
    h.tick_until(60_500);
    assert!(h.sent().is_empty());
    h.tick_until(61_000);
    assert_eq!(h.daemon.status().retrying, 0);
    // The import is never re-sent: its session is processed again.
    assert_eq!(h.sent(), vec![Job::Process { id: "s1".into() }]);
}

#[test]
fn pause_ends_a_detected_session_and_resume_picks_the_call_up_again() {
    let mut h = Harness::new();
    h.snapshot(vec![jitsi(100, StreamKind::Mic, true)]).unwrap();
    assert_eq!(h.phase(), Phase::Tentative);
    h.ok(Request::Pause);
    let status = h.daemon.status();
    assert_eq!((status.phase, status.paused), (Phase::Idle, true));
    assert!(h.store.list().unwrap().is_empty());

    h.ok(Request::Resume);
    let status = h.daemon.status();
    assert_eq!((status.phase, status.paused), (Phase::Tentative, false));
}

fn call() -> Vec<Stream> {
    vec![
        jitsi(100, StreamKind::Mic, true),
        jitsi(101, StreamKind::Output, true),
    ]
}

#[test]
fn a_pipewire_hiccup_mid_call_keeps_the_session() {
    let mut h = Harness::new();
    h.snapshot(call()).unwrap();
    h.tick_until(20_000);
    assert_eq!(h.phase(), Phase::Recording);
    let id = h.daemon.status().session.unwrap();

    h.disconnect();
    h.tick_until(21_000);
    // The monitor reconnects and the call's mic is still there.
    h.snapshot(call()).unwrap();
    h.tick_until(60_000);
    let status = h.daemon.status();
    assert_eq!(status.phase, Phase::Recording);
    assert_eq!(status.session, Some(id));
    assert!(h.sent().is_empty());
}

#[test]
fn a_pipewire_disconnect_ends_a_detected_session_as_if_the_call_ended() {
    let mut h = Harness::new();
    h.snapshot(call()).unwrap();
    h.disconnect();
    // A Tentative session goes once the gap is confirmed, and is deleted.
    assert_eq!(h.phase(), Phase::Tentative);
    h.tick_until(2_000);
    assert_eq!(h.phase(), Phase::Idle);
    assert!(h.store.list().unwrap().is_empty());

    // A recording one runs out its grace and is processed.
    h.snapshot(call()).unwrap();
    h.tick_until(30_000);
    assert_eq!(h.phase(), Phase::Recording);
    let id = h.daemon.status().session.unwrap();
    h.disconnect();
    h.tick_until(32_000);
    assert_eq!(h.phase(), Phase::Grace);
    h.tick_until(45_000);
    assert_eq!(h.phase(), Phase::Idle);
    assert_eq!(h.sent(), vec![Job::Process { id }]);

    // Not the user's stop: the call still going on when PipeWire is back is recorded.
    h.snapshot(call()).unwrap();
    assert_eq!(h.phase(), Phase::Tentative);
}

#[test]
fn a_pipewire_disconnect_does_not_restart_a_call_the_user_stopped() {
    let mut h = Harness::new();
    h.snapshot(call()).unwrap();
    h.tick_until(30_000);
    let id = h.daemon.status().session.unwrap();
    assert_eq!(h.ok(Request::Stop), format!("stopped {id}"));
    assert_eq!(h.sent(), vec![Job::Process { id }]);

    h.disconnect();
    h.tick_until(60_000);
    // The monitor reconnects and the call the user stopped is still on.
    h.snapshot(call()).unwrap();
    h.tick_until(61_000);
    let status = h.daemon.status();
    assert_eq!(status.phase, Phase::Idle);
    assert_eq!(status.session, None);
    assert_eq!(h.store.list().unwrap().len(), 1);
    assert!(h.sent().is_empty());
}

#[test]
fn a_pipewire_disconnect_leaves_a_manual_session_alone() {
    let mut h = Harness::new();
    h.ok(Request::Record { title: None });
    let id = h.daemon.status().session.unwrap();
    h.disconnect();
    h.tick_until(60_000);
    let status = h.daemon.status();
    assert_eq!(status.phase, Phase::Recording);
    assert_eq!(status.session, Some(id));
}

#[test]
fn the_configured_mic_is_recorded_when_present() {
    let targets = Targets::default();
    let mut config = Config::default();
    config.capture.mic = Some("headset".into());
    let spy = targets.clone();
    let mut h = Harness::with(config, move |inner| {
        Box::new(Spy {
            inner,
            targets: spy,
        })
    });
    h.snapshot(vec![jitsi(100, StreamKind::Mic, true)]).unwrap();
    assert_eq!(
        *targets.lock().unwrap(),
        vec![
            (Track::Me, Some("alsa_input.usb-headset".to_string())),
            (Track::Others, None)
        ]
    );
    let id = h.daemon.status().session.unwrap();
    let meta = h.store.session(&id).unwrap().load_meta().unwrap();
    assert_eq!(
        meta.tracks.me.device.as_deref(),
        Some("alsa_input.usb-headset")
    );
}

/// The target each capture was opened with.
type Targets = Arc<Mutex<Vec<(Track, Option<String>)>>>;

/// Records which target each capture was opened with.
struct Spy {
    inner: FakeSource,
    targets: Targets,
}

impl CaptureSource for Spy {
    fn open(&self, track: Track, target: Option<&str>) -> Result<Capture> {
        self.targets
            .lock()
            .unwrap()
            .push((track, target.map(Into::into)));
        self.inner.open(track, target)
    }
}

struct Broken;

impl CaptureSource for Broken {
    fn open(&self, _track: Track, _target: Option<&str>) -> Result<Capture> {
        bail!("pw-record not found")
    }
}

#[test]
fn a_capture_that_cannot_start_leaves_the_detector_idle() {
    let mut h = Harness::with(Config::default(), |_| Box::new(Broken));
    let err = h
        .snapshot(vec![jitsi(100, StreamKind::Mic, true)])
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("pw-record not found"),
        "{err:#}"
    );
    assert_eq!(h.phase(), Phase::Idle);

    let Response::Error { message } = h.request(Request::Record { title: None }) else {
        panic!("expected an error");
    };
    assert!(message.contains("pw-record not found"), "{message}");
    assert_eq!(h.phase(), Phase::Idle);
    assert!(h.store.list().unwrap().is_empty());
}

/// Captures that deliver one second of audio and end on their own.
struct Short;

impl CaptureSource for Short {
    fn open(&self, _track: Track, _target: Option<&str>) -> Result<Capture> {
        let bytes = pcm(1, 0..0);
        Ok(Capture::new(Box::new(std::io::Cursor::new(bytes)), || {}))
    }
}

#[test]
fn a_capture_that_ends_on_its_own_finishes_the_session() {
    let mut h = Harness::with(Config::default(), |_| Box::new(Short));
    h.ok(Request::Record { title: None });
    let id = h.daemon.status().session.unwrap();
    h.tick_until_ended(&id);
    let meta = h.store.session(&id).unwrap().load_meta().unwrap();
    assert_eq!(meta.state, SessionState::Processing);
    assert_eq!(h.sent(), vec![Job::Process { id }]);
}

#[test]
fn a_capture_that_dies_mid_call_does_not_block_the_call() {
    let mut h = Harness::new();
    h.snapshot(call()).unwrap();
    let id = h.daemon.status().session.unwrap();
    for track in Track::BOTH {
        h.source.writer(track).write_all(&pcm(1, 0..0)).unwrap();
    }
    h.tick_until(30_000);
    assert_eq!(h.phase(), Phase::Recording);
    h.source.close(Track::Me);
    h.tick_until_ended(&id);
    assert_eq!(h.sent(), vec![Job::Process { id: id.clone() }]);

    // pw-record leaving the graph makes a snapshot: the call is still on.
    h.snapshot(call()).unwrap();
    let status = h.daemon.status();
    assert_eq!(status.phase, Phase::Tentative);
    assert_ne!(status.session, Some(id));
}

/// A PipeWire/WirePlumber restart: both captures die and the call app reconnects, all
/// before the daemon notices, so the graph has settled by the time the session ends and no
/// snapshot follows. The call is still on: a new session must start without one (seen on
/// 2026-10-21, where the call went unrecorded until kute was restarted).
#[test]
fn a_capture_that_dies_after_the_graph_settled_starts_a_new_session() {
    let mut h = Harness::new();
    h.snapshot(call()).unwrap();
    let id = h.daemon.status().session.unwrap();
    for track in Track::BOTH {
        h.source.writer(track).write_all(&pcm(1, 0..0)).unwrap();
    }
    h.tick_until(30_000);
    assert_eq!(h.phase(), Phase::Recording);
    // The last snapshot: the call's mic, relinked; kute's captures already gone.
    h.snapshot(call()).unwrap();
    for track in Track::BOTH {
        h.source.close(track);
    }
    h.tick_until_ended(&id);
    assert_eq!(h.sent(), vec![Job::Process { id: id.clone() }]);

    // No snapshot from here on: the graph doesn't change again.
    h.tick_until(h.now + 1_000);
    let status = h.daemon.status();
    assert_eq!(status.phase, Phase::Tentative);
    assert_ne!(status.session, Some(id));
}

/// Captures that end at once, without a sample.
struct Empty;

impl CaptureSource for Empty {
    fn open(&self, _track: Track, _target: Option<&str>) -> Result<Capture> {
        Ok(Capture::new(Box::new(std::io::empty()), || {}))
    }
}

#[test]
fn a_capture_that_dies_at_once_discards_and_blocks_the_call() {
    let mut h = Harness::with(Config::default(), |_| Box::new(Empty));
    h.snapshot(call()).unwrap();
    assert_eq!(h.phase(), Phase::Tentative);
    let id = h.daemon.status().session.unwrap();
    h.tick_until_ended(&id);
    assert_eq!(h.phase(), Phase::Idle);
    // Nothing was recorded: no folder, nothing to process.
    assert!(h.store.list().unwrap().is_empty());
    assert!(h.sent().is_empty());
    // pw-record is likely broken: don't start a session that fails again every tick.
    h.now = 1_000;
    h.snapshot(call()).unwrap();
    assert_eq!(h.phase(), Phase::Idle);
}

/// A manual session with a 10 s conversation gap whose first conversation has closed,
/// so a `Process` for it is running on the worker. Returns the session ID.
fn record_until_a_conversation_closes(h: &mut Harness) -> String {
    h.ok(Request::Record { title: None });
    let id = h.daemon.status().session.unwrap();
    let writers = [
        (Track::Me, pcm(40, 1_000..3_000)),
        (Track::Others, pcm(40, 0..0)),
    ]
    .map(|(track, bytes)| {
        let mut w = h.source.writer(track);
        std::thread::spawn(move || w.write_all(&bytes).unwrap())
    });
    for w in writers {
        w.join().unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut sent = Vec::new();
    while sent.is_empty() {
        assert!(Instant::now() < deadline, "no conversation closed");
        std::thread::sleep(Duration::from_millis(10));
        h.tick_until(h.now + 500);
        sent = h.sent();
    }
    assert_eq!(sent, vec![Job::Process { id: id.clone() }]);
    id
}

fn gap_10s() -> Config {
    let mut config = Config::default();
    config.detect.conversation_gap = jiff::SignedDuration::from_secs(10);
    config
}

#[test]
fn a_conversation_that_closes_mid_call_is_processed_while_recording() {
    let mut h = Harness::with(gap_10s(), |source| Box::new(source));
    let id = record_until_a_conversation_closes(&mut h);
    assert_eq!(h.phase(), Phase::Recording);
    let meta = h.store.session(&id).unwrap().load_meta().unwrap();
    assert_eq!(meta.state, SessionState::Recording);
    assert_eq!(meta.conversations.len(), 1);
}

/// What the worker's pipeline writes to the index for a transcribed conversation.
fn index_a_line(store: &Store, id: &str, text: &str) {
    let index = Index::open(&store.index_path()).unwrap();
    let meta = Meta::new(
        id.into(),
        "2026-10-19T10:00:00+02:00[+02:00]".parse().unwrap(),
        Trigger {
            kind: TriggerKind::Manual,
            app_name: None,
            binary: None,
        },
        "auto".into(),
    );
    index.upsert_session(&meta).unwrap();
    let line = Line {
        start_ms: 1_000,
        track: Track::Me,
        text: text.into(),
    };
    index.replace_lines(id, 1, &[line]).unwrap();
}

fn search(store: &Store, query: &str) -> Vec<String> {
    let index = Index::open(&store.index_path()).unwrap();
    let hits = index.search(query, 10).unwrap();
    hits.into_iter().map(|h| h.session_id).collect()
}

#[test]
fn discard_removes_the_session_from_the_index_and_drops_its_jobs() {
    let mut h = Harness::with(gap_10s(), |source| Box::new(source));
    let id = record_until_a_conversation_closes(&mut h);
    // The worker transcribes conversation 1, indexes it, and fails another one.
    index_a_line(&h.store, &id, "budget");
    assert_eq!(search(&h.store, "budget"), [id.as_str()]);
    let process = Job::Process { id: id.clone() };
    h.done(process.clone(), Some(&id), Some(Duration::from_secs(60)));
    // More work for the session: one job running, one waiting, plus a job for another.
    let summarize = Job::Summarize {
        id: id.clone(),
        conversation: None,
        lang: None,
        backend: None,
    };
    h.ok(Request::Job {
        job: summarize.clone(),
    });
    h.ok(Request::Job {
        job: process.clone(),
    });
    h.ok(Request::Job { job: Job::Reindex });
    assert_eq!(h.sent(), vec![summarize.clone()]);

    assert_eq!(h.ok(Request::Discard), format!("discarded {id}"));
    assert!(search(&h.store, "budget").is_empty());
    let status = h.daemon.status();
    assert_eq!(status.queued, 1);

    // The running job indexed the session again before it noticed the folder was gone.
    index_a_line(&h.store, &id, "budget");
    h.done(summarize, Some(&id), Some(Duration::from_secs(60)));
    assert!(search(&h.store, "budget").is_empty());
    assert_eq!(h.sent(), vec![Job::Reindex]);
    h.done(Job::Reindex, None, None);
    h.tick_until(h.now + 120_000);
    assert!(h.sent().is_empty());
}

#[test]
fn discard_unindexes_a_session_whose_jobs_have_all_finished() {
    let mut h = Harness::with(gap_10s(), |source| Box::new(source));
    let id = record_until_a_conversation_closes(&mut h);
    index_a_line(&h.store, &id, "budget");
    h.done(Job::Process { id: id.clone() }, Some(&id), None);
    assert_eq!(h.daemon.status().running_job, None);
    // No job is left for it, but its closed conversation is in the index.
    assert_eq!(h.ok(Request::Discard), format!("discarded {id}"));
    assert!(search(&h.store, "budget").is_empty());
}

/// A session in the store with one summarized conversation, as the worker leaves it.
fn summarized_session(store: &Store) -> String {
    let start: jiff::Zoned = "2026-10-21T15:15:21+02:00[+02:00]".parse().unwrap();
    let session = store.create_session(&start).unwrap();
    let trigger = Trigger {
        kind: TriggerKind::App,
        app_name: Some("Element".into()),
        binary: Some("element-desktop".into()),
    };
    let mut meta = Meta::new(session.id.clone(), start, trigger, "en".into());
    meta.conversations
        .push(kute::store::Conversation::new(1, 0, 420_000));
    session.save_meta(&meta).unwrap();
    std::fs::create_dir_all(session.summaries_dir()).unwrap();
    std::fs::write(
        session.conversation_summary_path(1),
        "## Summary\n\n- A design problem.\n",
    )
    .unwrap();
    session.id
}

#[test]
fn a_new_summary_pops_up_a_notification_that_opens_it() {
    let mut h = Harness::new();
    let id = summarized_session(&h.store);
    let outcome = Outcome {
        session: Some(id.clone()),
        summarized: vec![1],
        ..Outcome::default()
    };
    h.finished(Job::Process { id: id.clone() }, Ok(outcome), None);
    let notes = h.notes.notes();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0].title, "Summary ready: Element");
    assert_eq!(notes[0].body, "15:15 to 15:22\nA design problem.");
    assert_eq!(notes[0].open, h.store.session(&id).unwrap().summary_path());
}

#[test]
fn jobs_without_new_summaries_stay_quiet() {
    let mut h = Harness::new();
    let id = summarized_session(&h.store);
    // Done, but nothing summarized (e.g. a conversation under min_words).
    h.done(Job::Process { id: id.clone() }, Some(&id), None);
    // Failed.
    h.finished(
        Job::Process { id: id.clone() },
        Err("GPU out of memory".into()),
        None,
    );
    // Summarized, but the session is gone (discarded meanwhile).
    let outcome = Outcome {
        session: Some("2020-01-01T000000".into()),
        summarized: vec![1],
        ..Outcome::default()
    };
    h.finished(
        Job::Process {
            id: "2020-01-01T000000".into(),
        },
        Ok(outcome),
        None,
    );
    assert!(h.notes.notes().is_empty(), "{:?}", h.notes.notes());
}
