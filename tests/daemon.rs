//! The daemon end to end, driven through `Daemon::handle` with a fake clock: no PipeWire,
//! no models. Captures are `FakeSource` pipes and speech detection is `RunVad`.

use anyhow::{Result, bail};
use kute::capture::{Capture, CaptureSource};
use kute::config::Config;
use kute::daemon::{Daemon, Event};
use kute::detect::Phase;
use kute::ipc::{Incoming, Request, Response};
use kute::jobs::{Job, Outcome};
use kute::pw::{App, MonitorEvent, Snapshot, Source, Stream, StreamKind};
use kute::store::{SessionState, Store, Track, TriggerKind};
use kute::testing::{FakeSource, run_vad};
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
        let daemon = Daemon::new(store.clone(), config, source(fake.clone()), run_vad(), tx);
        Harness {
            _tmp: tmp,
            store,
            source: fake,
            daemon,
            worker,
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
        let done = WorkerDone {
            job,
            result: Ok(Outcome {
                message: String::new(),
                session: session.map(Into::into),
            }),
            retry_after,
            session: session.map(Into::into),
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
    h.tick_until(60_500);
    assert!(h.sent().is_empty());
    h.tick_until(61_000);
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

#[test]
fn a_pipewire_disconnect_ends_the_session() {
    let mut h = Harness::new();
    h.snapshot(vec![jitsi(100, StreamKind::Mic, true)]).unwrap();
    h.daemon
        .handle(
            Event::Monitor(MonitorEvent::Disconnected("gone".into())),
            500,
        )
        .unwrap();
    assert_eq!(h.phase(), Phase::Idle);
    // A Tentative session is deleted.
    assert!(h.store.list().unwrap().is_empty());

    h.ok(Request::Record { title: None });
    let id = h.daemon.status().session.unwrap();
    h.daemon
        .handle(
            Event::Monitor(MonitorEvent::Disconnected("gone".into())),
            1_000,
        )
        .unwrap();
    assert_eq!(h.phase(), Phase::Idle);
    assert_eq!(h.sent(), vec![Job::Process { id }]);
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
    let deadline = Instant::now() + Duration::from_secs(30);
    while h.phase() != Phase::Idle {
        assert!(Instant::now() < deadline, "the session never finished");
        std::thread::sleep(Duration::from_millis(10));
        h.tick_until(h.now + 500);
    }
    let meta = h.store.session(&id).unwrap().load_meta().unwrap();
    assert_eq!(meta.state, SessionState::Processing);
    assert_eq!(h.sent(), vec![Job::Process { id }]);
}

#[test]
fn a_conversation_that_closes_mid_call_is_processed_while_recording() {
    let mut config = Config::default();
    config.detect.conversation_gap = jiff::SignedDuration::from_secs(10);
    let mut h = Harness::with(config, |source| Box::new(source));
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
    assert_eq!(h.phase(), Phase::Recording);
    let meta = h.store.session(&id).unwrap().load_meta().unwrap();
    assert_eq!(meta.state, SessionState::Recording);
    assert_eq!(meta.conversations.len(), 1);
}
