//! One recording in progress: both captures, both track threads, and the conversations
//! the splitter finds in their speech. Also repairs sessions a crashed daemon left behind.

use crate::audio::samples_to_ms;
use crate::capture::{CaptureSource, StopHandle};
use crate::config::Config;
use crate::pw::App;
use crate::recorder::{RecorderEvent, TrackOptions, VadFactory, repair_wav_header, spawn_track};
use crate::split::{SplitEvent, Splitter};
use crate::store::{Conversation, Meta, Session, SessionState, Store, Track, Trigger, TriggerKind};
use crate::time;
use anyhow::{Context, Result};
use jiff::Zoned;
use std::sync::mpsc::{self, Receiver};
use std::thread::JoinHandle;

pub struct StartRequest {
    /// The app that triggered the session; `None` for a manual one.
    pub trigger: Option<App>,
    pub title: Option<String>,
    /// A source `node.name` for the Me track; `None` records the default source.
    pub mic_target: Option<String>,
    /// Device names for `meta.toml`: (Me, Others).
    pub devices: (Option<String>, Option<String>),
}

pub struct LiveSession {
    session: Session,
    /// Me, Others.
    stops: [StopHandle; 2],
    threads: Vec<(Track, JoinHandle<()>)>,
    /// Only the track threads send here, so nothing else can take their events.
    rx: Receiver<RecorderEvent>,
    splitter: Splitter,
    /// Wall-clock instant of session time 0: the first sample of either track.
    t0: Option<Zoned>,
    /// How much later than `t0` each track's first sample arrived.
    offsets: [u64; 2],
    /// How far speech detection has covered each track, in session time.
    analysed: [Option<u64>; 2],
    /// Sample counts from `Ended`.
    samples: [Option<u64>; 2],
    /// A track had no live speech detection or lost it, so the splitter can't be trusted
    /// with the audio after the last closed conversation.
    vad_lost: bool,
    /// End of the last closed conversation.
    closed_until: u64,
    /// Set by `finish`/`discard`: from then on a track ending is expected.
    stopping: bool,
    failed: Option<String>,
    next_n: u32,
}

fn idx(track: Track) -> usize {
    match track {
        Track::Me => 0,
        Track::Others => 1,
    }
}

impl LiveSession {
    /// Creates the session folder and meta, then starts recording both tracks.
    pub fn start(
        store: &Store,
        config: &Config,
        req: StartRequest,
        source: &dyn CaptureSource,
        vad: VadFactory,
    ) -> Result<LiveSession> {
        let now = time::now();
        let session = store.create_session(&now)?;
        match Self::start_in(session.clone(), now, config, req, source, vad) {
            Ok(live) => Ok(live),
            Err(e) => {
                if let Err(e) = session.delete() {
                    eprintln!("kute: {e:#}");
                }
                Err(e)
            }
        }
    }

    /// `now` stands in for `started_at` until the first sample arrives.
    fn start_in(
        session: Session,
        now: Zoned,
        config: &Config,
        req: StartRequest,
        source: &dyn CaptureSource,
        vad: VadFactory,
    ) -> Result<LiveSession> {
        let trigger = match req.trigger {
            Some(app) => Trigger {
                kind: TriggerKind::App,
                app_name: app.name,
                binary: app.binary,
            },
            None => Trigger {
                kind: TriggerKind::Manual,
                app_name: None,
                binary: None,
            },
        };
        let mut meta = Meta::new(
            session.id.clone(),
            now,
            trigger,
            config.summary.language.clone(),
        );
        meta.title = req.title;
        meta.tracks.me.device = req.devices.0;
        meta.tracks.others.device = req.devices.1;
        session.save_meta(&meta)?;

        // Dropping `me` on an error stops it.
        let me = source
            .open(Track::Me, req.mic_target.as_deref())
            .context("starting the microphone capture")?;
        let others = source
            .open(Track::Others, None)
            .context("starting the speaker capture")?;
        let stops = [me.stop_handle(), others.stop_handle()];
        let (tx, rx) = mpsc::channel();
        let threads = [(Track::Me, me), (Track::Others, others)]
            .map(|(track, capture)| {
                let handle = spawn_track(
                    track,
                    capture,
                    session.wav_path(track),
                    vad.clone(),
                    TrackOptions { gap_fill: true },
                    tx.clone(),
                );
                (track, handle)
            })
            .into();
        let gap_ms = config.detect.conversation_gap.as_millis().max(0) as u64;
        Ok(LiveSession {
            session,
            stops,
            threads,
            rx,
            splitter: Splitter::new(gap_ms),
            t0: None,
            offsets: [0; 2],
            analysed: [None; 2],
            samples: [None; 2],
            vad_lost: false,
            closed_until: 0,
            stopping: false,
            failed: None,
            next_n: 1,
        })
    }

    pub fn id(&self) -> &str {
        &self.session.id
    }

    /// Handles the events the tracks have sent so far. Returns the numbers of the
    /// conversations that just closed (to enqueue). The daemon calls it every tick.
    pub fn pump(&mut self) -> Result<Vec<u32>> {
        let mut closed = Vec::new();
        while let Ok(ev) = self.rx.try_recv() {
            closed.extend(self.on_event(ev)?);
        }
        Ok(closed)
    }

    /// Events must come in the order each track sent them: a track sends a block's
    /// `Speech` before its `Analysed`, which is what `Splitter` needs.
    fn on_event(&mut self, ev: RecorderEvent) -> Result<Vec<u32>> {
        match ev {
            RecorderEvent::FirstSample { track, at, vad } => {
                self.vad_lost |= !vad;
                match &self.t0 {
                    None => {
                        self.session.update_meta(|m| m.started_at = at.clone())?;
                        self.t0 = Some(at);
                    }
                    // The threads race to send, so the earlier sample can arrive second: it
                    // becomes session time 0. Only ever a few ms, before any speech.
                    Some(t0) if at < *t0 => {
                        let shift = time::ms_between(&at, t0);
                        for t in Track::BOTH {
                            self.offsets[idx(t)] = if t == track { 0 } else { shift };
                        }
                        let offsets = self.offsets;
                        self.session.update_meta(|m| {
                            m.started_at = at.clone();
                            for t in Track::BOTH {
                                m.tracks.get_mut(t).offset_ms = offsets[idx(t)];
                            }
                        })?;
                        self.t0 = Some(at);
                    }
                    Some(t0) => {
                        let offset = time::ms_between(t0, &at);
                        self.offsets[idx(track)] = offset;
                        self.session
                            .update_meta(|m| m.tracks.get_mut(track).offset_ms = offset)?;
                    }
                }
            }
            RecorderEvent::Speech {
                track,
                start_ms,
                end_ms,
            } => {
                let offset = self.offsets[idx(track)];
                if let Some(ev) = self.splitter.speech(start_ms + offset, end_ms + offset) {
                    return self.on_split(ev);
                }
            }
            RecorderEvent::Analysed { track, until_ms } => {
                self.analysed[idx(track)] = Some(until_ms + self.offsets[idx(track)]);
                // A track that hasn't analysed anything yet holds the other back.
                if let [Some(a), Some(b)] = self.analysed
                    && let Some(ev) = self.splitter.advance(a.min(b))
                {
                    return self.on_split(ev);
                }
            }
            RecorderEvent::VadLost { .. } => self.vad_lost = true,
            RecorderEvent::Ended { track, samples } => {
                self.samples[idx(track)] = Some(samples);
                if !self.stopping {
                    self.fail(format!("capture of {} ended unexpectedly", track.as_str()));
                }
            }
            RecorderEvent::Failed { track, error } => {
                self.fail(format!("recording the {} track: {error}", track.as_str()));
            }
        }
        Ok(Vec::new())
    }

    fn fail(&mut self, error: String) {
        eprintln!("kute: {}: {error}", self.session.id);
        self.failed.get_or_insert(error);
    }

    fn on_split(&mut self, ev: SplitEvent) -> Result<Vec<u32>> {
        match ev {
            SplitEvent::Opened { start_ms } => {
                self.session
                    .update_meta(|m| m.open_conversation_ms = Some(start_ms))?;
                Ok(Vec::new())
            }
            SplitEvent::Closed { start_ms, end_ms } => {
                let n = self.next_n;
                self.session.update_meta(|m| {
                    m.conversations.push(Conversation::new(n, start_ms, end_ms));
                    m.open_conversation_ms = None;
                })?;
                self.next_n += 1;
                self.closed_until = end_ms;
                Ok(vec![n])
            }
        }
    }

    /// Whether either track has delivered a sample yet.
    pub fn has_audio(&self) -> bool {
        self.t0.is_some()
    }

    /// Whether a conversation has closed, so the worker may have transcribed and indexed it.
    pub fn has_closed_conversation(&self) -> bool {
        self.next_n > 1
    }

    /// Set when a track failed or its capture ended on its own; the daemon then finishes
    /// the session.
    pub fn failed(&self) -> Option<&str> {
        self.failed.as_deref()
    }

    /// Stops both captures and waits for the track threads, which send everything they
    /// have before they exit.
    fn stop(&mut self) {
        self.stopping = true;
        self.stop_captures();
        for (track, handle) in self.threads.drain(..) {
            if handle.join().is_err() {
                eprintln!("kute: the {} track thread panicked", track.as_str());
            }
        }
    }

    /// Stops recording, closes the last conversation and hands the session over to
    /// processing.
    pub fn finish(mut self) -> Result<Session> {
        self.stop();
        // The threads are gone, so every event they sent is already in the channel.
        let events: Vec<RecorderEvent> = self.rx.try_iter().collect();
        for ev in events {
            self.on_event(ev)?;
        }
        let end = Track::BOTH
            .into_iter()
            .map(|t| {
                let samples = self.samples[idx(t)].unwrap_or(0);
                self.offsets[idx(t)] + samples_to_ms(samples as usize)
            })
            .max()
            .unwrap_or(0);
        let last = if self.vad_lost {
            // Speech detection doesn't cover what came after: it is all one conversation.
            (self.closed_until < end).then_some((self.closed_until, end))
        } else {
            match self.splitter.finish(end) {
                Some(SplitEvent::Closed { start_ms, end_ms }) => Some((start_ms, end_ms)),
                _ => None,
            }
        };
        let n = self.next_n;
        self.session.update_meta(|m| {
            if let Some((start_ms, end_ms)) = last {
                m.conversations.push(Conversation::new(n, start_ms, end_ms));
            }
            m.open_conversation_ms = None;
            m.ended_at = Some(time::add_ms(&m.started_at, end));
            m.state = SessionState::Processing;
        })?;
        Ok(self.session.clone())
    }

    /// Stops recording and deletes the session.
    pub fn discard(mut self) -> Result<()> {
        self.stop();
        self.session.clone().delete()
    }

    fn stop_captures(&self) {
        for stop in &self.stops {
            stop.stop();
        }
    }
}

/// A session dropped without `finish` or `discard` (after an error) still stops its
/// captures, so nothing keeps recording without an owner. The track threads then exit
/// on their own; their events go nowhere.
impl Drop for LiveSession {
    fn drop(&mut self) {
        self.stop_captures();
    }
}

/// Repairs the sessions a crashed daemon left in `recording`: fixes the WAV headers,
/// closes the open conversation at the end of the audio and moves them to processing.
pub fn recover(store: &Store) -> Result<Vec<Session>> {
    let mut recovered = Vec::new();
    for session in store.list()? {
        let meta = match session.load_meta() {
            Ok(meta) => meta,
            Err(e) => {
                eprintln!("kute: skipping {}: {e:#}", session.id);
                continue;
            }
        };
        if meta.state != SessionState::Recording {
            continue;
        }
        let mut end = 0;
        for track in Track::BOTH {
            let path = session.wav_path(track);
            if !path.is_file() {
                continue;
            }
            match repair_wav_header(&path) {
                Ok(samples) => {
                    let ms = meta.tracks.get(track).offset_ms + samples_to_ms(samples as usize);
                    end = end.max(ms);
                }
                Err(e) => eprintln!("kute: {}: {e:#}", session.id),
            }
        }
        let updated = session.update_meta(|m| {
            m.interrupted = true;
            match m.open_conversation_ms.take() {
                Some(start_ms) if start_ms < end => {
                    let n = m.conversations.iter().map(|c| c.n).max().unwrap_or(0) + 1;
                    m.conversations.push(Conversation::new(n, start_ms, end));
                }
                Some(start_ms) => eprintln!(
                    "kute: {}: dropping the open conversation at {start_ms} ms, after the \
                     end of the audio ({end} ms)",
                    session.id
                ),
                // Recorded without live speech detection: the whole session is one.
                None if m.conversations.is_empty() && end > 0 => {
                    m.conversations.push(Conversation::new(1, 0, end));
                }
                None => {}
            }
            m.ended_at = Some(time::add_ms(&m.started_at, end));
            m.state = SessionState::Processing;
        });
        if let Err(e) = updated {
            eprintln!("kute: could not recover {}: {e:#}", session.id);
            continue;
        }
        eprintln!("kute: recovered interrupted session {}", session.id);
        recovered.push(session);
    }
    Ok(recovered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{Capture, CaptureSource};
    use crate::config::{Config, DetectConfig};
    use crate::recorder::{RecorderEvent, TrackWriter};
    use crate::store::{
        Conversation, Meta, SessionState, Stage, Store, Track, Trigger, TriggerKind,
    };
    use crate::testing::{FakeSource, RunVad, crash_wav, run_vad};
    use crate::transcribe::{Span, SpeechDetector};
    use anyhow::{Result, bail};
    use jiff::{SignedDuration, Zoned};
    use std::io::Write;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn store() -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        (tmp, store)
    }

    fn config(gap_secs: i64) -> Config {
        Config {
            detect: DetectConfig {
                conversation_gap: SignedDuration::from_secs(gap_secs),
                ..DetectConfig::default()
            },
            ..Config::default()
        }
    }

    fn manual() -> StartRequest {
        StartRequest {
            trigger: None,
            title: Some("Planning".into()),
            mic_target: None,
            devices: (Some("mic".into()), Some("speakers.monitor".into())),
        }
    }

    /// `seconds` of S16LE PCM, non-zero (speech for `RunVad`) inside `speech`.
    fn pcm(seconds: u64, speech: std::ops::Range<u64>) -> Vec<u8> {
        (0..seconds * 16_000)
            .flat_map(|i| {
                let v: i16 = if speech.contains(&(i / 16)) { 1_000 } else { 0 };
                v.to_le_bytes()
            })
            .collect()
    }

    /// Pumps until `done` holds, collecting the conversations that closed.
    fn pump_until(
        live: &mut LiveSession,
        closed: &mut Vec<u32>,
        done: impl Fn(&LiveSession) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !done(live) {
            closed.extend(live.pump().unwrap());
            assert!(Instant::now() < deadline, "timed out pumping");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn a_session_with_two_chats_becomes_two_conversations() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let mut live =
            LiveSession::start(&store, &config(30), manual(), &source, run_vad()).unwrap();
        let me = pcm(100, 1_000..3_000);
        let others = pcm(100, 60_000..62_000);
        const CHUNK: usize = 32_000;
        // Me's first sample comes first, so it is session time 0.
        let mut me_w = source.writer(Track::Me);
        me_w.write_all(&me[..CHUNK]).unwrap();
        let mut closed = Vec::new();
        pump_until(&mut live, &mut closed, |l| l.t0.is_some());
        // Pipes hold 64 KiB: feed both tracks in step from another thread.
        let mut others_w = source.writer(Track::Others);
        let writer = std::thread::spawn(move || {
            let mut me_rest = me[CHUNK..].chunks(CHUNK);
            for chunk in others.chunks(CHUNK) {
                others_w.write_all(chunk).unwrap();
                if let Some(chunk) = me_rest.next() {
                    me_w.write_all(chunk).unwrap();
                }
            }
        });
        pump_until(&mut live, &mut closed, |l| {
            l.analysed.iter().all(|a| a.is_some_and(|a| a >= 100_000))
        });
        writer.join().unwrap();
        assert_eq!(closed, [1, 2]);
        assert_eq!(live.failed(), None);

        let session = live.finish().unwrap();
        let meta = session.load_meta().unwrap();
        assert_eq!(meta.conversations.len(), 2, "{:?}", meta.conversations);
        let (c1, c2) = (&meta.conversations[0], &meta.conversations[1]);
        assert_eq!((c1.n, c1.start_ms, c1.end_ms), (1, 0, 5_000));
        assert_eq!(c2.n, 2);
        assert!(c2.start_ms.abs_diff(58_000) <= 50, "{c2:?}");
        assert!(c2.end_ms.abs_diff(64_000) <= 50, "{c2:?}");
        assert!(meta.conversations.iter().all(|c| c.stage == Stage::Pending));
        assert_eq!(meta.state, SessionState::Processing);
        assert_eq!(meta.open_conversation_ms, None);
        let duration = meta.duration_ms().unwrap();
        assert!(duration.abs_diff(100_000) <= 50, "{duration}");
        assert_eq!(meta.title.as_deref(), Some("Planning"));
        assert_eq!(meta.trigger.kind, TriggerKind::Manual);
        assert_eq!(meta.tracks.me.device.as_deref(), Some("mic"));
        assert_eq!(
            meta.tracks.others.device.as_deref(),
            Some("speakers.monitor")
        );
        assert!(!meta.interrupted);
    }

    #[test]
    fn offsets_come_from_first_samples() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let mut live =
            LiveSession::start(&store, &config(30), manual(), &source, run_vad()).unwrap();
        let me_at: Zoned = "2026-10-19T10:00:00.100+02:00[+02:00]".parse().unwrap();
        let others_at = crate::time::add_ms(&me_at, 37);
        live.on_event(RecorderEvent::FirstSample {
            track: Track::Me,
            at: me_at.clone(),
            vad: true,
        })
        .unwrap();
        live.on_event(RecorderEvent::FirstSample {
            track: Track::Others,
            at: others_at,
            vad: true,
        })
        .unwrap();
        let meta = store.session(live.id()).unwrap().load_meta().unwrap();
        assert_eq!(meta.started_at, me_at);
        assert_eq!(meta.tracks.me.offset_ms, 0);
        assert_eq!(meta.tracks.others.offset_ms, 37);
        live.discard().unwrap();
    }

    #[test]
    fn an_earlier_first_sample_arriving_later_becomes_session_time_0() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let mut live =
            LiveSession::start(&store, &config(30), manual(), &source, run_vad()).unwrap();
        let others_at: Zoned = "2026-10-19T10:00:00.100+02:00[+02:00]".parse().unwrap();
        // Me's thread sent first, but its sample came 37 ms later.
        live.on_event(RecorderEvent::FirstSample {
            track: Track::Me,
            at: crate::time::add_ms(&others_at, 37),
            vad: true,
        })
        .unwrap();
        live.on_event(RecorderEvent::FirstSample {
            track: Track::Others,
            at: others_at.clone(),
            vad: true,
        })
        .unwrap();
        assert_eq!(live.offsets, [37, 0]);
        let meta = store.session(live.id()).unwrap().load_meta().unwrap();
        assert_eq!(meta.started_at, others_at);
        assert_eq!(meta.tracks.me.offset_ms, 37);
        assert_eq!(meta.tracks.others.offset_ms, 0);
        live.discard().unwrap();
    }

    #[test]
    fn dropping_a_session_stops_its_captures() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let live = LiveSession::start(&store, &config(30), manual(), &source, run_vad()).unwrap();
        drop(live);
        for track in Track::BOTH {
            let err = source.writer(track).write_all(&[0; 2]).unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::BrokenPipe);
        }
    }

    #[test]
    fn opening_a_conversation_is_saved_for_crash_recovery() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let mut live =
            LiveSession::start(&store, &config(30), manual(), &source, run_vad()).unwrap();
        let at = crate::time::now();
        for track in Track::BOTH {
            live.on_event(RecorderEvent::FirstSample {
                track,
                at: at.clone(),
                vad: true,
            })
            .unwrap();
        }
        live.on_event(RecorderEvent::Speech {
            track: Track::Others,
            start_ms: 12_000,
            end_ms: 13_000,
        })
        .unwrap();
        let session = store.session(live.id()).unwrap();
        assert_eq!(
            session.load_meta().unwrap().open_conversation_ms,
            Some(10_000)
        );
        live.discard().unwrap();
    }

    #[test]
    fn without_live_vad_the_session_is_one_conversation() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let live =
            LiveSession::start(&store, &config(30), manual(), &source, Arc::new(|| None)).unwrap();
        // Only Me delivers audio, so its first sample is session time 0.
        source
            .writer(Track::Me)
            .write_all(&pcm(10, 0..10_000))
            .unwrap();
        let session = live.finish().unwrap();
        let meta = session.load_meta().unwrap();
        assert_eq!(meta.conversations, [Conversation::new(1, 0, 10_000)]);
        assert_eq!(meta.duration_ms(), Some(10_000));
        assert_eq!(meta.state, SessionState::Processing);
    }

    /// `RunVad` for the first `ok` blocks, then errors, like a Silero that breaks.
    struct FailAfter {
        ok: usize,
    }

    impl SpeechDetector for FailAfter {
        fn speech_regions(&mut self, samples: &[f32]) -> Result<Vec<Span>> {
            if self.ok == 0 {
                bail!("speech detection broke");
            }
            self.ok -= 1;
            RunVad.speech_regions(samples)
        }
    }

    fn fail_after(ok: usize) -> VadFactory {
        Arc::new(move || Some(Box::new(FailAfter { ok }) as Box<dyn SpeechDetector>))
    }

    /// Writes `me` and `others` from threads (pipes hold 64 KiB) and waits for both.
    fn write_both(source: &FakeSource, me: Vec<u8>, others: Vec<u8>) {
        let writers = [(Track::Me, me), (Track::Others, others)].map(|(track, bytes)| {
            let mut w = source.writer(track);
            std::thread::spawn(move || w.write_all(&bytes).unwrap())
        });
        for w in writers {
            w.join().unwrap();
        }
    }

    #[test]
    fn a_session_whose_live_vad_fails_at_once_is_one_conversation() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let live =
            LiveSession::start(&store, &config(30), manual(), &source, fail_after(0)).unwrap();
        write_both(&source, pcm(25, 1_000..3_000), pcm(25, 15_000..17_000));
        let session = live.finish().unwrap();
        let meta = session.load_meta().unwrap();
        let end = meta.duration_ms().unwrap();
        assert!(end >= 25_000, "{end}");
        assert_eq!(meta.conversations, [Conversation::new(1, 0, end)]);
    }

    #[test]
    fn a_track_that_gets_no_live_vad_makes_the_session_one_conversation() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        // Only the first detector loads (the model went away, say).
        let loaded = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let vad: VadFactory = Arc::new(move || {
            let first = !loaded.swap(true, std::sync::atomic::Ordering::SeqCst);
            first.then(|| Box::new(RunVad) as Box<dyn SpeechDetector>)
        });
        let live = LiveSession::start(&store, &config(30), manual(), &source, vad).unwrap();
        write_both(&source, pcm(25, 1_000..3_000), pcm(25, 15_000..17_000));
        let session = live.finish().unwrap();
        let meta = session.load_meta().unwrap();
        let end = meta.duration_ms().unwrap();
        assert_eq!(meta.conversations, [Conversation::new(1, 0, end)]);
    }

    #[test]
    fn after_live_vad_is_lost_the_rest_is_one_conversation() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        // Block 1 (0-10 s) closes conversation 1; block 2 fails on both tracks.
        let live =
            LiveSession::start(&store, &config(5), manual(), &source, fail_after(1)).unwrap();
        write_both(&source, pcm(25, 1_000..3_000), pcm(25, 15_000..17_000));
        let session = live.finish().unwrap();
        let meta = session.load_meta().unwrap();
        let end = meta.duration_ms().unwrap();
        assert_eq!(meta.conversations.len(), 2, "{:?}", meta.conversations);
        let (c1, c2) = (&meta.conversations[0], &meta.conversations[1]);
        assert_eq!(c1.start_ms, 0);
        assert!(c1.end_ms.abs_diff(5_000) <= 50, "{c1:?}");
        assert_eq!((c2.n, c2.start_ms, c2.end_ms), (2, c1.end_ms, end));
    }

    #[test]
    fn discard_removes_the_folder() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let live = LiveSession::start(&store, &config(30), manual(), &source, run_vad()).unwrap();
        source
            .writer(Track::Others)
            .write_all(&pcm(1, 0..0))
            .unwrap();
        let dir = store.session(live.id()).unwrap().dir;
        assert!(dir.join("meta.toml").is_file());
        live.discard().unwrap();
        assert!(!dir.exists());
    }

    #[test]
    fn a_track_that_ends_before_finish_is_a_failure() {
        let (_tmp, store) = store();
        let source = FakeSource::default();
        let mut live =
            LiveSession::start(&store, &config(30), manual(), &source, run_vad()).unwrap();
        assert_eq!(live.failed(), None);
        live.on_event(RecorderEvent::Ended {
            track: Track::Others,
            samples: 0,
        })
        .unwrap();
        assert_eq!(live.failed(), Some("capture of others ended unexpectedly"));
        live.discard().unwrap();

        let mut live =
            LiveSession::start(&store, &config(30), manual(), &source, run_vad()).unwrap();
        live.on_event(RecorderEvent::Failed {
            track: Track::Me,
            error: "disk full".into(),
        })
        .unwrap();
        assert!(live.failed().unwrap().contains("disk full"));
        live.discard().unwrap();
    }

    /// Opens Me from a `FakeSource` and fails on Others.
    struct OthersFails(FakeSource);

    impl CaptureSource for OthersFails {
        fn open(&self, track: Track, target: Option<&str>) -> Result<Capture> {
            match track {
                Track::Me => self.0.open(track, target),
                Track::Others => bail!("no sink"),
            }
        }
    }

    #[test]
    fn a_failed_capture_stops_the_other_and_removes_the_folder() {
        let (_tmp, store) = store();
        let source = OthersFails(FakeSource::default());
        let err = LiveSession::start(&store, &config(30), manual(), &source, run_vad())
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("no sink"), "{err:#}");
        assert_eq!(std::fs::read_dir(store.sessions_dir()).unwrap().count(), 0);
        // Me's capture was stopped: its pipe is closed.
        assert!(source.0.writer(Track::Me).write_all(&[0; 2]).is_err());
    }

    #[test]
    fn recover_repairs_an_interrupted_session() {
        let (_tmp, store) = store();
        let started: Zoned = "2026-10-19T10:00:00+02:00[+02:00]".parse().unwrap();
        let session = store.create_session(&started).unwrap();
        let mut meta = Meta::new(
            session.id.clone(),
            started.clone(),
            Trigger {
                kind: TriggerKind::Manual,
                app_name: None,
                binary: None,
            },
            "auto".into(),
        );
        meta.open_conversation_ms = Some(1_000);
        meta.tracks.others.offset_ms = 40;
        session.save_meta(&meta).unwrap();
        // Me: 3_125 ms. Others: 40 + 3_100 ms, the longer one.
        for (track, samples) in [(Track::Me, 50_000), (Track::Others, 49_600)] {
            let path = session.wav_path(track);
            let mut w = TrackWriter::create(&path).unwrap();
            w.write(&vec![7; samples]).unwrap();
            w.finish().unwrap();
            crash_wav(&path);
        }
        // A session that finished normally is left alone.
        let done_at: Zoned = "2026-10-19T09:00:00+02:00[+02:00]".parse().unwrap();
        let done = store.create_session(&done_at).unwrap();
        let mut done_meta = Meta::new(
            done.id.clone(),
            done_at,
            meta.trigger.clone(),
            "auto".into(),
        );
        done_meta.state = SessionState::Processing;
        done.save_meta(&done_meta).unwrap();

        let recovered = recover(&store).unwrap();
        assert_eq!(
            recovered.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            [session.id.as_str()]
        );
        let meta = session.load_meta().unwrap();
        assert!(meta.interrupted);
        assert_eq!(meta.state, SessionState::Processing);
        assert_eq!(meta.conversations, [Conversation::new(1, 1_000, 3_140)]);
        assert_eq!(meta.open_conversation_ms, None);
        assert_eq!(meta.duration_ms(), Some(3_140));
        for (track, samples) in [(Track::Me, 50_000), (Track::Others, 49_600)] {
            let wav = hound::WavReader::open(session.wav_path(track)).unwrap();
            assert_eq!(wav.duration(), samples);
        }
        assert_eq!(done.load_meta().unwrap(), done_meta);
    }

    fn crashed_session(store: &Store, at: &str, me_samples: usize) -> crate::store::Session {
        let started: Zoned = at.parse().unwrap();
        let session = store.create_session(&started).unwrap();
        let meta = Meta::new(
            session.id.clone(),
            started,
            Trigger {
                kind: TriggerKind::Manual,
                app_name: None,
                binary: None,
            },
            "auto".into(),
        );
        session.save_meta(&meta).unwrap();
        if me_samples > 0 {
            let path = session.wav_path(Track::Me);
            let mut w = TrackWriter::create(&path).unwrap();
            w.write(&vec![7; me_samples]).unwrap();
            w.finish().unwrap();
            crash_wav(&path);
        }
        session
    }

    #[test]
    fn recover_makes_a_session_without_live_vad_one_conversation() {
        let (_tmp, store) = store();
        let session = crashed_session(&store, "2026-10-19T10:00:00+02:00[+02:00]", 32_000);
        recover(&store).unwrap();
        let meta = session.load_meta().unwrap();
        assert_eq!(meta.conversations, [Conversation::new(1, 0, 2_000)]);
        assert_eq!(meta.conversations[0].stage, Stage::Pending);
        assert_eq!(meta.state, SessionState::Processing);
    }

    #[test]
    fn recover_skips_a_session_it_cannot_update() {
        use std::os::unix::fs::PermissionsExt;
        let (_tmp, store) = store();
        let stuck = crashed_session(&store, "2026-10-19T09:00:00+02:00[+02:00]", 0);
        let fine = crashed_session(&store, "2026-10-19T10:00:00+02:00[+02:00]", 16_000);
        // `update_meta` can't create its temp file in a read-only folder.
        let mode = |m| std::fs::Permissions::from_mode(m);
        std::fs::set_permissions(&stuck.dir, mode(0o500)).unwrap();
        let recovered = recover(&store);
        std::fs::set_permissions(&stuck.dir, mode(0o700)).unwrap();
        let recovered = recovered.unwrap();
        assert_eq!(
            recovered.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            [fine.id.as_str()]
        );
        assert_eq!(stuck.load_meta().unwrap().state, SessionState::Recording);
        assert_eq!(fine.load_meta().unwrap().state, SessionState::Processing);
    }
}
