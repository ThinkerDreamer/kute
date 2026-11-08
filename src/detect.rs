//! Decides when a call starts and ends, from PipeWire stream snapshots (spec section 5).
//!
//! Pure: no I/O, and time is the caller's monotonic clock in milliseconds.

use crate::config::DetectConfig;
use crate::pw::{App, Stream, StreamKind};
use jiff::SignedDuration;
use std::collections::BTreeSet;
use std::fmt;

/// A call that drops out for less than this is still alive: the Jitsi Meet app closes its
/// pre-join mic stream and opens a new one on joining.
pub const MIC_GAP_MS: u64 = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Idle,
    Tentative,
    Recording,
    Grace,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Start capturing. `None` is a manual session.
    Start { trigger: Option<App> },
    /// Tentative becomes Recording.
    Commit,
    /// Finalize the capture and enqueue it.
    Finish,
    /// Delete the capture.
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectError {
    AlreadyRecording,
    NotRecording,
}

impl fmt::Display for DetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DetectError::AlreadyRecording => "already recording",
            DetectError::NotRecording => "nothing is being recorded",
        })
    }
}

impl std::error::Error for DetectError {}

pub struct Detector {
    min_duration_ms: u64,
    stop_grace_ms: u64,
    /// Lowercased.
    ignore_apps: Vec<String>,
    /// Lowercased.
    allow_apps: Option<Vec<String>>,
    own_pid: u32,
    end_manual: bool,
    phase: Phase,
    manual: bool,
    /// A manual session that follows a call: it ends with the call, like an automatic one.
    following: bool,
    /// During a manual session not yet following a call, since when a mic has been live.
    mic_since: Option<u64>,
    paused: bool,
    trigger: Option<App>,
    /// Serials of the trigger app's output streams seen while a mic stream was alive.
    bound: BTreeSet<u64>,
    /// While Idle, the running output streams of the last snapshot without a live mic. They
    /// were playing before the call, such as another browser tab, so they are never bound.
    before_call: BTreeSet<u64>,
    /// The last snapshot, so `tick` can judge liveness between snapshots.
    last: Vec<Stream>,
    started_ms: u64,
    /// When the session stopped being alive. Grace counts from here too.
    dead_since: Option<u64>,
    /// Set by `stop` and `discard`: no automatic Start until the mic has been gone for
    /// `MIC_GAP_MS`, so a call the user just stopped doesn't start again at once.
    blocked: bool,
    /// When the blocked detector last saw no mic stream.
    quiet_since: Option<u64>,
    /// Set by `graph_lost` until the next snapshot: the empty `last` only means the
    /// streams are unknown, so it doesn't count as quiet time toward lifting `blocked`.
    lost: bool,
}

fn ms(d: SignedDuration) -> u64 {
    d.as_millis().clamp(0, u64::MAX as i128) as u64
}

fn lowercase(rules: &[String]) -> Vec<String> {
    rules.iter().map(|r| r.to_lowercase()).collect()
}

/// An app matches a rule when its name or its binary does, case-insensitively.
fn app_matches(app: &App, rules: &[String]) -> bool {
    [&app.name, &app.binary]
        .into_iter()
        .flatten()
        .any(|v| rules.contains(&v.to_lowercase()))
}

impl Detector {
    pub fn new(config: &DetectConfig, own_pid: u32) -> Detector {
        Detector {
            min_duration_ms: ms(config.min_duration),
            stop_grace_ms: ms(config.stop_grace),
            ignore_apps: lowercase(&config.ignore_apps),
            allow_apps: config.allow_apps.as_deref().map(lowercase),
            own_pid,
            end_manual: config.end_manual,
            phase: Phase::Idle,
            manual: false,
            following: false,
            mic_since: None,
            paused: false,
            trigger: None,
            bound: BTreeSet::new(),
            before_call: BTreeSet::new(),
            last: Vec::new(),
            started_ms: 0,
            dead_since: None,
            blocked: false,
            quiet_since: None,
            lost: false,
        }
    }

    /// A new snapshot of the graph's streams. Ignored entirely while paused.
    pub fn streams(&mut self, streams: &[Stream], now_ms: u64) -> Vec<Action> {
        if self.paused {
            return Vec::new();
        }
        self.lost = false;
        self.last = streams.to_vec();
        if self.phase == Phase::Idle {
            if self.live_mics().next().is_none() {
                self.before_call = self
                    .outputs()
                    .filter(|s| s.running)
                    .map(|s| s.serial)
                    .collect();
            }
            return self.start_if_called(now_ms);
        }
        self.advance(now_ms)
    }

    /// While Idle: starts a session if the last snapshot has a live mic and nothing blocks
    /// it. Runs on every snapshot and every tick, because a session the daemon ended (a
    /// capture failed) can leave the call's mic open in a graph that won't change again.
    fn start_if_called(&mut self, now_ms: u64) -> Vec<Action> {
        if self.paused || self.blocked(now_ms) {
            return Vec::new();
        }
        // Lowest serial first, since the graph sorts its streams but callers may not.
        let Some(first) = self
            .live_mics()
            .min_by_key(|s| s.serial)
            .map(|s| s.app.clone())
        else {
            return Vec::new();
        };
        self.phase = Phase::Tentative;
        self.trigger = Some(first.clone());
        self.started_ms = now_ms;
        let mut actions = vec![Action::Start {
            trigger: Some(first),
        }];
        actions.extend(self.advance(now_ms));
        actions
    }

    /// The monitor lost the PipeWire graph: no stream is known until the next snapshot.
    /// An automatic session ends as if the call had ended, but the time without a graph
    /// doesn't lift the restart block of a `stop` or `discard` (R26).
    pub fn graph_lost(&mut self, now_ms: u64) -> Vec<Action> {
        if self.paused {
            return Vec::new();
        }
        self.lost = true;
        self.quiet_since = None;
        self.last.clear();
        self.advance(now_ms)
    }

    /// Time passing without a new snapshot.
    pub fn tick(&mut self, now_ms: u64) -> Vec<Action> {
        if self.phase == Phase::Idle {
            return self.start_if_called(now_ms);
        }
        self.advance(now_ms)
    }

    /// `kute record`: a manual session, or turns a Tentative one manual.
    pub fn record(&mut self, now_ms: u64) -> Result<Vec<Action>, DetectError> {
        match self.phase {
            Phase::Recording | Phase::Grace => Err(DetectError::AlreadyRecording),
            Phase::Tentative => {
                self.phase = Phase::Recording;
                self.manual = true;
                self.dead_since = None;
                // The mic that started it counts toward following the call.
                self.mic_since = Some(self.started_ms);
                Ok(vec![Action::Commit])
            }
            Phase::Idle => {
                self.phase = Phase::Recording;
                self.manual = true;
                self.started_ms = now_ms;
                let mut actions = vec![Action::Start { trigger: None }, Action::Commit];
                // A call already in progress, as after a `stop`, may be followed.
                actions.extend(self.advance(now_ms));
                Ok(actions)
            }
        }
    }

    /// `kute stop`: ends any session at once, a Tentative one included.
    pub fn stop(&mut self) -> Result<Vec<Action>, DetectError> {
        self.end(Action::Finish).inspect(|_| self.block())
    }

    /// `kute discard`: ends the session and deletes it.
    pub fn discard(&mut self) -> Result<Vec<Action>, DetectError> {
        self.end(Action::Discard).inspect(|_| self.block())
    }

    /// The daemon ending the session on its own (a capture failed): like `stop`, but a
    /// call still in progress may start a new session right away.
    pub fn stop_unblocked(&mut self) -> Result<Vec<Action>, DetectError> {
        self.end(Action::Finish)
    }

    /// `discard` without blocking a new session, like `stop_unblocked`.
    pub fn discard_unblocked(&mut self) -> Result<Vec<Action>, DetectError> {
        self.end(Action::Discard)
    }

    /// Stop listening: ends an automatic session; a manual one continues.
    pub fn pause(&mut self) -> Vec<Action> {
        self.paused = true;
        self.last.clear();
        if self.manual {
            return Vec::new();
        }
        match self.phase {
            Phase::Idle => Vec::new(),
            Phase::Tentative => self.end(Action::Discard).unwrap_or_default(),
            Phase::Recording | Phase::Grace => self.end(Action::Finish).unwrap_or_default(),
        }
    }

    /// A call already in progress starts a new Tentative session on the next snapshot.
    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn manual(&self) -> bool {
        self.manual
    }

    /// Whether a manual session follows a call and will end with it.
    pub fn following(&self) -> bool {
        self.following
    }

    pub fn trigger(&self) -> Option<&App> {
        self.trigger.as_ref()
    }

    /// When the current session started; `None` when Idle.
    pub fn started_ms(&self) -> Option<u64> {
        (self.phase != Phase::Idle).then_some(self.started_ms)
    }

    fn end(&mut self, action: Action) -> Result<Vec<Action>, DetectError> {
        if self.phase == Phase::Idle {
            return Err(DetectError::NotRecording);
        }
        self.phase = Phase::Idle;
        self.manual = false;
        self.following = false;
        self.mic_since = None;
        self.trigger = None;
        self.bound.clear();
        self.started_ms = 0;
        self.dead_since = None;
        Ok(vec![action])
    }

    fn block(&mut self) {
        self.blocked = true;
        self.quiet_since = None;
    }

    /// Whether automatic Start is still blocked, lifting the block once snapshots have
    /// shown no mic for `MIC_GAP_MS`.
    fn blocked(&mut self, now_ms: u64) -> bool {
        if !self.blocked {
            return false;
        }
        if self.lost || self.live_mics().next().is_some() {
            self.quiet_since = None;
            return true;
        }
        let since = *self.quiet_since.get_or_insert(now_ms);
        if since.saturating_add(MIC_GAP_MS) <= now_ms {
            self.blocked = false;
            self.quiet_since = None;
        }
        self.blocked
    }

    fn ignored(&self, s: &Stream) -> bool {
        s.monitor
            || s.pid == Some(self.own_pid)
            || s.node_name
                .as_deref()
                .is_some_and(|n| n.starts_with("kute-"))
            || app_matches(&s.app, &self.ignore_apps)
            || self
                .allow_apps
                .as_deref()
                .is_some_and(|allow| !app_matches(&s.app, allow))
    }

    fn live_mics(&self) -> impl Iterator<Item = &Stream> {
        self.last
            .iter()
            .filter(|s| s.kind == StreamKind::Mic && !self.ignored(s))
    }

    fn outputs(&self) -> impl Iterator<Item = &Stream> {
        self.last.iter().filter(|s| s.kind == StreamKind::Output)
    }

    /// Whether the last snapshot keeps the session alive, binding the trigger app's output
    /// streams while a mic stream is there, except those already playing before the call.
    fn alive(&mut self) -> bool {
        let mic = self.live_mics().next().is_some();
        if mic && let Some(trigger) = &self.trigger {
            let outputs: Vec<u64> = self
                .outputs()
                .filter(|s| s.app == *trigger && !self.before_call.contains(&s.serial))
                .map(|s| s.serial)
                .collect();
            self.bound.extend(outputs);
        }
        mic || self
            .outputs()
            .any(|s| s.running && self.bound.contains(&s.serial))
    }

    /// Whether a manual session is to be moved along like an automatic one: once a mic has
    /// been live for `min_duration`, so a website's mic check doesn't end it, the session
    /// follows that call. While paused or without a graph, it carries on whatever happens.
    fn follow(&mut self, now_ms: u64) -> bool {
        if !self.end_manual {
            return false;
        }
        if self.paused || self.lost {
            self.dead_since = None;
            self.mic_since = None;
            return false;
        }
        if self.following {
            return true;
        }
        let Some(app) = self
            .live_mics()
            .min_by_key(|s| s.serial)
            .map(|s| s.app.clone())
        else {
            self.mic_since = None;
            return false;
        };
        let since = *self.mic_since.get_or_insert(now_ms);
        if now_ms.saturating_sub(since) < self.min_duration_ms {
            return false;
        }
        self.following = true;
        self.dead_since = None;
        self.trigger = Some(app);
        true
    }

    /// Moves a session along: commits, confirms gaps, runs out the grace. A manual one
    /// only once it follows a call.
    fn advance(&mut self, now_ms: u64) -> Vec<Action> {
        if self.phase == Phase::Idle || (self.manual && !self.follow(now_ms)) {
            return Vec::new();
        }
        let alive = self.alive();
        if alive {
            self.dead_since = None;
        } else if self.dead_since.is_none() {
            self.dead_since = Some(now_ms);
        }
        let dead = self
            .dead_since
            .is_some_and(|t| t.saturating_add(MIC_GAP_MS) <= now_ms);

        match self.phase {
            Phase::Tentative if dead => return self.end(Action::Discard).unwrap_or_default(),
            Phase::Tentative => {
                if alive && now_ms.saturating_sub(self.started_ms) >= self.min_duration_ms {
                    self.phase = Phase::Recording;
                    return vec![Action::Commit];
                }
            }
            Phase::Recording if dead => self.phase = Phase::Grace,
            Phase::Grace if alive => self.phase = Phase::Recording,
            _ => {}
        }
        if self.phase == Phase::Grace
            && let Some(since) = self.dead_since
            && now_ms.saturating_sub(since) >= self.stop_grace_ms
        {
            return self.end(Action::Finish).unwrap_or_default();
        }
        Vec::new()
    }
}

/// Feeds a trace's snapshots to a detector the way the daemon does: each snapshot at its
/// time, and a tick every 500 ms in between and up to `until_ms`. Returns each action with
/// the time it came.
pub fn replay(
    detector: &mut Detector,
    snapshots: &[(u64, crate::pw::Snapshot)],
    until_ms: u64,
) -> Vec<(u64, Action)> {
    const TICK_MS: u64 = 500;
    let mut actions = Vec::new();
    let mut next_tick = TICK_MS;
    let mut snaps = snapshots.iter().peekable();
    loop {
        let snap_at = snaps.peek().map(|(t, _)| *t);
        match snap_at {
            Some(t) if t <= next_tick && t <= until_ms => {
                let (_, snap) = snaps.next().expect("peeked");
                actions.extend(
                    detector
                        .streams(&snap.streams, t)
                        .into_iter()
                        .map(|a| (t, a)),
                );
            }
            _ if next_tick <= until_ms => {
                actions.extend(detector.tick(next_tick).into_iter().map(|a| (next_tick, a)));
                next_tick += TICK_MS;
            }
            _ => break,
        }
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DetectConfig {
        DetectConfig::default()
    }

    fn app(name: &str, binary: &str) -> App {
        App {
            name: Some(name.into()),
            binary: Some(binary.into()),
        }
    }

    fn jitsi() -> App {
        app("Jitsi Meet", "electron")
    }

    fn firefox() -> App {
        app("Firefox", "firefox")
    }

    fn stream(serial: u64, kind: StreamKind, app: App, running: bool) -> Stream {
        Stream {
            serial,
            kind,
            app,
            node_name: None,
            pid: Some(1000 + serial as u32),
            monitor: false,
            running,
        }
    }

    fn mic(serial: u64, app: App) -> Stream {
        stream(serial, StreamKind::Mic, app, true)
    }

    fn out(serial: u64, app: App, running: bool) -> Stream {
        stream(serial, StreamKind::Output, app, running)
    }

    fn meter(serial: u64) -> Stream {
        Stream {
            node_name: Some("PlasmaPA-VolumeMeter".into()),
            monitor: true,
            ..mic(
                serial,
                App {
                    name: None,
                    binary: Some("plasmashell".into()),
                },
            )
        }
    }

    fn kute(serial: u64) -> Stream {
        Stream {
            node_name: Some("kute-me".into()),
            ..mic(
                serial,
                App {
                    name: None,
                    binary: Some("pw-record".into()),
                },
            )
        }
    }

    /// A detector in Recording, started by Jitsi at 0 with mic 88 and output 114.
    fn recording() -> Detector {
        let mut d = Detector::new(&cfg(), 1);
        d.streams(&[mic(88, jitsi()), out(114, jitsi(), true)], 0);
        assert_eq!(d.tick(20_000), vec![Action::Commit]);
        d
    }

    #[test]
    fn website_mic_check_is_discarded() {
        let mut d = Detector::new(&cfg(), 1);
        assert!(matches!(
            d.streams(&[mic(10, firefox())], 0)[..],
            [Action::Start { .. }]
        ));
        assert_eq!(d.phase(), Phase::Tentative);
        assert!(d.streams(&[], 5_000).is_empty()); // gap not yet confirmed
        assert!(d.tick(6_999).is_empty());
        assert_eq!(d.tick(7_000), vec![Action::Discard]); // 5 s + 2 s
        assert_eq!(d.phase(), Phase::Idle);
        assert_eq!(d.trigger(), None);
    }

    #[test]
    fn start_names_the_lowest_serial_mic_app() {
        let mut d = Detector::new(&cfg(), 1);
        let actions = d.streams(&[mic(30, firefox()), mic(12, jitsi())], 0);
        assert_eq!(
            actions,
            vec![Action::Start {
                trigger: Some(jitsi())
            }]
        );
        assert_eq!(d.trigger(), Some(&jitsi()));
        assert!(!d.manual());
    }

    #[test]
    fn jitsi_lobby_to_call_is_one_session() {
        // Pre-join mic closes, the call's mic opens 0.5 s later.
        let mut d = Detector::new(&cfg(), 1);
        d.streams(&[mic(85, jitsi())], 0);
        d.streams(&[], 11_000);
        assert!(
            d.streams(&[mic(88, jitsi()), out(114, jitsi(), true)], 11_500)
                .is_empty()
        );
        assert!(d.tick(19_999).is_empty());
        assert_eq!(d.tick(20_000), vec![Action::Commit]);
        assert_eq!(d.phase(), Phase::Recording);
    }

    #[test]
    fn call_commits_then_finishes_after_grace() {
        let mut d = recording();
        // Leaving the room: the mic closes and the output idles.
        assert!(d.streams(&[out(114, jitsi(), false)], 100_000).is_empty());
        assert_eq!(d.phase(), Phase::Recording);
        assert!(d.tick(102_000).is_empty());
        assert_eq!(d.phase(), Phase::Grace);
        assert!(d.tick(114_999).is_empty());
        assert_eq!(d.tick(115_000), vec![Action::Finish]);
        assert_eq!(d.phase(), Phase::Idle);
        assert_eq!(d.trigger(), None);
    }

    #[test]
    fn grace_returns_to_recording_when_alive_again() {
        let mut d = recording();
        d.streams(&[], 100_000);
        d.tick(103_000);
        assert_eq!(d.phase(), Phase::Grace);
        assert!(d.streams(&[mic(90, jitsi())], 110_000).is_empty());
        assert_eq!(d.phase(), Phase::Recording);
        assert!(d.tick(200_000).is_empty());
    }

    #[test]
    fn bound_output_keeps_session_alive_only_while_running() {
        let mut d = recording();
        // The app closes its mic on mute; the output keeps running.
        assert!(d.streams(&[out(114, jitsi(), true)], 100_000).is_empty());
        for t in (101..=160).map(|s| s * 1_000) {
            assert!(d.tick(t).is_empty());
            assert_eq!(d.phase(), Phase::Recording);
        }
        assert!(d.streams(&[out(114, jitsi(), false)], 160_000).is_empty());
        assert!(d.tick(162_000).is_empty());
        assert_eq!(d.phase(), Phase::Grace);
        assert_eq!(d.tick(175_000), vec![Action::Finish]);
    }

    #[test]
    fn new_output_node_after_the_call_does_not_count() {
        let mut d = recording();
        d.streams(&[out(114, jitsi(), false)], 100_000);
        // Same app, new node, appearing after the mic closed.
        assert!(
            d.streams(
                &[out(114, jitsi(), false), out(200, jitsi(), true)],
                101_000
            )
            .is_empty()
        );
        assert!(d.tick(102_000).is_empty());
        assert_eq!(d.phase(), Phase::Grace);
        assert_eq!(d.tick(115_000), vec![Action::Finish]);
    }

    #[test]
    fn output_playing_before_the_call_is_not_bound() {
        // Another tab of the same browser plays sound before a Meet call starts in it.
        let mut d = Detector::new(&cfg(), 1);
        assert!(d.streams(&[out(50, firefox(), true)], 0).is_empty());
        d.streams(&[out(50, firefox(), true), mic(60, firefox())], 10_000);
        d.streams(
            &[
                out(50, firefox(), true),
                mic(60, firefox()),
                out(70, firefox(), true),
            ],
            15_000,
        );
        assert_eq!(d.tick(30_000), vec![Action::Commit]);
        // Leaving the call closes the mic and idles the call's output; the tab plays on.
        d.streams(
            &[out(50, firefox(), true), out(70, firefox(), false)],
            100_000,
        );
        assert!(d.tick(102_000).is_empty());
        assert_eq!(d.phase(), Phase::Grace);
        assert_eq!(d.tick(115_000), vec![Action::Finish]);
    }

    #[test]
    fn idle_output_from_before_the_call_can_still_be_bound() {
        // The app keeps an idle output node around and plays the call through it.
        let mut d = Detector::new(&cfg(), 1);
        d.streams(&[out(50, jitsi(), false)], 0);
        d.streams(&[out(50, jitsi(), true), mic(60, jitsi())], 10_000);
        assert_eq!(d.tick(30_000), vec![Action::Commit]);
        // The app closes its mic on mute; the call's output keeps running.
        d.streams(&[out(50, jitsi(), true)], 100_000);
        assert!(d.tick(110_000).is_empty());
        assert_eq!(d.phase(), Phase::Recording);
    }

    #[test]
    fn output_of_another_app_is_not_bound() {
        let mut d = Detector::new(&cfg(), 1);
        // Same binary, different name: not the trigger app.
        let other = app("Signal", "electron");
        d.streams(&[mic(88, jitsi()), out(120, other.clone(), true)], 0);
        d.tick(20_000);
        d.streams(&[out(120, other, true)], 100_000);
        assert!(d.tick(102_000).is_empty());
        assert_eq!(d.phase(), Phase::Grace);
    }

    #[test]
    fn peak_meters_never_start_a_session() {
        let mut d = Detector::new(&cfg(), 1);
        assert!(d.streams(&[meter(1), meter(2)], 0).is_empty());
        for t in (1..=60).map(|s| s * 1_000) {
            assert!(d.tick(t).is_empty());
        }
        assert_eq!(d.phase(), Phase::Idle);
    }

    #[test]
    fn own_capture_streams_are_ignored() {
        let own = Stream {
            pid: Some(4242),
            ..mic(7, firefox())
        };
        let mut d = Detector::new(&cfg(), 4242);
        assert!(d.streams(&[kute(5), own.clone()], 0).is_empty());
        assert!(d.tick(60_000).is_empty());
        assert_eq!(d.phase(), Phase::Idle);

        // During a session they don't keep it alive either.
        d.streams(&[mic(88, jitsi())], 100_000);
        assert_eq!(d.tick(120_000), vec![Action::Commit]);
        d.streams(&[kute(5), own, meter(3)], 130_000);
        d.tick(132_000);
        assert_eq!(d.phase(), Phase::Grace);
        assert_eq!(d.tick(145_000), vec![Action::Finish]);
    }

    #[test]
    fn ignore_and_allow_apps_match_name_or_binary() {
        let ignore = DetectConfig {
            ignore_apps: vec!["ELECTRON".into()],
            ..cfg()
        };
        let mut d = Detector::new(&ignore, 1);
        assert!(d.streams(&[mic(88, jitsi())], 0).is_empty());
        assert!(matches!(
            d.streams(&[mic(88, jitsi()), mic(90, firefox())], 1_000)[..],
            [Action::Start { .. }]
        ));
        assert_eq!(d.trigger(), Some(&firefox()));

        let allow = DetectConfig {
            allow_apps: Some(vec!["jitsi meet".into()]),
            ..cfg()
        };
        let mut d = Detector::new(&allow, 1);
        assert!(d.streams(&[mic(90, firefox())], 0).is_empty());
        assert_eq!(
            d.streams(&[mic(90, firefox()), mic(91, jitsi())], 1_000),
            vec![Action::Start {
                trigger: Some(jitsi())
            }]
        );
    }

    #[test]
    fn manual_record_ignores_liveness() {
        let mut d = Detector::new(&cfg(), 1);
        assert_eq!(
            d.record(0),
            Ok(vec![Action::Start { trigger: None }, Action::Commit])
        );
        assert_eq!(d.phase(), Phase::Recording);
        assert!(d.manual());
        assert_eq!(d.trigger(), None);
        for t in (1..=600).map(|s| s * 1_000) {
            assert!(d.streams(&[], t).is_empty());
            assert!(d.tick(t).is_empty());
        }
        assert_eq!(d.phase(), Phase::Recording);
        assert_eq!(d.stop(), Ok(vec![Action::Finish]));
        assert_eq!(d.phase(), Phase::Idle);
        assert!(!d.manual());
        assert_eq!(d.stop(), Err(DetectError::NotRecording));
        assert_eq!(d.discard(), Err(DetectError::NotRecording));
    }

    #[test]
    fn record_during_tentative_commits_as_manual() {
        let mut d = Detector::new(&cfg(), 1);
        d.streams(&[mic(10, firefox())], 0);
        assert_eq!(d.record(3_000), Ok(vec![Action::Commit]));
        assert_eq!(d.phase(), Phase::Recording);
        assert!(d.manual());
        // The mic closing no longer ends it.
        d.streams(&[], 4_000);
        assert!(d.tick(60_000).is_empty());
        assert_eq!(d.discard(), Ok(vec![Action::Discard]));
        assert_eq!(d.phase(), Phase::Idle);
    }

    #[test]
    fn record_while_recording_fails() {
        let mut d = recording();
        assert_eq!(d.record(30_000), Err(DetectError::AlreadyRecording));
        d.streams(&[], 100_000);
        d.tick(102_000);
        assert_eq!(d.phase(), Phase::Grace);
        assert_eq!(d.record(103_000), Err(DetectError::AlreadyRecording));

        let mut d = Detector::new(&cfg(), 1);
        d.record(0).unwrap();
        assert_eq!(d.record(1_000), Err(DetectError::AlreadyRecording));
    }

    #[test]
    fn stop_ends_automatic_sessions_at_once() {
        let mut d = Detector::new(&cfg(), 1);
        d.streams(&[mic(10, firefox())], 0);
        assert_eq!(d.stop(), Ok(vec![Action::Finish]));
        assert_eq!(d.phase(), Phase::Idle);

        let mut d = recording();
        assert_eq!(d.discard(), Ok(vec![Action::Discard]));
        assert_eq!(d.phase(), Phase::Idle);
    }

    #[test]
    fn pause_ends_automatic_sessions_and_ignores_streams() {
        let mut d = recording();
        assert_eq!(d.pause(), vec![Action::Finish]);
        assert!(d.paused());
        assert_eq!(d.phase(), Phase::Idle);
        assert!(d.streams(&[mic(88, jitsi())], 30_000).is_empty());
        assert!(d.tick(60_000).is_empty());
        assert_eq!(d.phase(), Phase::Idle);

        d.resume();
        assert!(!d.paused());
        assert!(matches!(
            d.streams(&[mic(88, jitsi())], 61_000)[..],
            [Action::Start { .. }]
        ));

        // Tentative is discarded.
        assert_eq!(d.pause(), vec![Action::Discard]);
    }

    #[test]
    fn manual_sessions_survive_pause_and_work_while_paused() {
        let mut d = Detector::new(&cfg(), 1);
        d.record(0).unwrap();
        assert!(d.pause().is_empty());
        assert_eq!(d.phase(), Phase::Recording);
        assert_eq!(d.stop(), Ok(vec![Action::Finish]));
        assert!(d.paused());

        assert_eq!(
            d.record(5_000),
            Ok(vec![Action::Start { trigger: None }, Action::Commit])
        );
        assert_eq!(d.discard(), Ok(vec![Action::Discard]));
    }

    #[test]
    fn error_messages() {
        assert_eq!(
            DetectError::AlreadyRecording.to_string(),
            "already recording"
        );
        assert_eq!(
            DetectError::NotRecording.to_string(),
            "nothing is being recorded"
        );
    }

    #[test]
    fn phase_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&Phase::Tentative).unwrap(),
            "\"tentative\""
        );
    }

    /// The mic stays open for a minute after the user's `stop`/`discard`, then closes at
    /// 100 s; a mic appearing after the confirmed gap starts a new session.
    fn assert_blocked_until_gap(d: &mut Detector) {
        for t in (1..=60).map(|s| 40_000 + s * 1_000) {
            assert!(d.streams(&[mic(88, jitsi())], t).is_empty());
            assert!(d.tick(t).is_empty());
        }
        assert_eq!(d.phase(), Phase::Idle);
        assert!(d.streams(&[], 100_000).is_empty());
        assert!(d.tick(101_999).is_empty());
        assert!(d.streams(&[mic(88, jitsi())], 101_999).is_empty());
        d.streams(&[], 102_000);
        assert!(d.tick(104_000).is_empty());
        assert!(matches!(
            d.streams(&[mic(95, jitsi())], 104_500)[..],
            [Action::Start { .. }]
        ));
    }

    #[test]
    fn stop_blocks_restart_until_the_mic_gap() {
        let mut d = recording();
        assert_eq!(d.stop(), Ok(vec![Action::Finish]));
        // The daemon stopping its capture streams produces a snapshot with the call's mic.
        assert!(
            d.streams(&[mic(88, jitsi()), out(114, jitsi(), true)], 30_000)
                .is_empty()
        );
        assert_blocked_until_gap(&mut d);
    }

    #[test]
    fn discard_from_tentative_blocks_restart_until_the_mic_gap() {
        let mut d = Detector::new(&cfg(), 1);
        d.streams(&[mic(88, jitsi())], 0);
        assert_eq!(d.discard(), Ok(vec![Action::Discard]));
        assert!(d.streams(&[mic(88, jitsi())], 30_000).is_empty());
        assert_blocked_until_gap(&mut d);
    }

    #[test]
    fn manual_stop_during_a_call_blocks_restart() {
        let mut d = Detector::new(&cfg(), 1);
        d.record(0).unwrap();
        d.streams(&[mic(88, jitsi())], 10_000);
        assert_eq!(d.stop(), Ok(vec![Action::Finish]));
        assert!(d.streams(&[mic(88, jitsi())], 30_000).is_empty());
        assert_blocked_until_gap(&mut d);
    }

    #[test]
    fn record_still_works_while_restart_is_blocked() {
        let mut d = recording();
        d.stop().unwrap();
        assert_eq!(
            d.record(30_000),
            Ok(vec![Action::Start { trigger: None }, Action::Commit])
        );
    }

    #[test]
    fn a_session_the_daemon_ends_does_not_block_restart() {
        let mut d = recording();
        assert_eq!(d.stop_unblocked(), Ok(vec![Action::Finish]));
        assert_eq!(d.phase(), Phase::Idle);
        // The call's mic is still there: it starts a new session at once.
        assert!(matches!(
            d.streams(&[mic(88, jitsi()), out(114, jitsi(), true)], 30_000)[..],
            [Action::Start { .. }]
        ));
        assert_eq!(d.discard_unblocked(), Ok(vec![Action::Discard]));
        assert!(matches!(
            d.streams(&[mic(88, jitsi())], 31_000)[..],
            [Action::Start { .. }]
        ));
        assert_eq!(d.started_ms(), Some(31_000));
        d.stop_unblocked().unwrap();
        assert_eq!(d.started_ms(), None);
        assert_eq!(d.stop_unblocked(), Err(DetectError::NotRecording));
        assert_eq!(d.discard_unblocked(), Err(DetectError::NotRecording));
    }

    #[test]
    fn pause_in_grace_finishes() {
        let mut d = recording();
        d.streams(&[], 100_000);
        d.tick(102_000);
        assert_eq!(d.phase(), Phase::Grace);
        assert_eq!(d.pause(), vec![Action::Finish]);
        assert_eq!(d.phase(), Phase::Idle);
    }

    #[test]
    fn a_lost_graph_does_not_lift_the_restart_block() {
        let mut d = recording();
        d.stop().unwrap();
        assert!(
            d.streams(&[mic(88, jitsi()), out(114, jitsi(), true)], 30_000)
                .is_empty()
        );
        assert!(d.graph_lost(31_000).is_empty());
        for t in (32..=91).map(|s| s * 1_000) {
            assert!(d.tick(t).is_empty());
        }
        // The monitor reconnects and the call is still on.
        assert!(
            d.streams(&[mic(88, jitsi()), out(114, jitsi(), true)], 92_000)
                .is_empty()
        );
        assert!(d.tick(93_000).is_empty());
        assert_eq!(d.phase(), Phase::Idle);
    }

    #[test]
    fn the_restart_block_counts_quiet_time_from_the_first_snapshot_after_reconnect() {
        let mut d = recording();
        d.stop().unwrap();
        d.graph_lost(30_000);
        d.tick(40_000);
        // The call ended while the graph was lost.
        assert!(d.streams(&[], 50_000).is_empty());
        assert!(d.tick(51_999).is_empty());
        assert!(d.streams(&[mic(88, jitsi())], 51_999).is_empty());
        d.streams(&[], 52_000);
        assert!(d.tick(54_000).is_empty());
        assert!(matches!(
            d.streams(&[mic(95, jitsi())], 54_500)[..],
            [Action::Start { .. }]
        ));
    }

    #[test]
    fn a_stop_while_the_graph_is_lost_blocks_until_a_snapshot() {
        let mut d = recording();
        assert!(d.graph_lost(30_000).is_empty());
        assert_eq!(d.stop(), Ok(vec![Action::Finish]));
        for t in (31..=90).map(|s| s * 1_000) {
            assert!(d.tick(t).is_empty());
        }
        assert!(d.streams(&[mic(88, jitsi())], 91_000).is_empty());
        assert!(d.tick(92_000).is_empty());
        assert_eq!(d.phase(), Phase::Idle);
        d.streams(&[], 93_000);
        assert!(d.tick(95_000).is_empty());
        assert!(matches!(
            d.streams(&[mic(88, jitsi())], 95_500)[..],
            [Action::Start { .. }]
        ));
    }

    #[test]
    fn a_lost_graph_ends_automatic_sessions_as_if_the_call_ended() {
        let mut d = recording();
        assert!(d.graph_lost(100_000).is_empty());
        assert!(d.tick(102_000).is_empty());
        assert_eq!(d.phase(), Phase::Grace);
        assert!(d.tick(114_999).is_empty());
        assert_eq!(d.tick(115_000), vec![Action::Finish]);

        // A Tentative one is discarded once the gap is confirmed.
        assert!(matches!(
            d.streams(&[mic(88, jitsi())], 120_000)[..],
            [Action::Start { .. }]
        ));
        assert!(d.graph_lost(121_000).is_empty());
        assert_eq!(d.tick(123_000), vec![Action::Discard]);

        // A manual one carries on.
        d.record(130_000).unwrap();
        assert!(d.graph_lost(131_000).is_empty());
        assert!(d.tick(200_000).is_empty());
        assert_eq!(d.phase(), Phase::Recording);
    }

    #[test]
    fn a_manual_session_follows_a_call_and_ends_with_it() {
        // The user stops the automatic session by mistake, then records the rest by hand.
        let mut d = recording();
        d.stop().unwrap();
        let call = [mic(88, jitsi()), out(114, jitsi(), true)];
        assert!(d.streams(&call, 30_000).is_empty());
        d.record(30_000).unwrap();
        assert_eq!(d.trigger(), None);
        assert!(d.tick(49_999).is_empty());
        assert!(!d.following());
        assert!(d.tick(50_000).is_empty()); // min_duration with the mic open
        assert!(d.following());
        assert!(d.manual());
        assert_eq!(d.trigger(), Some(&jitsi()));
        // The other side keeps talking a little after the mic closes.
        assert!(d.streams(&[out(114, jitsi(), true)], 100_000).is_empty());
        assert!(d.tick(200_000).is_empty());
        assert_eq!(d.phase(), Phase::Recording);
        assert!(d.streams(&[], 300_000).is_empty());
        assert!(d.tick(302_000).is_empty());
        assert_eq!(d.phase(), Phase::Grace);
        assert!(d.tick(314_999).is_empty());
        assert_eq!(d.tick(315_000), vec![Action::Finish]);
        assert_eq!(d.phase(), Phase::Idle);
        assert!(!d.manual());
        assert!(!d.following());
    }

    #[test]
    fn a_call_joining_a_manual_session_is_followed_and_may_come_back_in_grace() {
        let mut d = Detector::new(&cfg(), 1);
        d.record(0).unwrap();
        assert!(d.streams(&[mic(88, jitsi())], 60_000).is_empty());
        assert!(d.tick(80_000).is_empty());
        assert!(d.following());
        d.streams(&[], 100_000);
        d.tick(102_000);
        assert_eq!(d.phase(), Phase::Grace);
        // A reconnect.
        assert!(d.streams(&[mic(90, jitsi())], 110_000).is_empty());
        assert_eq!(d.phase(), Phase::Recording);
        assert!(d.tick(200_000).is_empty());
        assert_eq!(d.phase(), Phase::Recording);
    }

    #[test]
    fn a_mic_check_does_not_end_a_manual_session() {
        let mut d = Detector::new(&cfg(), 1);
        d.record(0).unwrap();
        assert!(d.streams(&[mic(10, firefox())], 10_000).is_empty());
        assert!(d.tick(29_999).is_empty());
        assert!(d.streams(&[], 29_999).is_empty());
        for t in (30..=600).map(|s| s * 1_000) {
            assert!(d.tick(t).is_empty());
        }
        assert!(!d.following());
        assert_eq!(d.phase(), Phase::Recording);
        // Short opens don't add up to min_duration.
        for t in (0..10).map(|i| 700_000 + i * 20_000) {
            d.streams(&[mic(10, firefox())], t);
            d.streams(&[], t + 15_000);
        }
        assert!(d.tick(900_000).is_empty());
        assert!(!d.following());
    }

    #[test]
    fn a_tentative_session_turned_manual_follows_its_call() {
        let mut d = Detector::new(&cfg(), 1);
        d.streams(&[mic(88, jitsi())], 0);
        assert_eq!(d.record(3_000), Ok(vec![Action::Commit]));
        assert!(d.tick(19_999).is_empty());
        assert!(!d.following());
        assert!(d.tick(20_000).is_empty()); // counted from when the mic opened
        assert!(d.following());
        d.streams(&[], 60_000);
        assert_eq!(d.tick(75_000), vec![Action::Finish]);
    }

    #[test]
    fn end_manual_off_keeps_manual_sessions_until_stop() {
        let config = DetectConfig {
            end_manual: false,
            ..cfg()
        };
        let mut d = Detector::new(&config, 1);
        d.record(0).unwrap();
        d.streams(&[mic(88, jitsi())], 1_000);
        assert!(d.tick(60_000).is_empty());
        assert!(!d.following());
        d.streams(&[], 70_000);
        assert!(d.tick(600_000).is_empty());
        assert_eq!(d.phase(), Phase::Recording);
    }

    #[test]
    fn a_followed_manual_session_survives_pause_and_a_lost_graph() {
        let mut d = Detector::new(&cfg(), 1);
        d.record(0).unwrap();
        d.streams(&[mic(88, jitsi())], 1_000);
        d.tick(21_000);
        assert!(d.following());

        assert!(d.pause().is_empty());
        for t in (22..=200).map(|s| s * 1_000) {
            assert!(d.tick(t).is_empty());
        }
        assert_eq!(d.phase(), Phase::Recording);
        d.resume();
        assert!(d.streams(&[mic(88, jitsi())], 201_000).is_empty());

        assert!(d.graph_lost(210_000).is_empty());
        for t in (211..=400).map(|s| s * 1_000) {
            assert!(d.tick(t).is_empty());
        }
        assert_eq!(d.phase(), Phase::Recording);
        // Back without the call: it ends like an automatic one.
        assert!(d.streams(&[], 401_000).is_empty());
        assert!(d.tick(403_000).is_empty());
        assert_eq!(d.phase(), Phase::Grace);
        assert_eq!(d.tick(416_000), vec![Action::Finish]);
    }
}
