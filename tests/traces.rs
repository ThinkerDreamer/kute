//! Detector tests on PipeWire traces recorded with `kute debug trace` (spec section 10).
//! `synthetic-*` traces are written by hand in the same format; real ones are scrubbed
//! recordings.

use kute::config::DetectConfig;
use kute::detect::{Action, Detector, replay};
use kute::pw::trace;
use std::path::Path;

/// The actions without their times, and each `Start`'s app name.
fn run(name: &str, until_ms: u64) -> Vec<(u64, String)> {
    run_with(name, until_ms, &DetectConfig::default())
}

fn run_with(name: &str, until_ms: u64, config: &DetectConfig) -> Vec<(u64, String)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/traces")
        .join(name);
    let entries = trace::read_file(&path).unwrap();
    let snapshots = trace::snapshots(&entries);
    let mut detector = Detector::new(config, 1);
    replay(&mut detector, &snapshots, until_ms)
        .into_iter()
        .map(|(t, a)| {
            let label = match a {
                Action::Start { trigger } => {
                    format!("start {}", trigger.and_then(|t| t.name).unwrap_or_default())
                }
                Action::Commit => "commit".into(),
                Action::Finish => "finish".into(),
                Action::Discard => "discard".into(),
            };
            (t, label)
        })
        .collect()
}

#[test]
fn a_jitsi_call_is_one_session_from_pre_join_to_leaving() {
    let actions = run("synthetic-jitsi-call.jsonl", 120_000);
    let labels: Vec<_> = actions.iter().map(|(_, l)| l.as_str()).collect();
    // The 1 s gap on joining and the short join sound don't split it; the peak meter
    // before the call doesn't start anything.
    assert_eq!(labels, ["start Jitsi Meet", "commit", "finish"]);
    assert_eq!(actions[0].0, 2_000);
    // Default min_duration 20 s, stop_grace 15 s after the mic closed at 60 s.
    assert!((22_000..=22_500).contains(&actions[1].0), "{actions:?}");
    assert!((75_000..=75_500).contains(&actions[2].0), "{actions:?}");
}

#[test]
fn a_mic_check_never_becomes_a_recording() {
    let labels: Vec<_> = run("synthetic-mic-check.jsonl", 60_000)
        .into_iter()
        .map(|(_, l)| l)
        .collect();
    assert_eq!(labels, ["start Firefox", "discard"]);
}

#[test]
fn a_call_ends_while_obs_keeps_recording_the_mic() {
    // OBS has the mic open from 0 to 300 s; the Jitsi call runs from 5 s to 65 s.
    let actions = run("synthetic-obs-holds-mic.jsonl", 360_000);
    let labels: Vec<_> = actions.iter().map(|(_, l)| l.as_str()).collect();
    assert_eq!(labels, ["start Jitsi Meet", "commit", "finish"]);
    assert_eq!(actions[0].0, 5_000);
    assert!((80_000..=80_500).contains(&actions[2].0), "{actions:?}");
}

#[test]
fn with_recorders_counted_obs_holds_the_session_until_it_closes() {
    let config = DetectConfig {
        ignore_recorders: false,
        ..DetectConfig::default()
    };
    let actions = run_with("synthetic-obs-holds-mic.jsonl", 360_000, &config);
    let labels: Vec<_> = actions.iter().map(|(_, l)| l.as_str()).collect();
    assert_eq!(labels, ["start OBS", "commit", "finish"]);
    assert!((315_000..=315_500).contains(&actions[2].0), "{actions:?}");
}
