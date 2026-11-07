//! Runs `pw-dump --monitor` and turns its output into snapshots.

use super::{Graph, Snapshot};
use anyhow::{Context, Result};
use serde_json::Value;
use std::io::{BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum MonitorEvent {
    Snapshot(Snapshot),
    /// `pw-dump` exited or could not start; the monitor retries on its own.
    Disconnected(String),
}

/// Feeds `pw-dump`'s concatenated JSON arrays into a fresh graph, and calls `on_change`
/// after every array that changed the snapshot. The first array always counts as a change.
pub fn read_updates(r: impl Read, mut on_change: impl FnMut(Snapshot)) -> Result<()> {
    let mut graph = Graph::default();
    let mut last = None;
    read_arrays(r, |update| {
        graph.apply(&update);
        let snap = graph.snapshot();
        if last.as_ref() != Some(&snap) {
            last = Some(snap.clone());
            on_change(snap);
        }
    })
}

/// Calls `each` with every JSON array `pw-dump` prints, as it arrives.
pub fn read_arrays(r: impl Read, mut each: impl FnMut(Vec<Value>)) -> Result<()> {
    // `from_reader` reads a byte at a time, so give it a buffer.
    let r = BufReader::new(r);
    for update in serde_json::Deserializer::from_reader(r).into_iter::<Vec<Value>>() {
        each(update.context("parsing pw-dump output")?);
    }
    Ok(())
}

/// Starts `pw-dump --monitor` with its output piped.
pub fn spawn_pw_dump() -> std::io::Result<Child> {
    Command::new("pw-dump")
        .args(["--monitor", "--no-colors"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
}

/// Watches the graph on its own thread until `tx`'s receiver is gone, restarting
/// `pw-dump` with a backoff whenever it exits.
pub fn spawn(tx: Sender<MonitorEvent>) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut attempt = 0;
        loop {
            let started = Instant::now();
            let Some(reason) = run_once(&tx) else {
                return;
            };
            if tx.send(MonitorEvent::Disconnected(reason)).is_err() {
                return;
            }
            let delay;
            (delay, attempt) = backoff(attempt, started.elapsed());
            thread::sleep(delay);
        }
    })
}

/// One run of `pw-dump`. Returns why it ended, or `None` if the receiver is gone.
fn run_once(tx: &Sender<MonitorEvent>) -> Option<String> {
    let mut child = match spawn_pw_dump() {
        Ok(child) => child,
        Err(e) => return Some(format!("starting pw-dump: {e}")),
    };
    let stdout = child.stdout.take().expect("stdout is piped");
    let mut gone = false;
    let read = read_updates(stdout, |snap| {
        // Killing pw-dump closes the pipe, which ends `read_updates`.
        if !gone && tx.send(MonitorEvent::Snapshot(snap)).is_err() {
            gone = true;
            let _ = child.kill();
        }
    });
    let _ = child.kill();
    let status = child.wait();
    if gone {
        return None;
    }
    Some(match (read, status) {
        (Err(e), _) => format!("{e:#}"),
        (Ok(()), Ok(status)) => format!("pw-dump exited ({status})"),
        (Ok(()), Err(e)) => format!("waiting for pw-dump: {e}"),
    })
}

const BACKOFF_SECS: [u64; 5] = [1, 2, 5, 10, 30];

/// The wait before restart number `attempt` (0-based), and the next attempt number. A
/// run that lasted over a minute was a working connection, so the schedule starts over.
fn backoff(attempt: usize, ran_for: Duration) -> (Duration, usize) {
    let attempt = if ran_for > Duration::from_secs(60) {
        0
    } else {
        attempt
    };
    let secs = BACKOFF_SECS[attempt.min(BACKOFF_SECS.len() - 1)];
    (Duration::from_secs(secs), attempt + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pw::StreamKind;

    #[test]
    fn read_updates_reports_only_changes() {
        let input = r#"[
            {"id": 30, "type": "PipeWire:Interface:Node", "info": {"state": "running",
              "props": {"object.serial": 300, "media.class": "Audio/Source", "node.name": "mic"}}},
            {"id": 40, "type": "PipeWire:Interface:Node", "info": {"state": "running",
              "props": {"object.serial": 400, "media.class": "Stream/Input/Audio"}}},
            {"id": 70, "type": "PipeWire:Interface:Link", "info": {"output-node-id": 30, "input-node-id": 40}}
        ]
        [
            {"id": 80, "type": "PipeWire:Interface:Device", "info": {"props": {"device.name": "x"}}}
        ]
        [
            {"id": 40, "type": null, "info": null}
        ]"#;
        let mut snaps = Vec::new();
        read_updates(input.as_bytes(), |s| snaps.push(s)).unwrap();
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0].streams[0].kind, StreamKind::Mic);
        assert!(snaps[1].streams.is_empty());
    }

    #[test]
    fn read_updates_fails_on_garbage() {
        assert!(read_updates("[] nope".as_bytes(), |_| {}).is_err());
    }

    #[test]
    fn backoff_steps_up_then_resets_after_a_long_run() {
        let short = Duration::from_secs(3);
        let mut attempt = 0;
        let mut delays = Vec::new();
        for _ in 0..7 {
            let delay;
            (delay, attempt) = backoff(attempt, short);
            delays.push(delay.as_secs());
        }
        assert_eq!(delays, vec![1, 2, 5, 10, 30, 30, 30]);

        let (delay, attempt) = backoff(attempt, Duration::from_secs(61));
        assert_eq!((delay.as_secs(), attempt), (1, 1));
        let (delay, _) = backoff(attempt, Duration::from_secs(60));
        assert_eq!(delay.as_secs(), 2);
    }
}
