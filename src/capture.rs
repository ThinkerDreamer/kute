//! Audio capture for one track: raw 16 kHz mono S16LE PCM from `pw-record`, behind a
//! trait so the recorder can be tested with pipes.

use crate::store::Track;
use anyhow::{Context, Result};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};

/// A running capture: PCM comes out of `reader` until the capture is stopped.
pub struct Capture {
    pub reader: Box<dyn Read + Send>,
    stop: StopHandle,
}

impl Capture {
    pub fn new(reader: Box<dyn Read + Send>, stop: impl FnOnce() + Send + 'static) -> Capture {
        Capture {
            reader,
            stop: StopHandle(Arc::new(StopOnce(Mutex::new(Some(Box::new(stop)))))),
        }
    }

    /// A handle that stops this capture, for whoever keeps it after `reader` moves to the
    /// thread that reads it.
    pub fn stop_handle(&self) -> StopHandle {
        self.stop.clone()
    }

    pub fn stop(self) {
        self.stop.stop();
    }
}

/// Stops a capture. Only the first call from any handle does anything; the reader then
/// sees EOF. Dropping the last handle (and the `Capture`) without a call stops it too, so
/// a forgotten `pw-record` doesn't keep running.
#[derive(Clone)]
pub struct StopHandle(Arc<StopOnce>);

impl StopHandle {
    pub fn stop(&self) {
        let stop = self
            .0
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(stop) = stop {
            stop();
        }
    }
}

type StopFn = Box<dyn FnOnce() + Send>;

struct StopOnce(Mutex<Option<StopFn>>);

impl Drop for StopOnce {
    fn drop(&mut self) {
        let stop = self
            .0
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(stop) = stop {
            stop();
        }
    }
}

pub trait CaptureSource: Send {
    /// `target`: a node.name to record from (Me with capture.mic); None = the default.
    fn open(&self, track: Track, target: Option<&str>) -> Result<Capture>;
}

pub struct PwRecord;

impl CaptureSource for PwRecord {
    fn open(&self, track: Track, target: Option<&str>) -> Result<Capture> {
        let mut child = Command::new("pw-record")
            .args(pw_record_args(track, target))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .context("starting pw-record")?;
        let stdout = child.stdout.take().expect("stdout is piped");
        // pw-record writes to a pipe, so there is nothing to finalize: kill it outright.
        Ok(Capture::new(Box::new(stdout), move || {
            let _ = child.kill();
            let _ = child.wait();
        }))
    }
}

/// `--raw`: without it pw-record puts an AU header in front of the PCM on stdout.
pub fn pw_record_args(track: Track, target: Option<&str>) -> Vec<String> {
    let props = match track {
        Track::Me => r#"{ node.name = "kute-me" media.name = "kute (me)" }"#,
        Track::Others => {
            r#"{ node.name = "kute-others" media.name = "kute (others)" stream.capture.sink = true }"#
        }
    };
    let mut args: Vec<String> = [
        "--raw",
        "--rate",
        "16000",
        "--channels",
        "1",
        "--format",
        "s16",
        "-P",
        props,
    ]
    .map(String::from)
    .into();
    if let Some(target) = target {
        args.extend(["--target".to_string(), target.to_string()]);
    }
    args.push("-".to_string());
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pw_record_args_for_both_tracks() {
        let base = [
            "--raw",
            "--rate",
            "16000",
            "--channels",
            "1",
            "--format",
            "s16",
            "-P",
        ];
        let me = r#"{ node.name = "kute-me" media.name = "kute (me)" }"#;
        let others = r#"{ node.name = "kute-others" media.name = "kute (others)" stream.capture.sink = true }"#;
        let expect = |props: &str, target: Option<&str>| {
            let mut v: Vec<String> = base.iter().map(|s| s.to_string()).collect();
            v.push(props.to_string());
            if let Some(t) = target {
                v.extend(["--target".to_string(), t.to_string()]);
            }
            v.push("-".to_string());
            v
        };
        assert_eq!(pw_record_args(Track::Me, None), expect(me, None));
        assert_eq!(
            pw_record_args(Track::Me, Some("alsa_input.usb-mic")),
            expect(me, Some("alsa_input.usb-mic"))
        );
        assert_eq!(pw_record_args(Track::Others, None), expect(others, None));
    }

    #[test]
    fn stop_runs_once_from_any_handle() {
        let count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let c = count.clone();
        let capture = Capture::new(Box::new(std::io::empty()), move || {
            c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        let handle = capture.stop_handle();
        handle.stop();
        capture.stop();
        handle.stop();
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_the_last_handle_stops() {
        let count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let c = count.clone();
        let capture = Capture::new(Box::new(std::io::empty()), move || {
            c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        let handle = capture.stop_handle();
        drop(capture);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
        drop(handle);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);

        let c = count.clone();
        let capture = Capture::new(Box::new(std::io::empty()), move || {
            c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        capture.stop();
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
