//! Audio capture for one track: raw 16 kHz mono S16LE PCM from `pw-record`, behind a
//! trait so the recorder can be tested with pipes.

use crate::store::Track;
use anyhow::{Context, Result};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

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
            .args(pw_record_args(track, target, pw_record_has_raw()))
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

/// Whether this `pw-record` knows `--raw`. Older ones, such as PipeWire 1.0 on Ubuntu 24.04
/// and Linux Mint 22, reject it and exit at once; they write raw PCM to stdout anyway.
fn pw_record_has_raw() -> bool {
    static HAS_RAW: OnceLock<bool> = OnceLock::new();
    *HAS_RAW.get_or_init(|| {
        Command::new("pw-record")
            .arg("--help")
            .stdin(Stdio::null())
            .output()
            .map(|out| help_has_raw(&out.stdout) || help_has_raw(&out.stderr))
            .unwrap_or(true)
    })
}

fn help_has_raw(help: &[u8]) -> bool {
    String::from_utf8_lossy(help).contains("--raw")
}

/// `raw`: pass `--raw`, which a newer pw-record needs, or it puts an AU header in front of
/// the PCM on stdout. An older one has no such option and writes raw PCM there by default.
pub fn pw_record_args(track: Track, target: Option<&str>, raw: bool) -> Vec<String> {
    let props = match track {
        Track::Me => r#"{ node.name = "kute-me" media.name = "kute (me)" }"#,
        Track::Others => {
            r#"{ node.name = "kute-others" media.name = "kute (others)" stream.capture.sink = true }"#
        }
    };
    let mut args: Vec<String> = [
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
    if raw {
        args.insert(0, "--raw".to_string());
    }
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
            let mut v: Vec<String> = vec!["--raw".to_string()];
            v.extend(base.iter().map(|s| s.to_string()));
            v.push(props.to_string());
            if let Some(t) = target {
                v.extend(["--target".to_string(), t.to_string()]);
            }
            v.push("-".to_string());
            v
        };
        assert_eq!(pw_record_args(Track::Me, None, true), expect(me, None));
        assert_eq!(
            pw_record_args(Track::Me, Some("alsa_input.usb-mic"), true),
            expect(me, Some("alsa_input.usb-mic"))
        );
        assert_eq!(
            pw_record_args(Track::Others, None, true),
            expect(others, None)
        );
    }

    #[test]
    fn pw_record_args_without_raw_for_an_older_pw_record() {
        let args = pw_record_args(Track::Me, None, false);
        assert!(!args.contains(&"--raw".to_string()));
        assert_eq!(args.first().map(String::as_str), Some("--rate"));
        assert_eq!(args.last().map(String::as_str), Some("-"));
    }

    #[test]
    fn help_has_raw_matches_the_option() {
        assert!(help_has_raw(
            b"      --raw                           RAW mode\n"
        ));
        assert!(!help_has_raw(b"  -R, --remote   Remote daemon name\n"));
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
