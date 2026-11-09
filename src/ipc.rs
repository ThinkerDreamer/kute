//! The control socket: how CLI commands talk to the daemon. One JSON object per line each
//! way over a Unix socket; the daemon thread answers through a per-request reply channel.

use crate::detect::Phase;
use crate::jobs::Job;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// How long a connection handler waits for the daemon thread to answer.
const REPLY_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the server waits for a client to send its request line.
const REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the client waits for the daemon's answer; longer than `REPLY_TIMEOUT` so the
/// daemon's own "did not answer" error arrives first.
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Upper bound on one request or response line.
const MAX_LINE: u64 = 1 << 20;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    Status,
    Record {
        title: Option<String>,
    },
    Stop,
    Discard,
    Pause,
    Resume,
    Job {
        job: Job,
    },
    /// Which config and archive the daemon runs with, so a command can tell whether its
    /// jobs mean the same to the daemon.
    Identity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Status {
    pub phase: Phase,
    pub paused: bool,
    pub session: Option<String>,
    pub app: Option<String>,
    pub queued: usize,
    /// Sessions waiting out the backoff before their transcription is tried again.
    #[serde(default)]
    pub retrying: usize,
    pub running_job: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum Response {
    Ok {
        message: String,
    },
    Status {
        status: Status,
    },
    Error {
        message: String,
    },
    Identity {
        /// The daemon's config file (canonical), or `None` when it runs on the defaults.
        config: Option<PathBuf>,
        /// The daemon's archive (storage root), absolute.
        archive: PathBuf,
    },
}

/// A request received on the socket, with the channel the daemon answers on.
pub struct Incoming {
    pub request: Request,
    pub reply: Sender<Response>,
}

/// How the daemon's reply to a request it can't parse begins.
pub const BAD_REQUEST: &str = "bad request:";

pub fn socket_path() -> Result<PathBuf> {
    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .filter(|d| !d.is_empty())
        .context("XDG_RUNTIME_DIR is not set; is this a desktop session?")?;
    Ok(PathBuf::from(dir).join("kute.sock"))
}

/// Binds the socket and accepts connections on a background thread. A leftover socket file
/// with nobody listening is replaced; a live one means another daemon is running.
pub fn serve(path: &Path, tx: Sender<Incoming>) -> Result<JoinHandle<()>> {
    serve_with_timeout(path, tx, REPLY_TIMEOUT)
}

fn serve_with_timeout(
    path: &Path,
    tx: Sender<Incoming>,
    reply_timeout: Duration,
) -> Result<JoinHandle<()>> {
    if path.exists() {
        if UnixStream::connect(path).is_ok() {
            bail!("kute daemon is already running");
        }
        std::fs::remove_file(path)
            .with_context(|| format!("removing stale socket {}", path.display()))?;
    }
    let listener =
        UnixListener::bind(path).with_context(|| format!("binding {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("setting permissions on {}", path.display()))?;
    Ok(thread::spawn(move || {
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    let tx = tx.clone();
                    thread::spawn(move || {
                        if let Err(e) = handle(stream, &tx, reply_timeout) {
                            eprintln!("kute: control socket: {e:#}");
                        }
                    });
                }
                Err(e) => {
                    eprintln!("kute: control socket accept failed: {e}");
                    // Don't spin if accept keeps failing (e.g. out of file descriptors).
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }))
}

fn handle(mut stream: UnixStream, tx: &Sender<Incoming>, reply_timeout: Duration) -> Result<()> {
    stream.set_read_timeout(Some(REQUEST_READ_TIMEOUT))?;
    let mut line = String::new();
    BufReader::new((&stream).take(MAX_LINE)).read_line(&mut line)?;
    if line.trim().is_empty() {
        // A client that connected and left, such as another `serve` probing for a daemon.
        return Ok(());
    }
    let response = match serde_json::from_str::<Request>(line.trim()) {
        Err(e) => Response::Error {
            message: format!("{BAD_REQUEST} {e}"),
        },
        Ok(request) => {
            let (reply, answer) = channel();
            if tx.send(Incoming { request, reply }).is_err() {
                Response::Error {
                    message: "the daemon is shutting down".into(),
                }
            } else {
                answer
                    .recv_timeout(reply_timeout)
                    .unwrap_or_else(|_| Response::Error {
                        message: "the daemon did not answer".into(),
                    })
            }
        }
    };
    let mut out = serde_json::to_string(&response)?;
    out.push('\n');
    stream.write_all(out.as_bytes())?;
    Ok(())
}

/// Sends one request to the daemon. `Ok(None)` means there is no daemon (no socket, or
/// nobody listening); a daemon that is there but silent is an error.
pub fn request(path: &Path, req: &Request) -> Result<Option<Response>> {
    let mut stream = match UnixStream::connect(path) {
        Ok(s) => s,
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
            ) =>
        {
            return Ok(None);
        }
        Err(e) => return Err(e).with_context(|| format!("connecting to {}", path.display())),
    };
    stream.set_read_timeout(Some(CLIENT_READ_TIMEOUT))?;
    let mut out = serde_json::to_string(req)?;
    out.push('\n');
    stream.write_all(out.as_bytes())?;
    let mut line = String::new();
    BufReader::new((&stream).take(MAX_LINE))
        .read_line(&mut line)
        .context("reading the daemon's reply")?;
    if line.trim().is_empty() {
        bail!("the daemon closed the connection without replying");
    }
    let response = serde_json::from_str(line.trim()).context("parsing the daemon's reply")?;
    Ok(Some(response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::Receiver;

    /// Unix socket paths top out around 108 bytes, so keep these under /tmp.
    fn tmp() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("kute-ipc")
            .tempdir_in("/tmp")
            .unwrap()
    }

    fn status() -> Status {
        Status {
            phase: Phase::Recording,
            paused: false,
            session: Some("2026-10-19-1000-standup".into()),
            app: Some("zoom".into()),
            queued: 2,
            retrying: 1,
            running_job: Some("process".into()),
        }
    }

    /// Answers every incoming request with `answer(request)`.
    fn answer_with(rx: Receiver<Incoming>, answer: fn(&Request) -> Response) {
        thread::spawn(move || {
            for incoming in rx {
                let _ = incoming.reply.send(answer(&incoming.request));
            }
        });
    }

    #[test]
    fn request_round_trip() {
        let dir = tmp();
        let path = dir.path().join("k.sock");
        let (tx, rx) = channel();
        serve(&path, tx).unwrap();
        answer_with(rx, |req| {
            assert_eq!(*req, Request::Status);
            Response::Status { status: status() }
        });
        let got = request(&path, &Request::Status).unwrap();
        assert_eq!(got, Some(Response::Status { status: status() }));
    }

    #[test]
    fn job_requests_round_trip() {
        let dir = tmp();
        let path = dir.path().join("k.sock");
        let (tx, rx) = channel();
        serve(&path, tx).unwrap();
        answer_with(rx, |req| Response::Ok {
            message: serde_json::to_string(req).unwrap(),
        });
        let req = Request::Job {
            job: Job::Retranscribe {
                id: "s1".into(),
                model: None,
                languages: crate::store::LanguageChange {
                    me: Some(vec!["ru".into()]),
                    ..Default::default()
                },
            },
        };
        let Some(Response::Ok { message }) = request(&path, &req).unwrap() else {
            panic!("expected an Ok reply");
        };
        assert_eq!(serde_json::from_str::<Request>(&message).unwrap(), req);
    }

    #[test]
    fn no_daemon_means_none() {
        let dir = tmp();
        let got = request(&dir.path().join("missing.sock"), &Request::Status).unwrap();
        assert_eq!(got, None);
    }

    #[test]
    fn stale_socket_is_replaced_and_a_live_one_refused() {
        let dir = tmp();
        let path = dir.path().join("k.sock");
        drop(UnixListener::bind(&path).unwrap());
        assert!(path.exists());
        let (tx, _rx) = channel();
        serve(&path, tx.clone()).unwrap();
        let err = serve(&path, tx).unwrap_err();
        assert!(err.to_string().contains("already running"), "{err:#}");
    }

    #[test]
    fn socket_is_private() {
        let dir = tmp();
        let path = dir.path().join("k.sock");
        let (tx, _rx) = channel();
        serve(&path, tx).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn bad_json_gets_an_error_reply() {
        let dir = tmp();
        let path = dir.path().join("k.sock");
        let (tx, _rx) = channel();
        serve(&path, tx).unwrap();
        let mut stream = UnixStream::connect(&path).unwrap();
        stream.write_all(b"nonsense\n").unwrap();
        let mut line = String::new();
        BufReader::new(&stream).read_line(&mut line).unwrap();
        assert!(matches!(
            serde_json::from_str::<Response>(&line).unwrap(),
            Response::Error { .. }
        ));
    }

    #[test]
    fn a_silent_daemon_gets_an_error_reply() {
        let dir = tmp();
        let path = dir.path().join("k.sock");
        let (tx, rx) = channel();
        serve_with_timeout(&path, tx, Duration::from_millis(100)).unwrap();
        // Keep the receiver alive but never answer.
        let _keep = rx;
        let got = request(&path, &Request::Stop).unwrap();
        assert_eq!(
            got,
            Some(Response::Error {
                message: "the daemon did not answer".into()
            })
        );
    }
}
