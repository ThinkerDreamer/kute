//! Any program as the `Transcriber`: kute starts it once per transcription run and talks to it
//! in JSON lines, so a model that is slow to load (Qwen3-ASR, say) loads only once.
//!
//! The protocol, all on the program's stdin and stdout:
//!
//! - The program first prints one line, e.g.
//!   `{"name": "qwen3-asr-1.7b", "max_chunk_ms": 60000, "join_gap_ms": 10000}`. `name` is
//!   recorded in `meta.toml`; `max_chunk_ms` (optional) caps the audio per request, and
//!   `join_gap_ms` (optional) joins speech regions closer than that into one chunk.
//! - kute sends a request line followed by the audio: `samples` little-endian `f32` values,
//!   16 kHz mono, as raw bytes.
//!   - `{"op": "detect", "candidates": ["ru", "en"], "samples": N}` answers
//!     `{"language": "ru"}`. Empty candidates means any language.
//!   - `{"op": "transcribe", "language": "ru", "samples": N}` answers
//!     `{"segments": [{"start_ms", "end_ms", "text", "words": [{"start_ms", "end_ms", "text"}]}]}`
//!     with times relative to the audio sent.
//! - Either answer can instead be `{"error": "..."}`.
//! - kute closes stdin when it is done; the program should then exit.
//!
//! Language codes are the ones kute uses everywhere (`en`, `ru`, ...). The program's stderr
//! goes to kute's stderr (the journal, for the daemon).

use super::{CHUNK_JOIN_GAP_MS, CHUNK_MAX_MS, RawSegment, Transcriber};
use crate::config::expand_tilde;
use crate::store::Word;
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread;
use std::time::Duration;
use wait_timeout::ChildExt;

/// How long the program may take to load its model and print its first line.
const START_TIMEOUT: Duration = Duration::from_secs(600);
/// How long one request may take.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// How long the program gets to exit after stdin closes.
const EXIT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct CommandTranscriber {
    program: String,
    child: Child,
    /// Requests for the writer thread; dropping it closes the program's stdin.
    requests: Option<Sender<Vec<u8>>>,
    lines: Receiver<std::io::Result<String>>,
    name: String,
    max_chunk_ms: u64,
    join_gap_ms: u64,
    request_timeout: Duration,
}

#[derive(Deserialize)]
struct Hello {
    name: String,
    #[serde(default)]
    max_chunk_ms: Option<u64>,
    #[serde(default)]
    join_gap_ms: Option<u64>,
}

#[derive(Serialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum Request<'a> {
    Detect {
        candidates: &'a [String],
        samples: usize,
    },
    Transcribe {
        language: &'a str,
        samples: usize,
    },
}

#[derive(Deserialize)]
struct Reply {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    segments: Option<Vec<ReplySegment>>,
}

#[derive(Deserialize)]
struct ReplySegment {
    start_ms: u64,
    end_ms: u64,
    text: String,
    #[serde(default)]
    words: Vec<ReplyWord>,
}

#[derive(Deserialize)]
struct ReplyWord {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

impl CommandTranscriber {
    /// Starts `argv` and waits for its first line, which comes once its model has loaded. A
    /// leading `~` in any argument is the home directory.
    pub fn start(argv: &[String]) -> Result<CommandTranscriber> {
        Self::start_with(argv, START_TIMEOUT, REQUEST_TIMEOUT)
    }

    fn start_with(
        argv: &[String],
        timeout: Duration,
        request_timeout: Duration,
    ) -> Result<CommandTranscriber> {
        let (first, rest) = argv.split_first().context("transcribe.command is empty")?;
        let program = expand_tilde(Path::new(first));
        let name = program.display().to_string();
        let mut child = Command::new(&program)
            .args(rest.iter().map(|a| expand_tilde(Path::new(a))))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            // Its own process group, so stopping it also stops anything it started (a
            // wrapper script's Python, say) and the model never outlives the run.
            .process_group(0)
            .spawn()
            .with_context(|| format!("starting {name:?}"))?;
        let stdout = child.stdout.take().expect("stdout is piped");
        // Threads write the requests and read the replies, so a program that hangs (stops
        // reading, or stops answering) hits a timeout and is killed instead of blocking kute.
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let (requests, to_write) = channel::<Vec<u8>>();
        thread::spawn(move || {
            for buf in to_write {
                if stdin.write_all(&buf).and_then(|()| stdin.flush()).is_err() {
                    break;
                }
            }
        });
        let (tx, lines) = channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut t = CommandTranscriber {
            program: name,
            requests: Some(requests),
            child,
            lines,
            name: String::new(),
            max_chunk_ms: CHUNK_MAX_MS,
            join_gap_ms: CHUNK_JOIN_GAP_MS,
            request_timeout,
        };
        let line = t
            .read_line(timeout)
            .context("waiting for the program to start")?;
        let hello: Hello = serde_json::from_str(&line)
            .with_context(|| format!("{}: bad first line {line:?}", t.program))?;
        t.name = hello.name;
        if let Some(ms) = hello.max_chunk_ms {
            if ms < 1_000 {
                bail!("{}: max_chunk_ms {ms} is too small", t.program);
            }
            t.max_chunk_ms = ms;
        }
        if let Some(ms) = hello.join_gap_ms {
            t.join_gap_ms = ms;
        }
        Ok(t)
    }

    /// Kills the program and everything in its process group.
    fn kill(&mut self) {
        // SAFETY: plain syscall. The child leads the group created with `process_group(0)`,
        // so its pid is the group id and `-pid` targets that group.
        let killed = unsafe { libc::kill(-(self.child.id() as i32), libc::SIGKILL) } == 0;
        if !killed {
            let _ = self.child.kill();
        }
    }

    fn read_line(&mut self, timeout: Duration) -> Result<String> {
        match self.lines.recv_timeout(timeout) {
            Ok(Ok(line)) => Ok(line),
            Ok(Err(e)) => Err(anyhow!(e).context(format!("reading from {}", self.program))),
            Err(RecvTimeoutError::Timeout) => {
                self.kill();
                bail!(
                    "{} did not answer within {} s",
                    self.program,
                    timeout.as_secs()
                )
            }
            Err(RecvTimeoutError::Disconnected) => {
                // stdout closed; the program has normally exited, but don't wait forever for
                // one that keeps running without it.
                match self.child.wait_timeout(EXIT_TIMEOUT) {
                    Ok(Some(status)) => bail!("{} exited ({status})", self.program),
                    _ => {
                        self.kill();
                        bail!("{} closed its output", self.program)
                    }
                }
            }
        }
    }

    fn request(&mut self, request: &Request, samples: &[f32]) -> Result<Reply> {
        let mut buf = serde_json::to_vec(request)?;
        buf.push(b'\n');
        buf.reserve(samples.len() * 4);
        for s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        // A writer that has stopped means the program closed stdin or exited; reading the
        // reply below reports which.
        if let Some(requests) = &self.requests {
            let _ = requests.send(buf);
        } else {
            bail!("{} is closed", self.program);
        }
        let line = self.read_line(self.request_timeout)?;
        let reply: Reply = serde_json::from_str(&line)
            .with_context(|| format!("{}: bad reply {line:?}", self.program))?;
        if let Some(e) = reply.error {
            bail!("{}: {e}", self.program);
        }
        Ok(reply)
    }
}

impl Transcriber for CommandTranscriber {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn max_chunk_ms(&self) -> u64 {
        self.max_chunk_ms
    }

    fn chunk_join_gap_ms(&self) -> u64 {
        self.join_gap_ms
    }

    fn detect_language(&mut self, samples: &[f32], candidates: &[String]) -> Result<String> {
        let reply = self.request(
            &Request::Detect {
                candidates,
                samples: samples.len(),
            },
            samples,
        )?;
        let language = reply
            .language
            .with_context(|| format!("{}: detect reply has no language", self.program))?;
        let Some(code) = super::whisper::language_code(&language) else {
            bail!("{}: unknown language {language:?}", self.program);
        };
        if !candidates.is_empty() && !candidates.iter().any(|c| c == code) {
            bail!(
                "{}: detected {code}, which is not one of {}",
                self.program,
                candidates.join(",")
            );
        }
        Ok(code.to_string())
    }

    fn transcribe(&mut self, samples: &[f32], language: &str) -> Result<Vec<RawSegment>> {
        let reply = self.request(
            &Request::Transcribe {
                language,
                samples: samples.len(),
            },
            samples,
        )?;
        let segments = reply
            .segments
            .with_context(|| format!("{}: transcribe reply has no segments", self.program))?;
        Ok(segments
            .into_iter()
            .map(|s| RawSegment {
                start_ms: s.start_ms,
                end_ms: s.end_ms.max(s.start_ms),
                text: s.text,
                words: s
                    .words
                    .into_iter()
                    .map(|w| Word {
                        start_ms: w.start_ms,
                        end_ms: w.end_ms.max(w.start_ms),
                        text: w.text,
                    })
                    .collect(),
            })
            .collect())
    }
}

impl Drop for CommandTranscriber {
    fn drop(&mut self) {
        // Closing stdin asks the program to exit; one that doesn't is killed, so its model
        // never stays in VRAM.
        self.requests = None;
        let exited = matches!(self.child.wait_timeout(EXIT_TIMEOUT), Ok(Some(_)));
        // Also when it exited: something it started may still be running in its group.
        self.kill();
        if !exited {
            let _ = self.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake helper in Python: echoes the request back in its replies.
    const FAKE: &str = r#"#!/usr/bin/env python3
import json, struct, sys
out = sys.stdout
print(json.dumps({"name": "fake-asr", "max_chunk_ms": 30000, "join_gap_ms": 8000}), flush=True)
inp = sys.stdin.buffer
while True:
    line = inp.readline()
    if not line:
        break
    req = json.loads(line)
    data = inp.read(4 * req["samples"])
    n = len(data) // 4
    peak = max(struct.unpack("<%df" % n, data)) if n else 0.0
    if req["op"] == "detect":
        lang = req["candidates"][-1] if req["candidates"] else "russian"
        print(json.dumps({"language": lang}), flush=True)
    elif req["language"] == "xx":
        print(json.dumps({"error": "no such language"}), flush=True)
    else:
        text = "%s %d %.1f" % (req["language"], n, peak)
        seg = {"start_ms": 10, "end_ms": 900, "text": text,
               "words": [{"start_ms": 10, "end_ms": 900, "text": text}]}
        print(json.dumps({"segments": [seg]}), flush=True)
"#;

    /// Writes a helper script and returns the command that runs it. The interpreter runs it,
    /// rather than the script itself, so a test thread forking at the same time can't make
    /// exec fail with "text file busy".
    fn fake(dir: &Path, body: &str) -> Vec<String> {
        let path = dir.join("fake-asr");
        std::fs::write(&path, body).unwrap();
        let interpreter = if body.starts_with("#!/usr/bin/env python3") {
            "python3"
        } else {
            "/bin/sh"
        };
        vec![interpreter.into(), path.to_string_lossy().into_owned()]
    }

    /// Whether `pid` is running: not gone, and not a zombie nobody has reaped yet.
    fn running(pid: i32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat
                .rsplit_once(')')
                .is_some_and(|(_, rest)| !rest.trim_start().starts_with('Z')),
            Err(_) => false,
        }
    }

    fn have_python() -> bool {
        Command::new("python3")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    #[test]
    fn talks_to_a_helper() {
        if !have_python() {
            eprintln!("skipping: no python3");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut t = CommandTranscriber::start(&fake(tmp.path(), FAKE)).unwrap();
        assert_eq!(t.model_name(), "fake-asr");
        assert_eq!(t.max_chunk_ms(), 30_000);
        assert_eq!(t.chunk_join_gap_ms(), 8_000);

        let samples = vec![0.5f32; 16_000];
        let segs = t.transcribe(&samples, "ru").unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "ru 16000 0.5");
        assert_eq!((segs[0].start_ms, segs[0].end_ms), (10, 900));
        assert_eq!(segs[0].words.len(), 1);

        // Names are mapped to codes, and the answer must be a candidate.
        assert_eq!(t.detect_language(&samples, &[]).unwrap(), "ru");
        assert_eq!(
            t.detect_language(&samples, &["ru".into(), "en".into()])
                .unwrap(),
            "en"
        );

        let err = t.transcribe(&samples, "xx").unwrap_err();
        assert!(format!("{err:#}").contains("no such language"), "{err:#}");
        // The helper is still usable after an error reply.
        assert_eq!(t.transcribe(&[], "en").unwrap()[0].text, "en 0 0.0");
    }

    #[test]
    fn a_helper_that_stops_reading_times_out() {
        let tmp = tempfile::tempdir().unwrap();
        let script = "#!/bin/sh\necho '{\"name\": \"stuck\"}'\nexec sleep 30\n";
        let mut t = CommandTranscriber::start_with(
            &fake(tmp.path(), script),
            START_TIMEOUT,
            Duration::from_millis(300),
        )
        .unwrap();
        // Far more than a pipe holds, so a blocking write would never return.
        let err = t.transcribe(&vec![0.0; 2_000_000], "en").unwrap_err();
        assert!(format!("{err:#}").contains("did not answer"), "{err:#}");
    }

    #[test]
    fn a_helper_that_closes_its_output_but_keeps_running_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let script = "#!/bin/sh\necho '{\"name\": \"quiet\"}'\nexec sleep 30 >&-\n";
        let mut t = CommandTranscriber::start(&fake(tmp.path(), script)).unwrap();
        let err = t.transcribe(&[0.0; 10], "en").unwrap_err();
        assert!(format!("{err:#}").contains("closed its output"), "{err:#}");
    }

    #[test]
    fn stopping_a_helper_stops_what_it_started() {
        let tmp = tempfile::tempdir().unwrap();
        let pidfile = tmp.path().join("pid");
        let script = format!(
            "#!/bin/sh\nsleep 300 &\necho $! > {}\necho '{{\"name\": \"wrapper\"}}'\ncat > /dev/null\n",
            pidfile.display()
        );
        let t = CommandTranscriber::start(&fake(tmp.path(), &script)).unwrap();
        let pid: i32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        drop(t);
        let mut gone = false;
        for _ in 0..50 {
            if !running(pid) {
                gone = true;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(gone, "the helper's child {pid} is still running");
    }

    #[test]
    fn a_helper_that_exits_or_hangs_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let err = CommandTranscriber::start(&fake(tmp.path(), "#!/bin/sh\nexit 3\n"))
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("exited"), "{err:#}");

        let err = CommandTranscriber::start_with(
            &fake(tmp.path(), "#!/bin/sh\nexec sleep 30\n"),
            Duration::from_millis(200),
            REQUEST_TIMEOUT,
        )
        .err()
        .unwrap();
        assert!(format!("{err:#}").contains("did not answer"), "{err:#}");

        let err = CommandTranscriber::start(&fake(tmp.path(), "#!/bin/sh\necho hello\n"))
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("bad first line"), "{err:#}");

        let err = CommandTranscriber::start(&["/nonexistent/kute-helper".into()])
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("starting"), "{err:#}");
    }
}
