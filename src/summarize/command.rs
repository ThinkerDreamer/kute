//! Runs any command: the prompt goes to stdin, stdout is the summary.

use super::Summarizer;
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use wait_timeout::ChildExt;

pub struct CommandSummarizer {
    argv: Vec<String>,
    timeout: Duration,
}

impl CommandSummarizer {
    pub fn new(argv: Vec<String>, timeout: Duration) -> CommandSummarizer {
        CommandSummarizer { argv, timeout }
    }
}

impl Summarizer for CommandSummarizer {
    fn name(&self) -> String {
        format!("command: {}", self.argv.join(" "))
    }

    fn summarize(&self, prompt: &str) -> Result<String> {
        let program = &self.argv[0];
        let mut child = Command::new(program)
            .args(&self.argv[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Its own process group, so a timeout can take down grandchildren too.
            .process_group(0)
            .spawn()
            .with_context(|| format!("starting {program:?}"))?;

        // Feed stdin and drain stdout/stderr on their own threads so a large prompt or
        // a chatty command can't fill a pipe and deadlock.
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let input = prompt.to_owned();
        let writer = thread::spawn(move || {
            let _ = stdin.write_all(input.as_bytes());
        });
        let mut stdout = child.stdout.take().expect("stdout is piped");
        let out_reader = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        });
        let mut stderr = child.stderr.take().expect("stderr is piped");
        let err_reader = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });

        let Some(status) = child.wait_timeout(self.timeout)? else {
            // SAFETY: plain syscall. The child leads the group we created with
            // `process_group(0)`, so its pid is the group id and `-pid` targets that group.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
            // The group is dead, so the pipes are closed and the helper threads finish.
            let _ = writer.join();
            let _ = out_reader.join();
            let _ = err_reader.join();
            bail!(
                "{program} did not finish within {} s",
                self.timeout.as_secs()
            );
        };
        let _ = writer.join();
        let out = out_reader.join().unwrap_or_default();
        let err = err_reader.join().unwrap_or_default();
        if !status.success() {
            let err = String::from_utf8_lossy(&err);
            let tail: String = err
                .trim()
                .chars()
                .rev()
                .take(2_000)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            bail!("{program} failed ({status}): {tail}");
        }
        let text = String::from_utf8_lossy(&out).trim().to_string();
        if text.is_empty() {
            bail!("{program} produced no output");
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str, timeout_ms: u64) -> CommandSummarizer {
        CommandSummarizer::new(
            vec!["sh".into(), "-c".into(), script.into()],
            Duration::from_millis(timeout_ms),
        )
    }

    #[test]
    fn stdout_becomes_the_summary() {
        assert_eq!(sh("cat", 5_000).summarize("  notes  \n").unwrap(), "notes");
    }

    #[test]
    fn large_prompts_do_not_deadlock() {
        let big = "word ".repeat(400_000);
        assert_eq!(sh("wc -w", 10_000).summarize(&big).unwrap(), "400000");
    }

    #[test]
    fn failure_includes_stderr() {
        let err = sh("echo oops >&2; exit 3", 5_000)
            .summarize("x")
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("oops") && msg.contains("exit"), "{msg}");
    }

    #[test]
    fn empty_output_is_an_error() {
        assert!(
            format!(
                "{:#}",
                sh("cat >/dev/null", 5_000).summarize("x").unwrap_err()
            )
            .contains("no output")
        );
    }

    #[test]
    fn hanging_command_times_out() {
        let started = std::time::Instant::now();
        let err = sh("sleep 30", 300).summarize("x").unwrap_err();
        assert!(format!("{err:#}").contains("did not finish"), "{err:#}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn timeout_kills_grandchildren() {
        // sh starts a background sleep that writes its PID, then waits. After the timeout the
        // grandchild must be gone too.
        let tmp = tempfile::tempdir().unwrap();
        let pidfile = tmp.path().join("pid");
        let script = format!("sleep 30 & echo $! > {}; wait", pidfile.display());
        let s = sh(&script, 500);
        assert!(s.summarize("prompt").is_err());
        let pid: i32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            !std::path::Path::new(&format!("/proc/{pid}")).exists(),
            "grandchild {pid} survived"
        );
    }

    #[test]
    fn missing_program_is_reported() {
        let s = CommandSummarizer::new(vec!["kute-no-such-program".into()], Duration::from_secs(1));
        assert!(format!("{:#}", s.summarize("x").unwrap_err()).contains("kute-no-such-program"));
    }
}
