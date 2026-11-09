//! Runs any command: the prompt goes to stdin, stdout is the summary.

use super::Summarizer;
use crate::config::expand_tilde;
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
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

impl CommandSummarizer {
    /// The program and its arguments, with a leading `~` expanded as in `transcribe.command`.
    fn command_line(&self) -> (PathBuf, Vec<PathBuf>) {
        let expand = |a: &String| expand_tilde(Path::new(a));
        (
            expand(&self.argv[0]),
            self.argv[1..].iter().map(expand).collect(),
        )
    }
}

impl Summarizer for CommandSummarizer {
    fn name(&self) -> String {
        format!("command: {}", self.argv.join(" "))
    }

    fn summarize(&self, prompt: &str) -> Result<String> {
        let program = &self.argv[0];
        let (path, args) = self.command_line();
        let mut child = Command::new(path)
            .args(args)
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
            let killed = unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL) } == 0;
            if !killed {
                let _ = child.kill();
            }
            let _ = child.wait();
            // Don't join the helper threads: a descendant that left the group (setsid, a
            // double fork) may still hold the pipes. They exit on their own once the pipes close.
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
        // Poll briefly: the killed grandchild may linger as a zombie until init reaps it.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while process_alive(pid) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!process_alive(pid), "grandchild {pid} survived");
    }

    #[test]
    fn timeout_returns_even_if_a_descendant_escapes_the_group() {
        // The setsid'd sleep leaves the process group and keeps the stdout/stderr pipes open.
        let tmp = tempfile::tempdir().unwrap();
        let pidfile = tmp.path().join("pid");
        let script = format!("setsid sleep 30 & echo $! > {}; wait", pidfile.display());
        let started = std::time::Instant::now();
        let result = sh(&script, 300).summarize("prompt");
        let elapsed = started.elapsed();
        let pid: i32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // SAFETY: plain syscall on the PID we recorded for the escaped sleep.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
        assert!(result.is_err());
        assert!(elapsed < Duration::from_secs(3), "blocked for {elapsed:?}");
    }

    /// Alive means present in /proc and not a zombie.
    fn process_alive(pid: i32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            // The state letter follows the parenthesised command name.
            Ok(stat) => stat
                .rsplit_once(')')
                .is_some_and(|(_, rest)| !rest.trim_start().starts_with('Z')),
            Err(_) => false,
        }
    }

    #[test]
    fn a_leading_tilde_is_expanded() {
        let s = CommandSummarizer::new(
            vec!["~/bin/summarize".into(), "~/prompt.md".into(), "-p".into()],
            Duration::from_secs(1),
        );
        let home = dirs::home_dir().unwrap();
        assert_eq!(
            s.command_line(),
            (
                home.join("bin/summarize"),
                vec![home.join("prompt.md"), PathBuf::from("-p")]
            )
        );
    }

    #[test]
    fn missing_program_is_reported() {
        let s = CommandSummarizer::new(vec!["kute-no-such-program".into()], Duration::from_secs(1));
        assert!(format!("{:#}", s.summarize("x").unwrap_err()).contains("kute-no-such-program"));
    }
}
