//! Black-box checks of the CLI that need no models.

use std::process::Command;

/// Runs kute with the test's own runtime dir, so it never finds a real daemon's socket.
fn kute(config: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kute"))
        .env("XDG_RUNTIME_DIR", config.parent().unwrap())
        .arg("--config")
        .arg(config)
        .args(args)
        .output()
        .unwrap()
}

fn setup() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp.path().join("config.toml");
    std::fs::write(
        &config,
        format!("[storage]\ndir = {:?}\n", tmp.path().join("data")),
    )
    .unwrap();
    (tmp, config)
}

#[test]
fn help_lists_commands() {
    let out = Command::new(env!("CARGO_BIN_EXE_kute"))
        .arg("--help")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    for cmd in [
        "import",
        "process",
        "list",
        "show",
        "search",
        "title",
        "summarize",
        "retranscribe",
        "reindex",
        "model",
        "daemon",
        "status",
        "record",
        "stop",
        "discard",
        "pause",
        "resume",
    ] {
        assert!(text.contains(cmd), "missing {cmd} in:\n{text}");
    }
}

#[test]
fn empty_archive() {
    let (_tmp, config) = setup();
    let out = kute(&config, &["list"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("no sessions"));
    let out = kute(&config, &["search", "anything"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("no matches"));
    assert!(!kute(&config, &["show", "last"]).status.success());
}

#[test]
fn model_list_marks_missing_models() {
    let (_tmp, config) = setup();
    let out = kute(&config, &["model", "list"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("large-v3-turbo")
            && text.contains("(default)")
            && text.contains("silero-v6.2.0"),
        "{text}"
    );
}

#[test]
fn import_without_models_explains_what_to_do() {
    let (tmp, config) = setup();
    let me = tmp.path().join("me.wav");
    std::fs::copy(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/me.wav"),
        &me,
    )
    .unwrap();
    let out = kute(
        &config,
        &[
            "import",
            me.to_str().unwrap(),
            me.to_str().unwrap(),
            "--start",
            "2026-10-08T15:00:00",
        ],
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("kute model download"));
    // Nothing was imported, so nothing is half-processed.
    assert!(String::from_utf8_lossy(&kute(&config, &["list"]).stdout).contains("no sessions"));
}

#[test]
fn list_shows_newest_first() {
    use kute::store::{Meta, Store, Trigger, TriggerKind};
    let (tmp, config) = setup();
    let store = Store::open(tmp.path().join("data")).unwrap();
    for (start, title) in [
        ("2026-10-08T09:00:00+02:00[+02:00]", "older"),
        ("2026-10-09T09:00:00+02:00[+02:00]", "newer"),
    ] {
        let start: jiff::Zoned = start.parse().unwrap();
        let session = store.create_session(&start).unwrap();
        let trigger = Trigger {
            kind: TriggerKind::Import,
            app_name: None,
            binary: None,
        };
        let mut meta = Meta::new(session.id.clone(), start, trigger, "en".into());
        meta.title = Some(title.into());
        session.save_meta(&meta).unwrap();
    }
    let out = String::from_utf8_lossy(&kute(&config, &["list"]).stdout).into_owned();
    let newer = out.find("newer").expect(&out);
    assert!(newer < out.find("older").expect(&out), "{out}");
}

#[test]
fn control_commands_without_a_daemon() {
    let (_tmp, config) = setup();
    for args in [
        &["status"][..],
        &["record", "Standup"],
        &["stop"],
        &["discard"],
        &["pause"],
        &["resume"],
    ] {
        let out = kute(&config, args);
        assert!(!out.status.success(), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains("the kute daemon is not running (start it with `kute daemon`)"),
            "{args:?}: {err}"
        );
    }
}

/// A stand-in daemon on the socket the CLI looks for in `runtime_dir`. It runs with the
/// config file `config` (`None`: the defaults) on `archive`, and queues every job it gets,
/// sending each one on the returned channel.
fn stand_in_daemon(
    runtime_dir: &std::path::Path,
    config: Option<&std::path::Path>,
    archive: &std::path::Path,
) -> std::sync::mpsc::Receiver<kute::jobs::Job> {
    use kute::ipc::{Request, Response};
    let (tx, rx) = std::sync::mpsc::channel::<kute::ipc::Incoming>();
    kute::ipc::serve(&runtime_dir.join("kute.sock"), tx).unwrap();
    let config = config.map(|c| std::fs::canonicalize(c).unwrap());
    let archive = archive.to_path_buf();
    let (jobs, queued) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for incoming in rx {
            let reply = match incoming.request {
                Request::Identity => Response::Identity {
                    config: config.clone(),
                    archive: archive.clone(),
                },
                Request::Job { job } => {
                    let message = format!("queued {job}");
                    let _ = jobs.send(job);
                    Response::Ok { message }
                }
                other => panic!("unexpected {other:?}"),
            };
            let _ = incoming.reply.send(reply);
        }
    });
    queued
}

#[test]
fn a_job_queued_on_the_daemon_says_where_its_result_goes() {
    let (tmp, config) = setup();
    let queued = stand_in_daemon(tmp.path(), Some(&config), &tmp.path().join("data"));
    let out = kute(&config, &["reindex"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "queued reindex; the daemon logs the result (journalctl --user -u kute, or the \
         terminal running kute daemon)\n"
    );
    assert_eq!(queued.try_recv().unwrap(), kute::jobs::Job::Reindex);
}

#[test]
fn a_job_with_another_config_and_archive_runs_here() {
    let (tmp, config) = setup();
    let daemon_dir = tempfile::tempdir().unwrap();
    let queued = stand_in_daemon(tmp.path(), None, daemon_dir.path());
    let out = kute(&config, &["reindex"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!String::from_utf8_lossy(&out.stdout).contains("daemon"));
    assert!(queued.try_recv().is_err(), "the daemon got the job");
}

#[test]
fn a_job_with_another_config_on_the_daemons_archive_is_refused() {
    let (tmp, config) = setup();
    let queued = stand_in_daemon(tmp.path(), None, &tmp.path().join("data"));
    let out = kute(&config, &["reindex"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("the running daemon uses the default settings on this archive"),
        "{err}"
    );
    assert!(queued.try_recv().is_err(), "the daemon got the job");
}

#[test]
fn output_into_a_closed_pipe_ends_quietly() {
    use kute::store::{Meta, Store, Trigger, TriggerKind};
    use std::io::Read;
    let (tmp, config) = setup();
    let store = Store::open(tmp.path().join("data")).unwrap();
    let start: jiff::Zoned = "2026-10-08T09:00:00+02:00[+02:00]".parse().unwrap();
    let session = store.create_session(&start).unwrap();
    let trigger = Trigger {
        kind: TriggerKind::Import,
        app_name: None,
        binary: None,
    };
    session
        .save_meta(&Meta::new(session.id.clone(), start, trigger, "en".into()))
        .unwrap();
    // Far more than a pipe holds, so kute is still writing when the reader goes away.
    std::fs::write(
        session.transcript_path(),
        "[00:00] Me: hello\n".repeat(100_000),
    )
    .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_kute"))
        .env("XDG_RUNTIME_DIR", tmp.path())
        .arg("--config")
        .arg(&config)
        .args(["show", "last"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    stdout.read_exact(&mut [0; 16]).unwrap();
    drop(stdout); // like `| head`
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    let status = child.wait().unwrap();
    assert!(stderr.is_empty(), "{stderr}");
    // Ended by SIGPIPE, like any other tool at the head of a closed pipe, not a panic (101).
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.signal(), Some(libc::SIGPIPE), "{status}");
}
