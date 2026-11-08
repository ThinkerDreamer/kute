//! `kute service install` and `kute service uninstall`: the systemd user unit that runs
//! `kute daemon` at login.

use crate::detect::Phase;
use crate::ipc::{self, Request, Response, Status};
use anyhow::{Context, Result, bail};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The unit's name, as `systemctl --user` knows it.
pub const UNIT: &str = "kute.service";

/// The first line of every unit kute writes. A unit without it was written by hand, and is
/// only replaced with `--force`.
const MARKER: &str = "# Written by `kute service install`; it rewrites this file.";

/// What goes into the unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitSpec {
    /// The `kute` binary the daemon runs.
    pub exe: PathBuf,
    /// The daemon's `PATH`: where it finds pw-record, ffmpeg and a summarizer command.
    pub path: String,
    /// The optional file with the daemon's secrets, as the unit names it.
    pub env_file: String,
}

impl UnitSpec {
    /// The unit for this binary and the caller's `PATH`. `argv0` is how kute was started:
    /// see `daemon_exe`.
    pub fn current(argv0: Option<&OsStr>) -> Result<UnitSpec> {
        let shell_path = std::env::var_os("PATH").unwrap_or_default();
        let exe = daemon_exe(argv0, &shell_path)?;
        let home = dirs::home_dir().context("cannot find your home folder")?;
        let config = dirs::config_dir().context("cannot find your config folder")?;
        Ok(UnitSpec {
            path: daemon_path(&shell_path, &home, exe.parent()),
            env_file: env_file(&config.join("kute").join("env"), &home),
            exe,
        })
    }

    /// The unit file's text.
    pub fn render(&self) -> String {
        format!(
            "{MARKER}
[Unit]
Description=kute: record, transcribe and summarize calls
Documentation=https://github.com/ThinkerDreamer/kute
After=pipewire.service wireplumber.service
Wants=pipewire.service

[Service]
ExecStart={exe} daemon
Restart=on-failure
RestartSec=5
# A stop ends the current recording; the next start finishes what it left.
TimeoutStopSec=10
Environment={path}
# API keys for a summarizer, e.g. ANTHROPIC_API_KEY=...; keep the file private (chmod 600).
EnvironmentFile=-{env_file}

[Install]
WantedBy=default.target
",
            exe = quote(&self.exe.to_string_lossy()),
            path = quote(&format!("PATH={}", self.path)),
            env_file = self.env_file,
        )
    }
}

/// The binary the daemon should run: the one running now. When kute was started through a
/// link that leads to this binary (`~/.nix-profile/bin/kute`, `~/.cargo/bin/kute`), the link
/// is kept, so an update that moves the binary (a new Nix store path) still finds it.
fn daemon_exe(argv0: Option<&OsStr>, shell_path: &OsStr) -> Result<PathBuf> {
    let exe = std::env::current_exe().context("cannot find the running kute binary")?;
    let real = exe.canonicalize().unwrap_or_else(|_| exe.clone());
    let started_as = argv0.and_then(|a| find_program(Path::new(a), shell_path));
    Ok(match started_as {
        Some(link) if link.canonicalize().is_ok_and(|l| l == real) => link,
        _ => real,
    })
}

/// Where `program` (as typed: a bare name searched on `shell_path`, or a path) is, as an
/// absolute path without resolving links.
fn find_program(program: &Path, shell_path: &OsStr) -> Option<PathBuf> {
    if program.components().count() > 1 {
        return std::path::absolute(program).ok();
    }
    crate::program::find(program, shell_path)
}

/// The daemon's `PATH`: `~/.local/bin`, `~/.cargo/bin` and the binary's own folder first
/// (summarizer commands live there), then the `PATH` of the shell that ran the install,
/// which knows where pw-record and ffmpeg are on this system. No duplicates, and no
/// relative entries, which mean nothing to a service.
fn daemon_path(shell_path: &OsStr, home: &Path, exe_dir: Option<&Path>) -> String {
    let first = [home.join(".local/bin"), home.join(".cargo/bin")];
    let mut dirs: Vec<PathBuf> = Vec::new();
    let candidates = first
        .into_iter()
        .chain(exe_dir.map(Path::to_path_buf))
        .chain(std::env::split_paths(shell_path));
    for dir in candidates {
        if dir.is_absolute() && !dir.as_os_str().is_empty() && !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }
    if !dirs.iter().any(|d| d == Path::new("/usr/bin")) {
        dirs.push("/usr/bin".into());
    }
    dirs.iter()
        .map(|d| d.to_string_lossy())
        .collect::<Vec<_>>()
        .join(":")
}

/// The secrets file as the unit names it: under `%h` when it is in the home folder, so the
/// unit reads the same for everyone.
fn env_file(file: &Path, home: &Path) -> String {
    match file.strip_prefix(home) {
        Ok(rest) => format!("%h/{}", rest.to_string_lossy().replace('%', "%%")),
        Err(_) => file.to_string_lossy().replace('%', "%%"),
    }
}

/// One word on a systemd command or assignment line: `%` doubled (systemd expands `%h` and
/// friends), and in double quotes with `\` and `"` escaped when it has spaces or quotes.
fn quote(word: &str) -> String {
    let word = word.replace('%', "%%");
    if !word.contains([' ', '\t', '"', '\'', '\\']) {
        return word;
    }
    format!("\"{}\"", word.replace('\\', "\\\\").replace('"', "\\\""))
}

/// `~/.config/systemd/user/kute.service` (honouring `XDG_CONFIG_HOME`).
pub fn unit_path() -> Result<PathBuf> {
    let config = dirs::config_dir().context("cannot find your config folder")?;
    Ok(config.join("systemd").join("user").join(UNIT))
}

/// What installing does to the unit file that is there.
#[derive(Debug, PartialEq, Eq)]
pub enum Write {
    /// No unit yet.
    Create,
    /// The unit is already this one.
    Same,
    /// A unit kute wrote earlier, for another binary or an older kute.
    Update,
    /// A unit written by hand: moved to `kute.service.bak`, with `--force` only.
    Replace,
}

pub fn plan_write(existing: Option<&str>, new: &str, force: bool) -> Result<Write> {
    Ok(match existing {
        None => Write::Create,
        Some(old) if old == new => Write::Same,
        Some(old) if old.starts_with(MARKER) => Write::Update,
        Some(_) if force => Write::Replace,
        Some(_) => bail!(
            "a kute.service that kute didn't write is already installed; \
             `kute service install --force` replaces it and keeps the old one as kute.service.bak \
             (`--print` shows the new one)"
        ),
    })
}

/// What to do about a daemon that is running when the unit changes.
#[derive(Debug, PartialEq, Eq)]
pub enum Restart {
    /// Nothing runs: start the service.
    Start,
    /// The service runs and the unit is the same: leave it.
    Keep,
    /// The service runs an older unit and is idle: restart it now.
    Now,
    /// The service is recording or working: restarting would cut the recording in two.
    Later(String),
    /// A daemon runs outside systemd (in a terminal): starting the service would fail.
    Foreign,
}

/// `active` is whether systemd runs the service; `status` is the daemon's answer on the
/// control socket, `None` when nothing answers.
pub fn plan_restart(active: bool, changed: bool, status: Option<&Status>) -> Restart {
    match (active, status) {
        (false, Some(_)) => Restart::Foreign,
        (false, None) => Restart::Start,
        (true, _) if !changed => Restart::Keep,
        (true, Some(s)) if s.phase != Phase::Idle => Restart::Later(format!(
            "it may be recording a call ({}, {})",
            format!("{:?}", s.phase).to_lowercase(),
            s.session.as_deref().unwrap_or("no session")
        )),
        (true, Some(s)) if s.running_job.is_some() => Restart::Later(format!(
            "it is running {}",
            s.running_job.as_deref().unwrap_or("a job")
        )),
        (true, _) => Restart::Now,
    }
}

/// `kute service install`.
pub fn install(force: bool, print: bool) -> Result<()> {
    let argv0 = std::env::args_os().next();
    let spec = UnitSpec::current(argv0.as_deref())?;
    let unit = spec.render();
    if print {
        print!("{unit}");
        return Ok(());
    }
    let path = unit_path()?;
    let existing = match std::fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let write = plan_write(existing.as_deref(), &unit, force)?;
    if write == Write::Replace {
        let backup = path.with_extension("service.bak");
        std::fs::rename(&path, &backup)
            .with_context(|| format!("moving {} aside", path.display()))?;
        println!("kept the old unit as {}", backup.display());
    }
    if write != Write::Same {
        let dir = path.parent().expect("the unit path has a folder");
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        std::fs::write(&path, &unit).with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    println!("daemon: {}", spec.exe.display());

    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", UNIT])?;
    let status = ipc::socket_path()
        .ok()
        .and_then(|socket| ipc::request(&socket, &Request::Status).ok().flatten())
        .and_then(|response| match response {
            Response::Status { status } => Some(status),
            _ => None,
        });
    match plan_restart(is_active()?, write != Write::Same, status.as_ref()) {
        Restart::Start => {
            systemctl(&["start", UNIT])?;
            println!("started; follow its log with `journalctl --user -u kute -f`");
        }
        Restart::Keep => println!("already running"),
        Restart::Now => {
            systemctl(&["restart", UNIT])?;
            println!("restarted with the new unit");
        }
        Restart::Later(why) => println!(
            "not restarted, since {why}; run `systemctl --user restart kute` when it is done"
        ),
        Restart::Foreign => println!(
            "a kute daemon is running outside systemd; stop it, then run \
             `systemctl --user start kute`"
        ),
    }
    Ok(())
}

/// `kute service uninstall`: stops and disables the service and removes the unit kute wrote.
/// A unit written by hand is only disabled.
pub fn uninstall() -> Result<()> {
    let path = unit_path()?;
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("no {} to remove", path.display());
            return Ok(());
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    systemctl(&["disable", "--now", UNIT])?;
    if text.starts_with(MARKER) {
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        systemctl(&["daemon-reload"])?;
        println!("stopped kute and removed {}", path.display());
    } else {
        println!(
            "stopped and disabled kute; {} wasn't written by kute, so it stays",
            path.display()
        );
    }
    Ok(())
}

fn systemctl(args: &[&str]) -> Result<()> {
    let status = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()
        .context("running systemctl (is systemd running?)")?;
    if !status.success() {
        bail!("`systemctl --user {}` failed ({status})", args.join(" "));
    }
    Ok(())
}

fn is_active() -> Result<bool> {
    let status = Command::new("systemctl")
        .args(["--user", "--quiet", "is-active", UNIT])
        .status()
        .context("running systemctl (is systemd running?)")?;
    Ok(status.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(phase: Phase, running_job: Option<&str>) -> Status {
        Status {
            phase,
            paused: false,
            session: Some("2026-10-08T090312".into()),
            app: None,
            queued: 0,
            retrying: 0,
            running_job: running_job.map(Into::into),
        }
    }

    #[test]
    fn the_unit_runs_the_given_binary_with_the_given_path() {
        let unit = UnitSpec {
            exe: "/nix/store/abc-kute-0.1.0/bin/kute".into(),
            path: "/home/alex/.local/bin:/usr/bin".into(),
            env_file: env_file(
                Path::new("/home/alex/.config/kute/env"),
                Path::new("/home/alex"),
            ),
        }
        .render();
        assert!(unit.starts_with(MARKER), "{unit}");
        assert!(unit.contains("\nExecStart=/nix/store/abc-kute-0.1.0/bin/kute daemon\n"));
        assert!(unit.contains("\nEnvironment=PATH=/home/alex/.local/bin:/usr/bin\n"));
        assert!(unit.contains("\nEnvironmentFile=-%h/.config/kute/env\n"));
        assert!(unit.contains("\nWantedBy=default.target\n"));
    }

    #[test]
    fn odd_characters_are_quoted_for_systemd() {
        assert_eq!(quote("/usr/bin/kute"), "/usr/bin/kute");
        assert_eq!(quote("/opt/50%/kute"), "/opt/50%%/kute");
        assert_eq!(quote("/home/a b/kute"), "\"/home/a b/kute\"");
        assert_eq!(quote(r#"/x/"y"\z"#), r#""/x/\"y\"\\z""#);
    }

    #[test]
    fn the_path_puts_user_bins_first_without_duplicates() {
        let path = daemon_path(
            OsStr::new("/run/current-system/sw/bin:relative:/home/alex/.cargo/bin::/usr/bin"),
            Path::new("/home/alex"),
            Some(Path::new("/home/alex/.nix-profile/bin")),
        );
        assert_eq!(
            path,
            "/home/alex/.local/bin:/home/alex/.cargo/bin:/home/alex/.nix-profile/bin:\
             /run/current-system/sw/bin:/usr/bin"
        );
    }

    #[test]
    fn the_path_always_has_usr_bin() {
        let path = daemon_path(OsStr::new(""), Path::new("/home/alex"), None);
        assert_eq!(path, "/home/alex/.local/bin:/home/alex/.cargo/bin:/usr/bin");
    }

    #[test]
    fn a_link_to_the_running_binary_is_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        std::fs::create_dir(&bin).unwrap();
        let exe = std::env::current_exe().unwrap();
        std::os::unix::fs::symlink(&exe, bin.join("kute")).unwrap();
        let found = daemon_exe(Some(OsStr::new("kute")), bin.as_os_str()).unwrap();
        assert_eq!(found, bin.join("kute"));
    }

    #[test]
    fn a_name_that_leads_elsewhere_gives_the_running_binary() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("kute"), "#!/bin/sh\n").unwrap();
        let found = daemon_exe(Some(OsStr::new("kute")), tmp.path().as_os_str()).unwrap();
        assert_eq!(
            found,
            std::env::current_exe().unwrap().canonicalize().unwrap()
        );
    }

    #[test]
    fn a_hand_written_unit_needs_force() {
        let ours = UnitSpec {
            exe: "/usr/bin/kute".into(),
            path: "/usr/bin".into(),
            env_file: "/etc/kute/env".into(),
        }
        .render();
        let theirs = "[Unit]\nDescription=kute\n";
        assert_eq!(plan_write(None, &ours, false).unwrap(), Write::Create);
        assert_eq!(plan_write(Some(&ours), &ours, false).unwrap(), Write::Same);
        let older = ours.replace("/usr/bin/kute", "/home/alex/.cargo/bin/kute");
        assert_eq!(
            plan_write(Some(&older), &ours, false).unwrap(),
            Write::Update
        );
        let err = plan_write(Some(theirs), &ours, false).unwrap_err();
        assert!(err.to_string().contains("--force"), "{err}");
        assert_eq!(
            plan_write(Some(theirs), &ours, true).unwrap(),
            Write::Replace
        );
    }

    #[test]
    fn a_busy_daemon_is_not_restarted() {
        let idle = status(Phase::Idle, None);
        let recording = status(Phase::Recording, None);
        let working = status(Phase::Idle, Some("process 2026-10-08T090312"));
        assert_eq!(plan_restart(false, true, None), Restart::Start);
        assert_eq!(plan_restart(false, true, Some(&idle)), Restart::Foreign);
        assert_eq!(plan_restart(true, false, Some(&recording)), Restart::Keep);
        assert_eq!(plan_restart(true, true, Some(&idle)), Restart::Now);
        assert_eq!(plan_restart(true, true, None), Restart::Now);
        assert_eq!(
            plan_restart(true, true, Some(&recording)),
            Restart::Later("it may be recording a call (recording, 2026-10-08T090312)".into())
        );
        assert_eq!(
            plan_restart(true, true, Some(&status(Phase::Tentative, None))),
            Restart::Later("it may be recording a call (tentative, 2026-10-08T090312)".into())
        );
        assert_eq!(
            plan_restart(true, true, Some(&working)),
            Restart::Later("it is running process 2026-10-08T090312".into())
        );
    }
}
