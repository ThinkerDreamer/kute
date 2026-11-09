//! `kute doctor`: checks what kute needs, one line each, with what to do about a problem.
//!
//! It works without a valid config (that is one of the things it checks), and checks
//! programs on both the shell's `PATH` and the `PATH` the installed service runs with.

use crate::config::{Config, SummaryBackend, TranscribeBackend, expand_tilde};
use crate::gpu;
use crate::ipc::{self, Request, Response};
use crate::models;
use crate::program::{find, is_executable};
use crate::pw::{self, Snapshot};
use crate::service;
use crate::summarize::{anthropic, openai};
use anyhow::Result;
use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub level: Level,
    pub what: String,
    pub detail: String,
}

impl Check {
    fn new(level: Level, what: &str, detail: impl Into<String>) -> Check {
        Check {
            level,
            what: what.into(),
            detail: detail.into(),
        }
    }
    fn ok(what: &str, detail: impl Into<String>) -> Check {
        Check::new(Level::Ok, what, detail)
    }
    fn warn(what: &str, detail: impl Into<String>) -> Check {
        Check::new(Level::Warn, what, detail)
    }
    fn fail(what: &str, detail: impl Into<String>) -> Check {
        Check::new(Level::Fail, what, detail)
    }
}

impl fmt::Display for Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let level = match self.level {
            Level::Ok => "ok  ",
            Level::Warn => "WARN",
            Level::Fail => "FAIL",
        };
        write!(f, "{level}  {}: {}", self.what, self.detail)
    }
}

/// Runs every check, prints them, and fails when one failed.
pub fn run(config_path: Option<&Path>) -> Result<()> {
    let mut checks = Vec::new();
    let config = match Config::load(config_path) {
        Ok(config) => {
            let shown = config_path
                .map(Path::to_path_buf)
                .or_else(Config::default_path)
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            let detail =
                if config_path.is_some() || Config::default_path().is_some_and(|p| p.is_file()) {
                    format!("{shown} loaded")
                } else {
                    "no config file; using the defaults".into()
                };
            checks.push(Check::ok("config", detail));
            config
        }
        Err(e) => {
            checks.push(Check::fail(
                "config",
                format!("{e:#}; the checks below use the defaults"),
            ));
            Config::default()
        }
    };
    let print = |checks: &mut Vec<Check>| {
        for check in checks.drain(..) {
            println!("{check}");
        }
    };
    let mut worst = Level::Ok;
    let mut flush = |checks: &mut Vec<Check>| {
        worst = worst.max(checks.iter().map(|c| c.level).max().unwrap_or(Level::Ok));
        print(checks);
    };
    flush(&mut checks);

    checks.extend(storage(&config));
    flush(&mut checks);
    let shell_path = std::env::var_os("PATH").unwrap_or_default();
    let unit = service::unit_path()
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok());
    let service_path = unit.as_deref().and_then(unit_path_value);
    checks.extend(programs(&shell_path, service_path.as_deref()));
    flush(&mut checks);
    checks.push(pipewire(&config));
    flush(&mut checks);
    checks.push(gpu(&config));
    checks.extend(model_checks(&config, &shell_path, service_path.as_deref()));
    flush(&mut checks);
    checks.extend(summarizer(&config, &shell_path, service_path.as_deref()));
    flush(&mut checks);
    checks.extend(daemon(unit.is_some()));
    checks.push(notifications());
    checks.push(tray());
    flush(&mut checks);

    match worst {
        Level::Fail => anyhow::bail!("some checks failed"),
        Level::Warn => println!("\nkute can run; the warnings above limit what it does."),
        Level::Ok => println!("\nall good"),
    }
    Ok(())
}

fn storage(config: &Config) -> Vec<Check> {
    let dir = match config.data_dir() {
        Ok(dir) => dir,
        Err(e) => return vec![Check::fail("storage", format!("{e:#}"))],
    };
    storage_check(&dir)
}

/// Whether kute can write to `dir` (or create it, when it doesn't exist yet), and how much
/// room there is.
pub fn storage_check(dir: &Path) -> Vec<Check> {
    let shown = dir.display();
    // The folder kute would create `dir` in, when it isn't there yet.
    let Some(existing) = dir.ancestors().find(|a| a.exists()) else {
        return vec![Check::fail(
            "storage",
            format!("{shown}: no parent folder exists"),
        )];
    };
    // A name nothing else uses, created only if absent: never touches someone's file.
    let probe = existing.join(format!(".kute-doctor-{}", std::process::id()));
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
    {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
        }
        Err(e) => {
            return vec![Check::fail(
                "storage",
                format!("cannot write to {}: {e}", existing.display()),
            )];
        }
    }
    let note = if existing == dir {
        String::new()
    } else {
        "; it doesn't exist yet, kute creates it".into()
    };
    let mut checks = Vec::new();
    match free_bytes(existing) {
        // An hour of a call is about 230 MB of WAV (two 16 kHz tracks) before compression.
        Some(free) if free < 2_000_000_000 => checks.push(Check::warn(
            "storage",
            format!(
                "{shown}: only {:.1} GB free; an hour of call takes about 0.25 GB until it \
                 is compressed{note}",
                free as f64 / 1e9
            ),
        )),
        Some(free) => checks.push(Check::ok(
            "storage",
            format!("{shown} ({:.0} GB free{note})", free as f64 / 1e9),
        )),
        None => checks.push(Check::ok("storage", format!("{shown}{note}"))),
    }
    checks
}

fn free_bytes(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: a valid C string and a valid out pointer.
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

/// `PATH` from the unit's `Environment="PATH=..."` line.
pub fn unit_path_value(unit: &str) -> Option<String> {
    unit.lines().find_map(|line| {
        let value = line.trim().strip_prefix("Environment=")?;
        let value = match value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
            Some(quoted) => quoted.replace("\\\"", "\"").replace("\\\\", "\\"),
            None => value.to_owned(),
        };
        Some(value.strip_prefix("PATH=")?.replace("%%", "%"))
    })
}

/// Whether a configured command's program (a path, or a name on either `PATH`) can run.
fn command_found(program: &str, shell_path: &OsStr, service_path: Option<&str>) -> bool {
    let path = expand_tilde(Path::new(program));
    if path.components().count() > 1 {
        is_executable(&path)
    } else {
        find(program, shell_path).is_some()
            || service_path.is_some_and(|p| find(program, OsStr::new(p)).is_some())
    }
}

/// What each program is for, and whether kute can work without it.
const PROGRAMS: &[(&str, &str, Level)] = &[
    ("pw-dump", "watches PipeWire for calls", Level::Fail),
    ("pw-record", "records the two tracks", Level::Fail),
    (
        "ffmpeg",
        "compresses recordings and reads imports",
        Level::Fail,
    ),
    ("ffprobe", "reads the length of imports", Level::Fail),
    ("xdg-open", "opens summaries and folders", Level::Warn),
];

/// Each program on the shell's `PATH` and, when the service is installed, on its `PATH`:
/// the daemon runs with the second.
pub fn programs(shell_path: &OsStr, service_path: Option<&str>) -> Vec<Check> {
    PROGRAMS
        .iter()
        .map(|&(name, purpose, missing)| {
            let in_shell = find(name, shell_path);
            let in_service = service_path.map(|p| find(name, OsStr::new(p)));
            match (in_shell, in_service) {
                (Some(path), None | Some(Some(_))) => {
                    Check::ok(name, format!("{} ({purpose})", path.display()))
                }
                (Some(path), Some(None)) => Check::new(
                    missing,
                    name,
                    format!(
                        "{} is not on the service's PATH; run `kute service install` again",
                        path.display()
                    ),
                ),
                (None, Some(Some(path))) => Check::ok(
                    name,
                    format!("{} (on the service's PATH; {purpose})", path.display()),
                ),
                (None, _) => Check::new(
                    missing,
                    name,
                    format!("not found; kute needs it because it {purpose}"),
                ),
            }
        })
        .collect()
}

fn pipewire(config: &Config) -> Check {
    let output = Command::new("pw-dump")
        .arg("--no-colors")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output();
    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            return Check::fail(
                "PipeWire",
                format!("pw-dump failed ({}); is PipeWire running?", o.status),
            );
        }
        Err(e) => return Check::fail("PipeWire", format!("cannot run pw-dump: {e}")),
    };
    let mut snapshot = None;
    if let Err(e) = pw::read_updates(output.stdout.as_slice(), |s| snapshot = Some(s)) {
        return Check::fail("PipeWire", format!("cannot read pw-dump's output: {e:#}"));
    }
    pipewire_check(&snapshot.unwrap_or_default(), config.capture.mic.as_deref())
}

/// What a graph snapshot says about recording: a microphone to record, an output to
/// record from, and `capture.mic` finding its device.
pub fn pipewire_check(snap: &Snapshot, mic: Option<&str>) -> Check {
    let describe = |name: &str| {
        snap.sources
            .iter()
            .find(|s| s.name == name)
            .and_then(|s| s.description.clone())
            .unwrap_or_else(|| name.to_owned())
    };
    if snap.default_sink.is_none() {
        return Check::fail(
            "PipeWire",
            "no default output; the others' side can't be recorded",
        );
    }
    // A matching capture.mic is recorded whatever the default is.
    if let Some((mic, found)) = mic.and_then(|m| Some((m, snap.find_source(m)?))) {
        return Check::ok(
            "PipeWire",
            format!("capture.mic {mic:?} is {}", describe(&found)),
        );
    }
    let Some(source) = &snap.default_source else {
        return Check::fail(
            "PipeWire",
            "no default microphone; pick one in the sound settings",
        );
    };
    if let Some(mic) = mic {
        return Check::warn(
            "PipeWire",
            format!(
                "capture.mic {mic:?} matches no microphone, so the default ({}) is used; \
                 plugged in?",
                describe(source)
            ),
        );
    }
    Check::ok(
        "PipeWire",
        format!("default microphone: {}", describe(source)),
    )
}

/// The compiled-in Whisper backend, and whether the system has what it needs.
fn gpu(config: &Config) -> Check {
    if config.transcribe.backend == TranscribeBackend::Command {
        return Check::ok(
            "GPU",
            "transcribe.command transcribes and picks its own device; Whisper's GPU setting \
             doesn't apply",
        );
    }
    let backend = if cfg!(feature = "cuda") {
        "CUDA"
    } else if cfg!(feature = "vulkan") {
        "Vulkan"
    } else {
        return Check::warn(
            "GPU",
            "built without GPU support; transcription runs on the CPU (slow for large models)",
        );
    };
    if config.transcribe.gpu == Some(false) {
        return Check::warn(
            "GPU",
            format!("built with {backend}, but transcribe.gpu = false: transcription uses the CPU"),
        );
    }
    let device = gpu::device();
    if let Some(note) = gpu::integrated_note(config.transcribe.gpu, device.as_ref()) {
        return Check::ok("GPU", format!("{backend}, but {note}"));
    }
    let found = if cfg!(feature = "cuda") {
        Path::new("/dev/nvidiactl").exists()
    } else {
        !vulkan_drivers().is_empty()
    };
    if let Some(device) = device {
        Check::ok("GPU", format!("{backend} on {}", device.name))
    } else if found {
        Check::ok(
            "GPU",
            format!("{backend}; whisper.cpp falls back to the CPU if the GPU can't be used"),
        )
    } else if cfg!(feature = "cuda") {
        Check::warn(
            "GPU",
            "built with CUDA, but no NVIDIA driver is loaded: transcription will use the CPU",
        )
    } else {
        Check::warn(
            "GPU",
            "built with Vulkan, but no Vulkan driver is installed (no ICD file in \
             vulkan/icd.d): transcription will use the CPU",
        )
    }
}

/// Vulkan driver manifests in the loader's standard places.
fn vulkan_drivers() -> Vec<PathBuf> {
    let data_dirs = std::env::var("XDG_DATA_DIRS")
        .ok()
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| "/usr/local/share:/usr/share".into());
    let mut dirs: Vec<PathBuf> = vec!["/etc/vulkan/icd.d".into()];
    dirs.extend(
        data_dirs
            .split(':')
            .map(|d| Path::new(d).join("vulkan/icd.d")),
    );
    // NixOS puts drivers here.
    dirs.push("/run/opengl-driver/share/vulkan/icd.d".into());
    dirs.iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect()
}

fn model_checks(config: &Config, shell_path: &OsStr, service_path: Option<&str>) -> Vec<Check> {
    let Ok(dir) = config.data_dir().map(|d| d.join("models")) else {
        return Vec::new();
    };
    let mut checks = model_checks_in(&dir, &config.transcribe.model);
    if config.transcribe.backend == TranscribeBackend::Command {
        // The helper does the transcribing; only the VAD model is kute's.
        checks[0] = transcribe_command(&config.transcribe.command, shell_path, service_path);
    }
    checks
}

fn transcribe_command(argv: &[String], shell_path: &OsStr, service_path: Option<&str>) -> Check {
    let what = "transcriber";
    let Some(program) = argv.first() else {
        return Check::fail(
            what,
            "transcribe.backend = \"command\" needs transcribe.command",
        );
    };
    if command_found(program, shell_path, service_path) {
        Check::ok(what, format!("command {}", argv.join(" ")))
    } else {
        Check::fail(
            what,
            format!("transcribe.command {program:?} is not found or not executable"),
        )
    }
}

pub fn model_checks_in(models_dir: &Path, model: &str) -> Vec<Check> {
    let whisper = match models::resolve_whisper(models_dir, model) {
        Ok((path, _)) => Check::ok("Whisper model", format!("{model} ({})", path.display())),
        Err(e) => Check::fail("Whisper model", format!("{e:#}")),
    };
    let vad_path = models::path_for(models_dir, &models::VAD_MODEL);
    let vad = if vad_path.is_file() {
        Check::ok("VAD model", vad_path.display().to_string())
    } else {
        Check::fail(
            "VAD model",
            "not downloaded; transcription needs it; run `kute model download`",
        )
    };
    vec![whisper, vad]
}

fn summarizer(config: &Config, shell_path: &OsStr, service_path: Option<&str>) -> Vec<Check> {
    let cfg = &config.summary;
    let what = "summaries";
    match cfg.backend {
        SummaryBackend::None => vec![Check::ok(
            what,
            "off (summary.backend = \"none\"); transcripts only",
        )],
        SummaryBackend::Command => {
            let Some(program) = cfg.command.first() else {
                return vec![Check::fail(
                    what,
                    "summary.backend = \"command\" needs summary.command",
                )];
            };
            if command_found(program, shell_path, service_path) {
                vec![Check::ok(what, format!("command {program:?}"))]
            } else {
                vec![Check::fail(
                    what,
                    format!("summary.command {program:?} is not found on PATH"),
                )]
            }
        }
        SummaryBackend::Openai | SummaryBackend::Anthropic => {
            let anthropic = cfg.backend == SummaryBackend::Anthropic;
            let (name, default_url) = if anthropic {
                ("anthropic", anthropic::DEFAULT_BASE_URL)
            } else {
                ("openai", openai::DEFAULT_BASE_URL)
            };
            let url = cfg.base_url.clone().unwrap_or_else(|| default_url.into());
            let key_env = cfg
                .api_key_env
                .clone()
                .or_else(|| anthropic.then(|| "ANTHROPIC_API_KEY".into()));
            let mut checks = Vec::new();
            if !anthropic && cfg.model.is_none() {
                checks.push(Check::fail(
                    what,
                    "summary.backend = \"openai\" needs summary.model",
                ));
            }
            if let Some(env) = &key_env {
                checks.push(api_key_check(env, &env_file_path()));
            }
            checks.push(reachable(name, &url));
            checks
        }
    }
}

/// `~/.config/kute/env`, which the service reads API keys from.
fn env_file_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_default()
        .join("kute")
        .join("env")
}

/// The daemon gets its key from the service's env file; a command run in a shell gets it
/// from the shell.
pub fn api_key_check(env: &str, env_file: &Path) -> Check {
    let in_shell = std::env::var(env).is_ok_and(|v| !v.is_empty());
    let in_file = std::fs::read_to_string(env_file).is_ok_and(|text| {
        text.lines().any(|l| {
            let l = l.trim().strip_prefix("export ").unwrap_or(l.trim());
            l.strip_prefix(env)
                .and_then(|rest| rest.strip_prefix('='))
                .is_some_and(|v| !v.trim().trim_matches('"').is_empty())
        })
    });
    let file = env_file.display();
    match (in_file, in_shell) {
        (true, _) => Check::ok("API key", format!("{env} is set in {file}")),
        (false, true) => Check::warn(
            "API key",
            format!(
                "{env} is set in this shell, but the service won't see it: put \
                 {env}=... in {file} (chmod 600)"
            ),
        ),
        (false, false) => Check::fail(
            "API key",
            format!("{env} is not set; put {env}=... in {file} (chmod 600)"),
        ),
    }
}

/// Any HTTP answer counts: the point is that the server is there, not that a GET on its
/// root works.
fn reachable(name: &str, url: &str) -> Check {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    match agent.get(url).call() {
        Ok(_) => Check::ok("summarizer", format!("{name} at {url} answers")),
        Err(e) => Check::fail("summarizer", format!("{url} can't be reached: {e}")),
    }
}

fn daemon(unit_installed: bool) -> Vec<Check> {
    let mut checks = Vec::new();
    let status = ipc::socket_path()
        .ok()
        .and_then(|p| ipc::request(&p, &Request::Status).ok().flatten());
    checks.push(match status {
        Some(Response::Status { status }) => Check::ok(
            "daemon",
            format!(
                "running ({}{})",
                format!("{:?}", status.phase).to_lowercase(),
                if status.paused { ", paused" } else { "" }
            ),
        ),
        Some(_) => Check::warn("daemon", "answers, but not with its status"),
        None => Check::warn(
            "daemon",
            "not running, so nothing is recorded; `kute service install` starts it at login",
        ),
    });
    if unit_installed {
        let active = Command::new("systemctl")
            .args(["--user", "--quiet", "is-active", service::UNIT])
            .status()
            .is_ok_and(|s| s.success());
        checks.push(if active {
            Check::ok("service", format!("{} is active", service::UNIT))
        } else {
            Check::warn(
                "service",
                format!(
                    "{} is installed but not active; see `journalctl --user -u kute`",
                    service::UNIT
                ),
            )
        });
    } else {
        checks.push(Check::warn(
            "service",
            "not installed; `kute service install` runs the daemon at login",
        ));
    }
    checks
}

fn notifications() -> Check {
    match notify_rust::get_server_information() {
        Ok(info) => Check::ok("notifications", format!("{} {}", info.name, info.version)),
        Err(e) => Check::warn(
            "notifications",
            format!("no notification server ({e}); you won't see when kute records"),
        ),
    }
}

/// Whether a tray host is there to show kute's icon.
fn tray() -> Check {
    let output = Command::new("busctl")
        .args([
            "--user",
            "call",
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "NameHasOwner",
            "s",
            "org.kde.StatusNotifierWatcher",
        ])
        .stderr(Stdio::null())
        .output();
    match output {
        Ok(o) if o.status.success() && String::from_utf8_lossy(&o.stdout).contains("true") => {
            Check::ok("tray", "a StatusNotifierItem host is running")
        }
        Ok(o) if o.status.success() => Check::warn(
            "tray",
            "no tray host, so kute's icon won't show (on GNOME, install the AppIndicator \
             extension)",
        ),
        _ => Check::warn("tray", "couldn't ask D-Bus (busctl failed)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pw::Source;
    use std::os::unix::fs::PermissionsExt;

    fn executable(dir: &Path, name: &str) {
        let path = dir.join(name);
        std::fs::write(&path, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn level_of(checks: &[Check], what: &str) -> Level {
        checks.iter().find(|c| c.what == what).unwrap().level
    }

    #[test]
    fn the_gpu_check_leaves_a_command_backend_alone() {
        let config = Config::parse(
            "[transcribe]\nbackend = \"command\"\ncommand = [\"helper\"]\ngpu = false\n",
        )
        .unwrap();
        let check = gpu(&config);
        assert_eq!(check.level, Level::Ok);
        assert!(
            check.detail.contains("transcribe.command"),
            "{}",
            check.detail
        );
    }

    #[test]
    fn programs_missing_from_the_service_path_are_reported() {
        let shell = tempfile::tempdir().unwrap();
        let service = tempfile::tempdir().unwrap();
        for p in ["pw-dump", "pw-record", "ffmpeg", "ffprobe", "xdg-open"] {
            executable(shell.path(), p);
        }
        executable(service.path(), "pw-dump");
        let service_path = service.path().display().to_string();
        let checks = programs(shell.path().as_os_str(), Some(&service_path));
        assert_eq!(level_of(&checks, "pw-dump"), Level::Ok);
        assert_eq!(level_of(&checks, "pw-record"), Level::Fail);
        assert_eq!(level_of(&checks, "xdg-open"), Level::Warn);
        assert!(
            checks[1].detail.contains("service's PATH"),
            "{}",
            checks[1].detail
        );
    }

    #[test]
    fn a_file_that_cannot_be_run_is_not_found() {
        let shell = tempfile::tempdir().unwrap();
        std::fs::write(shell.path().join("ffmpeg"), "").unwrap();
        let checks = programs(shell.path().as_os_str(), None);
        assert_eq!(level_of(&checks, "ffmpeg"), Level::Fail);
    }

    #[test]
    fn storage_is_checked_where_it_would_be_created() {
        let tmp = tempfile::tempdir().unwrap();
        let checks = storage_check(&tmp.path().join("a/b"));
        assert_eq!(checks[0].level, Level::Ok, "{}", checks[0].detail);
        assert!(checks[0].detail.contains("kute creates it"));
        // The probe leaves nothing behind, and never touches a file already there.
        std::fs::write(tmp.path().join(".kute-doctor"), "mine").unwrap();
        storage_check(tmp.path());
        let names: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, [".kute-doctor"]);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join(".kute-doctor")).unwrap(),
            "mine"
        );
    }

    #[test]
    fn programs_without_a_service_only_need_the_shell() {
        let shell = tempfile::tempdir().unwrap();
        executable(shell.path(), "ffmpeg");
        let checks = programs(shell.path().as_os_str(), None);
        assert_eq!(level_of(&checks, "ffmpeg"), Level::Ok);
        assert_eq!(level_of(&checks, "ffprobe"), Level::Fail);
    }

    #[test]
    fn the_unit_path_line_is_read_back() {
        let unit = "[Service]\nEnvironment=\"PATH=/a b/bin:/usr/bin:/x%%y\"\n";
        assert_eq!(
            unit_path_value(unit).as_deref(),
            Some("/a b/bin:/usr/bin:/x%y")
        );
        assert_eq!(
            unit_path_value("Environment=PATH=/usr/bin\n").as_deref(),
            Some("/usr/bin")
        );
        assert_eq!(unit_path_value("Environment=FOO=1\n"), None);
    }

    #[test]
    fn the_unit_written_by_install_gives_its_path() {
        let spec = service::UnitSpec {
            exe: "/home/alex/.cargo/bin/kute".into(),
            path: "/home/alex/.local/bin:/usr/bin".into(),
            env_file: "%h/.config/kute/env".into(),
        };
        assert_eq!(
            unit_path_value(&spec.render()).as_deref(),
            Some("/home/alex/.local/bin:/usr/bin")
        );
    }

    fn snapshot() -> Snapshot {
        Snapshot {
            streams: Vec::new(),
            default_source: Some("alsa_input.usb-headset".into()),
            default_sink: Some("alsa_output.hdmi".into()),
            sources: vec![
                Source {
                    name: "alsa_input.usb-headset".into(),
                    description: Some("USB Headset".into()),
                },
                Source {
                    name: "alsa_input.pci".into(),
                    description: Some("Built-in Audio".into()),
                },
            ],
        }
    }

    #[test]
    fn pipewire_names_the_microphone_in_use() {
        let check = pipewire_check(&snapshot(), None);
        assert_eq!(check.level, Level::Ok);
        assert!(check.detail.contains("USB Headset"));
        let check = pipewire_check(&snapshot(), Some("built-in"));
        assert_eq!(check.level, Level::Ok);
        assert!(check.detail.contains("Built-in Audio"));
    }

    #[test]
    fn pipewire_warns_when_capture_mic_matches_nothing() {
        let check = pipewire_check(&snapshot(), Some("webcam"));
        assert_eq!(check.level, Level::Warn);
        assert!(check.detail.contains("USB Headset"), "{}", check.detail);
    }

    #[test]
    fn pipewire_fails_without_default_devices() {
        let mut snap = snapshot();
        snap.default_sink = None;
        assert_eq!(pipewire_check(&snap, None).level, Level::Fail);
        snap.default_sink = Some("alsa_output.hdmi".into());
        snap.default_source = None;
        assert_eq!(pipewire_check(&snap, None).level, Level::Fail);
        // A configured mic that is there doesn't need a default.
        assert_eq!(pipewire_check(&snap, Some("headset")).level, Level::Ok);
    }

    #[test]
    fn models_report_what_to_download() {
        let dir = tempfile::tempdir().unwrap();
        let checks = model_checks_in(dir.path(), "tiny");
        assert_eq!(checks[0].level, Level::Fail);
        assert!(checks[0].detail.contains("kute model download"));
        assert_eq!(checks[1].level, Level::Fail);
    }

    #[test]
    fn a_command_transcriber_is_checked_instead_of_the_whisper_model() {
        let path = OsStr::new("/usr/bin:/bin");
        let ok = transcribe_command(&["sh".into(), "helper.py".into()], path, None);
        assert_eq!(ok.level, Level::Ok);
        let missing = transcribe_command(&["/nonexistent/python".into()], path, None);
        assert_eq!(missing.level, Level::Fail);
        assert_eq!(transcribe_command(&[], path, None).level, Level::Fail);
    }

    #[test]
    fn an_api_key_in_the_env_file_is_enough() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("env");
        let env = "KUTE_DOCTOR_TEST_KEY_THAT_IS_NOT_SET";
        assert_eq!(api_key_check(env, &file).level, Level::Fail);
        std::fs::write(&file, format!("# keys\nexport {env}=\"sk-123\"\n")).unwrap();
        assert_eq!(api_key_check(env, &file).level, Level::Ok);
        std::fs::write(&file, format!("{env}=\n")).unwrap();
        assert_eq!(api_key_check(env, &file).level, Level::Fail);
    }

    #[test]
    fn lines_read_well() {
        let check = Check::warn("VAD model", "not downloaded");
        assert_eq!(check.to_string(), "WARN  VAD model: not downloaded");
    }
}
