//! The `kute` command line.

use crate::config::{Config, SummaryBackend};
use crate::daemon;
use crate::detect::{Action, Detector, replay};
use crate::doctor;
use crate::index::HitKind;
use crate::ipc::{self, Request, Response, Status};
use crate::jobs::{self, ConversationsFailed, Job};
use crate::models;
use crate::pipeline;
use crate::pw::trace;
use crate::render;
use crate::service;
use crate::store::{LanguageChange, SessionState, Store, Track};
use crate::time;
use crate::transcribe::parse_languages;
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "kute",
    version,
    about = "Records, transcribes and summarizes your calls, locally"
)]
struct Cli {
    /// Config file (default: ~/.config/kute/config.toml)
    #[arg(long, global = true, value_name = "FILE")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the daemon: detect, record and process calls (normally started by systemd)
    Daemon,
    /// Show what the daemon is doing
    Status,
    /// Start recording now; it goes on until `kute stop`
    Record { title: Vec<String> },
    /// Stop the current recording and process it
    Stop,
    /// Stop the current recording and delete it
    Discard,
    /// Stop detecting calls until `kute resume` (ends a detected recording)
    Pause,
    /// Detect calls again
    Resume,
    /// Import two recordings (your mic, everyone else) and process them
    Import {
        me: PathBuf,
        others: PathBuf,
        /// Start time, e.g. 2026-10-08T15:00:00 (default: <prefix>.start, then file birth time)
        #[arg(long)]
        start: Option<String>,
        #[arg(long)]
        title: Option<String>,
        #[command(flatten)]
        lang: LangArgs,
        /// Only import; run `kute process` later
        #[arg(long)]
        no_process: bool,
    },
    /// Finish any outstanding transcription, summaries and compression for a session
    Process { id: String },
    /// List sessions
    List {
        /// Only sessions newer than this, e.g. 7d or 12h
        #[arg(long)]
        since: Option<String>,
    },
    /// Print a session's transcript (or summary)
    Show {
        id: String,
        #[arg(long)]
        summary: bool,
    },
    /// Search all transcripts and summaries
    Search {
        #[arg(required = true)]
        query: Vec<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Set a session's title
    Title {
        id: String,
        #[arg(required = true)]
        title: Vec<String>,
    },
    /// Set notes the summarizer reads, e.g. who the other speakers are (no text: print them)
    Notes {
        id: String,
        text: Vec<String>,
        /// Remove the notes
        #[arg(long, conflicts_with = "text")]
        clear: bool,
    },
    /// Set the languages spoken in a session, e.g. ru, ru en to pick between them, or auto
    /// (nothing: print them)
    Language {
        id: String,
        /// Both tracks; replaces --me and --others set earlier
        codes: Vec<String>,
        /// Only your microphone's track, e.g. ru
        #[arg(long, value_name = "CODES")]
        me: Option<String>,
        /// Only the others' track, e.g. ru,en
        #[arg(long, value_name = "CODES")]
        others: Option<String>,
        /// Use `transcribe.language` from the config again
        #[arg(long, conflicts_with_all = ["codes", "me", "others"])]
        clear: bool,
    },
    /// Open a session's folder
    Open { id: String },
    /// Summarize again (all conversations, or one)
    Summarize {
        id: String,
        #[arg(long)]
        conversation: Option<u32>,
        /// Summary language for this session, e.g. it (saved in meta.toml)
        #[arg(long)]
        lang: Option<String>,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Transcribe again from the stored audio
    Retranscribe {
        id: String,
        #[arg(long)]
        model: Option<String>,
        #[command(flatten)]
        lang: LangArgs,
    },
    /// Rebuild the search index from the session folders
    Reindex,
    /// Manage Whisper and VAD models
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Run the daemon at login, as a systemd user service
    Service {
        #[command(subcommand)]
        command: ServiceCommand,
    },
    /// Check that everything kute needs is installed and set up
    Doctor,
    /// Tools for bug reports
    Debug {
        #[command(subcommand)]
        command: DebugCommand,
    },
}

#[derive(Subcommand)]
enum DebugCommand {
    /// Record PipeWire graph changes to a file, without serials, user or host names, until
    /// Ctrl-C
    Trace {
        /// Where to write it (default: kute-trace-<time>.jsonl here; - for stdout)
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
        /// Stop after this long, e.g. 2m
        #[arg(long)]
        duration: Option<String>,
    },
    /// Show what the call detector does with a recorded trace
    Replay { trace: PathBuf },
}

#[derive(Subcommand)]
enum ServiceCommand {
    /// Write the systemd user unit for this kute binary, enable it and start it
    Install {
        /// Replace a kute.service written by hand (kept as kute.service.bak)
        #[arg(long)]
        force: bool,
        /// Print the unit instead of installing it
        #[arg(long)]
        print: bool,
    },
    /// Stop and disable the service, and remove the unit kute wrote
    Uninstall,
}

/// Languages for `retranscribe` and `import`, saved in meta.toml (see `kute language`).
#[derive(Args)]
struct LangArgs {
    /// Languages spoken, e.g. ru or ru,en; auto to detect any
    #[arg(long, value_name = "CODES")]
    lang: Option<String>,
    /// Languages on your microphone's track only
    #[arg(long, value_name = "CODES")]
    me_lang: Option<String>,
    /// Languages on the others' track only
    #[arg(long, value_name = "CODES")]
    others_lang: Option<String>,
}

impl LangArgs {
    fn change(&self) -> Result<LanguageChange> {
        language_change(
            self.lang.as_deref(),
            self.me_lang.as_deref(),
            self.others_lang.as_deref(),
        )
    }
}

fn language_change(
    all: Option<&str>,
    me: Option<&str>,
    others: Option<&str>,
) -> Result<LanguageChange> {
    let parse = |codes: Option<&str>| codes.map(parse_languages).transpose();
    Ok(LanguageChange {
        all: parse(all)?,
        me: parse(me)?,
        others: parse(others)?,
    })
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Download a Whisper model (default: the configured one) and the VAD model
    Download { name: Option<String> },
    /// List known models
    List,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    Openai,
    Anthropic,
    Command,
}

impl From<BackendArg> for SummaryBackend {
    fn from(b: BackendArg) -> Self {
        match b {
            BackendArg::Openai => SummaryBackend::Openai,
            BackendArg::Anthropic => SummaryBackend::Anthropic,
            BackendArg::Command => SummaryBackend::Command,
        }
    }
}

pub fn run() -> Result<()> {
    whisper_rs::install_logging_hooks();
    let cli = Cli::parse();
    // The service needs neither the config nor the store, and must work when they don't.
    if let Command::Service { command } = &cli.command {
        return match command {
            ServiceCommand::Install { force, print } => service::install(*force, *print),
            ServiceCommand::Uninstall => service::uninstall(),
        };
    }
    if let Command::Debug {
        command: DebugCommand::Trace { out, duration },
    } = &cli.command
    {
        return debug_trace(out.as_deref(), duration.as_deref());
    }
    if let Command::Doctor = &cli.command {
        return doctor::run(cli.config.as_deref());
    }
    let config = Config::load(cli.config.as_deref())?;
    let store = Store::open(config.data_dir()?)?;

    match cli.command {
        Command::Daemon => daemon::run(store, config)?,
        Command::Status => control(Request::Status)?,
        Command::Record { title } => control(Request::Record {
            title: (!title.is_empty()).then(|| title.join(" ")),
        })?,
        Command::Stop => control(Request::Stop)?,
        Command::Discard => control(Request::Discard)?,
        Command::Pause => control(Request::Pause)?,
        Command::Resume => control(Request::Resume)?,
        Command::Import {
            me,
            others,
            start,
            title,
            lang,
            no_process,
        } => run_job(
            &store,
            &config,
            &Job::Import {
                // Absolute, so the job means the same files wherever it runs.
                me: std::path::absolute(&me)?,
                others: std::path::absolute(&others)?,
                start,
                title,
                languages: lang.change()?,
                process: !no_process,
            },
        )?,
        Command::Process { id } => {
            let session = store.resolve(&id)?;
            let job = Job::Process {
                id: session.id.clone(),
            };
            if !queue_on_daemon(&job)? {
                // The daemon's worker processes a recording session's closed conversations;
                // a second process doing the same at the same time would race it.
                if session.load_meta()?.state == SessionState::Recording {
                    bail!("session {} is still recording", session.id);
                }
                run_foreground(&store, &config, &job)?;
            }
        }
        Command::List { since } => list(&store, since.as_deref())?,
        Command::Show { id, summary } => {
            let session = store.resolve(&id)?;
            let (path, what) = if summary {
                (session.summary_path(), "summary")
            } else {
                (session.transcript_path(), "transcript")
            };
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("{} has no {what} yet", session.id))?;
            print!("{text}");
        }
        Command::Search { query, limit } => search(&store, &config, &query.join(" "), limit)?,
        Command::Title { id, title } => run_job(
            &store,
            &config,
            &Job::Title {
                id: store.resolve(&id)?.id,
                title: title.join(" "),
            },
        )?,
        Command::Notes { id, text, clear } => {
            let session = store.resolve(&id)?;
            if text.is_empty() && !clear {
                if let Some(notes) = session.load_meta()?.notes {
                    println!("{}", notes.trim_end());
                }
            } else {
                run_job(
                    &store,
                    &config,
                    &Job::Notes {
                        id: session.id,
                        notes: (!clear).then(|| text.join(" ")),
                    },
                )?;
            }
        }
        Command::Language {
            id,
            codes,
            me,
            others,
            clear,
        } => {
            let session = store.resolve(&id)?;
            let languages = if clear {
                LanguageChange {
                    all: Some(Vec::new()),
                    ..LanguageChange::default()
                }
            } else {
                let all = (!codes.is_empty()).then(|| codes.join(","));
                language_change(all.as_deref(), me.as_deref(), others.as_deref())?
            };
            if languages.is_empty() {
                let transcribe = session.load_meta()?.transcribe;
                for track in Track::BOTH {
                    match transcribe.for_track(track) {
                        [] => println!(
                            "{}: {} (from the config)",
                            track.as_str(),
                            config.transcribe.language
                        ),
                        codes => println!("{}: {}", track.as_str(), codes.join(",")),
                    }
                }
            } else {
                run_job(
                    &store,
                    &config,
                    &Job::Language {
                        id: session.id,
                        languages,
                    },
                )?;
            }
        }
        Command::Open { id } => {
            let session = store.resolve(&id)?;
            std::process::Command::new("xdg-open")
                .arg(&session.dir)
                .spawn()
                .context("running xdg-open")?;
        }
        Command::Summarize {
            id,
            conversation,
            lang,
            backend,
        } => run_job(
            &store,
            &config,
            &Job::Summarize {
                id: store.resolve(&id)?.id,
                conversation,
                lang,
                backend: backend.map(Into::into),
            },
        )?,
        Command::Retranscribe { id, model, lang } => run_job(
            &store,
            &config,
            &Job::Retranscribe {
                id: store.resolve(&id)?.id,
                model,
                languages: lang.change()?,
            },
        )?,
        Command::Reindex => run_job(&store, &config, &Job::Reindex)?,
        Command::Service { .. }
        | Command::Doctor
        | Command::Debug {
            command: DebugCommand::Trace { .. },
        } => unreachable!("handled before the config is loaded"),
        Command::Debug {
            command: DebugCommand::Replay { trace },
        } => debug_replay(&config, &trace)?,
        Command::Model {
            command: ModelCommand::List,
        } => model_list(&store, &config),
        Command::Model {
            command: ModelCommand::Download { name },
        } => {
            let name = name.unwrap_or_else(|| config.transcribe.model.clone());
            let info = models::find_whisper(&name)
                .with_context(|| format!("unknown Whisper model {name:?}"))?;
            for info in [info, &models::VAD_MODEL] {
                let mut progress = Progress::new(info.name, std::io::stderr().is_terminal());
                let result = models::download(&store.models_dir(), info, &mut |done, total| {
                    if let Some(text) = progress.update(done, total) {
                        eprint!("{text}");
                        let _ = std::io::stderr().flush();
                    }
                });
                if progress.line_open() {
                    eprintln!();
                }
                println!("{}", result?.display());
            }
        }
    }
    Ok(())
}

/// Download progress for one file. On a terminal it redraws one line, and only when the shown
/// value changes; otherwise (piped, logged) it prints a line every 10%.
struct Progress {
    name: String,
    tty: bool,
    last: String,
}

impl Progress {
    fn new(name: &str, tty: bool) -> Self {
        Self {
            name: name.to_string(),
            tty,
            last: String::new(),
        }
    }

    /// The text to print for this update, or `None` when nothing visible changed.
    fn update(&mut self, done: u64, total: u64) -> Option<String> {
        let mb = |bytes: u64| bytes as f64 / 1_000_000.0;
        let finished = done >= total;
        let text = if self.tty {
            let end = if finished { "\n" } else { "" };
            format!(
                "\r{}: {:.1} / {:.1} MB{end}",
                self.name,
                mb(done),
                mb(total)
            )
        } else {
            let tenth = (done * 10).checked_div(total).unwrap_or(10) * 10;
            if tenth == 0 {
                return None;
            }
            let shown = total * tenth / 100;
            format!(
                "{}: {:.1} / {:.1} MB ({tenth}%)\n",
                self.name,
                mb(shown),
                mb(total)
            )
        };
        if text == self.last {
            return None;
        }
        self.last = text.clone();
        Some(text)
    }

    /// Whether a terminal line was left without its newline (the download stopped early).
    fn line_open(&self) -> bool {
        self.tty && !self.last.is_empty() && !self.last.ends_with('\n')
    }
}

fn debug_trace(out: Option<&std::path::Path>, duration: Option<&str>) -> Result<()> {
    let duration = duration
        .map(|d| {
            let span: jiff::Span = d
                .parse()
                .with_context(|| format!("cannot read {d:?} as a duration (try 2m)"))?;
            let d = jiff::SignedDuration::try_from(span)?;
            Ok::<_, anyhow::Error>(d.unsigned_abs())
        })
        .transpose()?;
    if out == Some(std::path::Path::new("-")) {
        return trace::record(&mut std::io::stdout().lock(), duration);
    }
    let path = match out {
        Some(path) => path.to_path_buf(),
        None => PathBuf::from(format!(
            "kute-trace-{}.jsonl",
            time::now().strftime("%Y%m%d-%H%M%S")
        )),
    };
    let mut file =
        std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    eprintln!(
        "kute: recording PipeWire changes to {} (Ctrl-C to stop)",
        path.display()
    );
    eprintln!(
        "kute: serials, MAC addresses, your user and host names and your home folder are \
         replaced; app and device names are kept"
    );
    trace::record(&mut file, duration)
}

/// Replays a trace through the detector with this config's `[detect]` settings and prints
/// each decision, so a bug report's trace shows what went wrong.
fn debug_replay(config: &Config, path: &std::path::Path) -> Result<()> {
    let entries = trace::read_file(path)?;
    let snapshots = trace::snapshots(&entries);
    let detect = &config.detect;
    let tail = (detect.min_duration + detect.stop_grace)
        .unsigned_abs()
        .as_millis() as u64;
    let until = entries.last().map_or(0, |e| e.t_ms) + tail + 1_000;
    let mut detector = Detector::new(detect, 0);
    let mut shown = 0;
    let mut last: Option<&crate::pw::Snapshot> = None;
    let mut actions = replay(&mut detector, &snapshots, until)
        .into_iter()
        .peekable();
    let at = |t: u64| format!("{:>8.1} s", t as f64 / 1000.0);
    for (t, snap) in &snapshots {
        while let Some((ta, action)) = actions.next_if(|(ta, _)| ta < t) {
            println!("{}  => {}", at(ta), describe_action(&action));
        }
        let empty = crate::pw::Snapshot::default();
        for line in trace::describe_changes(last.unwrap_or(&empty), snap) {
            println!("{}  {line}", at(*t));
            shown += 1;
        }
        last = Some(snap);
    }
    for (ta, action) in actions {
        println!("{}  => {}", at(ta), describe_action(&action));
    }
    if shown == 0 {
        println!("no audio streams in this trace");
    }
    Ok(())
}

fn describe_action(action: &Action) -> String {
    match action {
        Action::Start { trigger } => format!(
            "start (tentative), {}",
            trigger
                .as_ref()
                .and_then(|a| a.name.clone().or_else(|| a.binary.clone()))
                .unwrap_or_else(|| "manual".into())
        ),
        Action::Commit => "recording".into(),
        Action::Finish => "finish and process".into(),
        Action::Discard => "discard".into(),
    }
}

/// Sends a control command to the daemon, which must be running, and prints its reply.
fn control(request: Request) -> Result<()> {
    match ipc::request(&ipc::socket_path()?, &request)? {
        None => bail!("the kute daemon is not running (start it with `kute daemon`)"),
        Some(Response::Ok { message }) => println!("{message}"),
        Some(Response::Status { status }) => print_status(&status),
        Some(Response::Error { message }) => bail!("{message}"),
    }
    Ok(())
}

fn print_status(status: &Status) {
    let or_dash = |v: &Option<String>| v.clone().unwrap_or_else(|| "-".into());
    println!("phase: {}", format!("{:?}", status.phase).to_lowercase());
    println!("paused: {}", if status.paused { "yes" } else { "no" });
    println!("session: {}", or_dash(&status.session));
    println!("app: {}", or_dash(&status.app));
    println!("queued: {}", status.queued);
    println!("retrying: {}", status.retrying);
    println!("running: {}", or_dash(&status.running_job));
}

/// Hands a processing job to the daemon if it is running, and prints its reply. Returns
/// whether it did.
fn queue_on_daemon(job: &Job) -> Result<bool> {
    // Without a runtime dir there is no socket, so no daemon.
    let Ok(path) = ipc::socket_path() else {
        return Ok(false);
    };
    match ipc::request(&path, &Request::Job { job: job.clone() })? {
        None => Ok(false),
        Some(Response::Ok { message }) => {
            // Until there are notifications, failures only show up in the daemon's log.
            println!(
                "{message}; the daemon logs the result (journalctl --user -u kute, or the \
                 terminal running kute daemon)"
            );
            Ok(true)
        }
        Some(Response::Error { message }) => bail!("{message}"),
        Some(Response::Status { .. }) => bail!("the daemon sent a status instead of an answer"),
    }
}

/// Queues a processing job on the daemon, or runs it here when there is no daemon.
fn run_job(store: &Store, config: &Config, job: &Job) -> Result<()> {
    if queue_on_daemon(job)? {
        return Ok(());
    }
    run_foreground(store, config, job)
}

/// Runs a processing job in the foreground and prints what it reports.
fn run_foreground(store: &Store, config: &Config, job: &Job) -> Result<()> {
    let outcome = jobs::run(store, config, &mut None, job).inspect_err(|e| {
        // What did get written is still worth showing, before the error.
        if let Some(failed) = e.downcast_ref::<ConversationsFailed>() {
            for line in &failed.written {
                println!("{line}");
            }
        }
    })?;
    if !outcome.message.is_empty() {
        println!("{}", outcome.message);
    }
    Ok(())
}

fn list(store: &Store, since: Option<&str>) -> Result<()> {
    let cutoff = match since {
        Some(s) => {
            let span: jiff::Span = s
                .parse()
                .with_context(|| format!("cannot read {s:?} as a duration (try 7d)"))?;
            Some(time::now().checked_sub(span)?)
        }
        None => None,
    };
    let mut shown = 0;
    for session in store.list()?.into_iter().rev() {
        let meta = match session.load_meta() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("kute: skipping {}: {e:#}", session.id);
                continue;
            }
        };
        if cutoff.as_ref().is_some_and(|c| meta.started_at < *c) {
            continue;
        }
        let state = format!("{:?}", meta.state).to_lowercase();
        println!(
            "{:<20} {:<28} {:>11} {:<10} {:>3} conv  {state}",
            meta.id,
            meta.title.as_deref().unwrap_or("-"),
            meta.duration_ms()
                .map(time::format_duration)
                .unwrap_or_else(|| "-".into()),
            meta.app_label(),
            meta.conversations.len(),
        );
        shown += 1;
    }
    if shown == 0 {
        println!("no sessions");
    }
    Ok(())
}

fn search(store: &Store, config: &Config, query: &str, limit: usize) -> Result<()> {
    let hits = pipeline::open_index(store)?.search(query, limit)?;
    if hits.is_empty() {
        println!("no matches");
    }
    for hit in hits {
        match hit.kind {
            HitKind::Line => {
                let at = hit
                    .started_at
                    .parse::<jiff::Zoned>()
                    .map(|z| {
                        time::add_ms(&z, hit.t_ms.unwrap_or(0))
                            .strftime("%H:%M:%S")
                            .to_string()
                    })
                    .unwrap_or_default();
                let track = if hit.track.as_deref() == Some("me") {
                    Track::Me
                } else {
                    Track::Others
                };
                println!(
                    "{}  {at}  {}: {}",
                    hit.session_id,
                    render::label(&config.labels, track),
                    hit.text
                );
            }
            HitKind::Summary => {
                println!(
                    "{}  conversation {} summary: {}",
                    hit.session_id,
                    hit.conversation,
                    hit.text.replace('\n', " ")
                );
            }
        }
    }
    Ok(())
}

fn model_list(store: &Store, config: &Config) {
    for info in models::WHISPER_MODELS.iter().chain([&models::VAD_MODEL]) {
        let present = if models::path_for(&store.models_dir(), info).is_file() {
            "downloaded"
        } else {
            "-"
        };
        let default = if info.name == config.transcribe.model {
            " (default)"
        } else {
            ""
        };
        println!(
            "{:<22} {:>6} MB  {present}{default}",
            info.name,
            info.size / 1_000_000
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn updates(tty: bool, total: u64, step: u64) -> Vec<String> {
        let mut progress = Progress::new("m", tty);
        let mut out = Vec::new();
        let mut done = 0;
        while done < total {
            done = (done + step).min(total);
            out.extend(progress.update(done, total));
        }
        out
    }

    #[test]
    fn terminal_progress_prints_only_when_the_text_changes() {
        // 1.6 GB in 64 KiB chunks would be ~25,000 updates; one decimal gives one per 100 kB.
        let lines = updates(true, 1_000_000, 65_536);
        assert_eq!(lines.first().unwrap(), "\rm: 0.1 / 1.0 MB");
        assert_eq!(lines.last().unwrap(), "\rm: 1.0 / 1.0 MB\n");
        let mut unique = lines.clone();
        unique.dedup();
        assert_eq!(unique.len(), lines.len());
    }

    #[test]
    fn piped_progress_prints_a_line_every_ten_percent() {
        let lines = updates(false, 1_600_000_000, 65_536);
        assert_eq!(lines.len(), 10);
        assert_eq!(lines[0], "m: 160.0 / 1600.0 MB (10%)\n");
        assert_eq!(lines[9], "m: 1600.0 / 1600.0 MB (100%)\n");
    }

    #[test]
    fn small_models_show_a_fraction_of_a_megabyte() {
        assert_eq!(updates(true, 885_098, 885_098), ["\rm: 0.9 / 0.9 MB\n"]);
    }
}
