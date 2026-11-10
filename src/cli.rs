//! The `kute` command line.

use crate::bench;
use crate::config::{Config, SummaryBackend, TranscribeBackend};
use crate::daemon;
use crate::detect::{Action, Detector, Phase, replay};
use crate::doctor;
use crate::index::HitKind;
use crate::ipc::{self, DaemonClient, Request, Response, Status};
use crate::jobs::{self, ConversationsFailed, Job};
use crate::models;
use crate::pipeline;
use crate::pw::trace;
use crate::render;
use crate::service;
use crate::store::{LanguageChange, Meta, Session, SessionState, SkipReason, Stage, Store, Track};
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
    /// Config file (default: ~/.config/kute/config.toml). Jobs go to the running daemon only
    /// if it uses the same file; otherwise they run here
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
    Status {
        /// Print it as JSON, for scripts: `"running": false` when no daemon is running,
        /// `"recording": true` while kute captures audio (from the moment a mic opens, before
        /// it counts as a call, until the session ends)
        #[arg(long)]
        json: bool,
    },
    /// Start recording now; it goes on until `kute stop`, or until a call it follows ends
    /// (unless detect.end_manual = false)
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
    Process {
        #[arg(required_unless_present = "all")]
        id: Option<String>,
        /// Every session with work left (failed or unfinished transcriptions and summaries)
        #[arg(long, conflicts_with = "id")]
        all: bool,
    },
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
    /// Name the others on a call, e.g. Jordan or "Jordan, Sam": their label in the
    /// transcript and the summary (no name: print it)
    Others {
        id: String,
        name: Vec<String>,
        /// Use `labels.others` from the config again
        #[arg(long, conflicts_with = "name")]
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
    /// Open a session's folder (with notify.open, xdg-open by default)
    Open { id: String },
    /// Summarize a session again (all its conversations, or one with --conversation), or
    /// with --all every conversation in every session that has no summary yet
    Summarize {
        #[arg(required_unless_present = "all")]
        id: Option<String>,
        /// Every conversation in every session that has no summary yet: never summarized
        /// (no summarizer was set up) or failed. Ones too short to summarize stay skipped
        #[arg(long, conflicts_with_all = ["id", "conversation", "lang"])]
        all: bool,
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
    Doctor {
        /// Time transcription on the GPU and on the CPU, and offer to use the faster one
        #[arg(long)]
        bench: bool,
        /// With --bench: time this recording (any format ffmpeg reads) instead of a built-in
        /// clip
        #[arg(long, requires = "bench", value_name = "FILE")]
        sample: Option<PathBuf>,
        /// With --bench: set transcribe.gpu in the config without asking
        #[arg(long, requires = "bench")]
        save: bool,
    },
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
    /// Download a Whisper model (default: the configured one, if the backend is whisper) and
    /// the VAD model
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
    if !matches!(cli.command, Command::Daemon) {
        die_quietly_on_closed_pipe();
    }
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
    if let Command::Doctor {
        bench,
        sample,
        save,
    } = &cli.command
    {
        if *bench {
            return bench::run(bench::Options {
                config_path: cli.config.as_deref(),
                sample: sample.as_deref(),
                save: *save,
            });
        }
        return doctor::run(cli.config.as_deref());
    }
    let config = Config::load(cli.config.as_deref())?;
    let store = Store::open(config.data_dir()?)?;

    match cli.command {
        Command::Daemon => daemon::run(store, config)?,
        Command::Status { json: false } => control(Request::Status)?,
        Command::Status { json: true } => status_json()?,
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
        Command::Process { id: Some(id), .. } => process(&store, &config, &store.resolve(&id)?)?,
        Command::Process { id: None, .. } => process_all(&store, &config)?,
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
        Command::Others { id, name, clear } => {
            let session = store.resolve(&id)?;
            if name.is_empty() && !clear {
                println!("{}", session.load_meta()?.labels(&config.labels).others);
            } else {
                run_job(
                    &store,
                    &config,
                    &Job::Others {
                        id: session.id,
                        name: (!clear).then(|| name.join(" ")),
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
            let Some((program, args)) = config.notify.open.split_first() else {
                bail!("notify.open is empty, so there is nothing to open the folder with");
            };
            std::process::Command::new(program)
                .args(args)
                .arg(&session.dir)
                .spawn()
                .with_context(|| format!("running {program} (notify.open)"))?;
        }
        Command::Summarize {
            id: None, backend, ..
        } => summarize_all(&store, &config, backend.map(Into::into))?,
        Command::Summarize {
            id: Some(id),
            conversation,
            lang,
            backend,
            ..
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
        | Command::Doctor { .. }
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
            // A command backend transcribes with its own model and needs only the VAD model,
            // unless a Whisper model is asked for by name.
            let name = name.or_else(|| {
                (config.transcribe.backend == TranscribeBackend::Whisper)
                    .then(|| config.transcribe.model.clone())
            });
            let mut wanted = Vec::new();
            if let Some(name) = name {
                wanted.push(
                    models::find_whisper(&name)
                        .with_context(|| format!("unknown Whisper model {name:?}"))?,
                );
            }
            wanted.push(&models::VAD_MODEL);
            for info in wanted {
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
        Some(Response::Identity { .. }) => {
            bail!("the daemon sent its identity instead of an answer")
        }
    }
    Ok(())
}

/// Prints the daemon's status as one line of JSON. No daemon is not an error here: a script
/// asking whether it may restart kute wants an answer either way.
fn status_json() -> Result<()> {
    let status = match ipc::request(&ipc::socket_path()?, &Request::Status)? {
        None => None,
        Some(Response::Status { status }) => Some(status),
        Some(Response::Error { message }) => bail!("{message}"),
        Some(_) => bail!("the daemon sent another reply instead of its status"),
    };
    println!(
        "{}",
        serde_json::to_string(&status_document(status.as_ref()))?
    );
    Ok(())
}

/// What `kute status --json` prints: the daemon's status plus `running` and `recording`.
fn status_document(status: Option<&Status>) -> serde_json::Value {
    let Some(status) = status else {
        return serde_json::json!({ "running": false, "recording": false });
    };
    let mut doc = serde_json::to_value(status).expect("a status always serializes");
    doc["running"] = true.into();
    // Grace is still a recording: the call may come back.
    doc["recording"] = (status.phase != Phase::Idle).into();
    doc
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

/// Makes writing to a closed stdout end the process silently, as other command line tools
/// do, instead of panicking: Rust ignores SIGPIPE, so `kute list | head` would otherwise
/// fail in `println!`. Not for the daemon, which must survive a client hanging up on it.
fn die_quietly_on_closed_pipe() {
    // SAFETY: setting a signal's disposition to the default has no preconditions, and
    // runs before any other thread exists.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

/// Hands a processing job to the daemon if one runs with this config on this archive (see
/// `DaemonClient::connect`), and prints its reply. Returns whether it did.
fn queue_on_daemon(store: &Store, config: &Config, job: &Job) -> Result<bool> {
    let Some(daemon) = DaemonClient::connect(store.root(), config.source.as_deref())? else {
        return Ok(false);
    };
    let Some(message) = daemon.queue(job)? else {
        return Ok(false);
    };
    // Until there are notifications, failures only show up in the daemon's log.
    println!(
        "{message}; the daemon logs the result (journalctl --user -u kute, or the \
         terminal running kute daemon)"
    );
    Ok(true)
}

/// Queues a processing job on the daemon, or runs it here when there is no daemon.
fn run_job(store: &Store, config: &Config, job: &Job) -> Result<()> {
    if queue_on_daemon(store, config, job)? {
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

/// `kute process <id>`: on the daemon, or here when none runs.
fn process(store: &Store, config: &Config, session: &Session) -> Result<()> {
    let job = Job::Process {
        id: session.id.clone(),
    };
    if !queue_on_daemon(store, config, &job)? {
        // The daemon's worker processes a recording session's closed conversations;
        // a second process doing the same at the same time would race it.
        if session.load_meta()?.state == SessionState::Recording {
            bail!("session {} is still recording", session.id);
        }
        run_foreground(store, config, &job)?;
    }
    Ok(())
}

/// Whether `kute process --all` has work for this session: a conversation not yet
/// transcribed or summarized (or failed at it), or a session left processing.
fn needs_processing(meta: &Meta) -> bool {
    match meta.state {
        SessionState::Recording => false,
        SessionState::Processing => true,
        SessionState::Done => meta
            .conversations
            .iter()
            .any(|c| c.needs_transcription() || c.needs_summary()),
    }
}

/// The sessions to go through for `--all`, oldest first, skipping ones whose meta can't be
/// read.
fn sessions_with_meta(store: &Store) -> Result<Vec<(Session, Meta)>> {
    let mut sessions = Vec::new();
    for session in store.list()? {
        match session.load_meta() {
            Ok(meta) => sessions.push((session, meta)),
            Err(e) => eprintln!("kute: skipping {}: {e:#}", session.id),
        }
    }
    Ok(sessions)
}

/// Runs `each` on every session it picks; a failure is printed and the rest still run.
fn for_each_session(
    picked: Vec<(Session, Meta)>,
    nothing: &str,
    mut each: impl FnMut(&Session, &Meta) -> Result<()>,
) -> Result<()> {
    if picked.is_empty() {
        println!("{nothing}");
        return Ok(());
    }
    let mut failed = 0;
    for (session, meta) in &picked {
        if let Err(e) = each(session, meta) {
            eprintln!("kute: {}: {e:#}", session.id);
            failed += 1;
        }
    }
    if failed > 0 {
        bail!("{failed} of {} session(s) failed", picked.len());
    }
    Ok(())
}

fn process_all(store: &Store, config: &Config) -> Result<()> {
    let picked = sessions_with_meta(store)?
        .into_iter()
        .filter(|(_, meta)| needs_processing(meta))
        .collect();
    for_each_session(picked, "no session has work left", |session, _| {
        process(store, config, session)
    })
}

/// What `kute summarize --all` does for this session, if anything: the conversations
/// transcribed and never summarized for want of a summarizer, or failed at it. The
/// skipped ones are reset first; then one summarize job runs, which summarizes every
/// conversation waiting for a summary, each once. Its conversation is `None` when none of
/// the session's transcribed conversations has a summary, so it redoes the session's too.
fn missing_summaries(meta: &Meta) -> Option<Backfill> {
    if meta.state == SessionState::Recording {
        return None;
    }
    let transcribed: Vec<_> = meta
        .conversations
        .iter()
        .filter(|c| !c.needs_transcription())
        .collect();
    // Sessions from before `skip_reason` don't say why; the pipeline skips the too short
    // ones again.
    let skipped: Vec<u32> = transcribed
        .iter()
        .filter(|c| {
            c.stage == Stage::Skipped
                && matches!(c.skip_reason, None | Some(SkipReason::NoSummarizer))
        })
        .map(|c| c.n)
        .collect();
    let missing: Vec<u32> = transcribed
        .iter()
        .filter(|c| c.needs_summary() || skipped.contains(&c.n))
        .map(|c| c.n)
        .collect();
    let first = *missing.first()?;
    if missing.len() == transcribed.len() {
        // The job resets them all itself.
        return Some(Backfill {
            reset: Vec::new(),
            conversation: None,
        });
    }
    Some(Backfill {
        reset: skipped,
        conversation: Some(first),
    })
}

#[derive(Debug, PartialEq)]
struct Backfill {
    /// Skipped conversations to mark as waiting for a summary before the job runs.
    reset: Vec<u32>,
    conversation: Option<u32>,
}

fn summarize_all(store: &Store, config: &Config, backend: Option<SummaryBackend>) -> Result<()> {
    if backend.unwrap_or(config.summary.backend) == SummaryBackend::None {
        bail!(
            "no summarizer is configured (summary.backend = \"none\"); set one in the config or pass --backend"
        );
    }
    let picked = sessions_with_meta(store)?
        .into_iter()
        .filter(|(_, meta)| missing_summaries(meta).is_some())
        .collect();
    for_each_session(
        picked,
        "every conversation has a summary",
        |session, meta| {
            let Some(backfill) = missing_summaries(meta) else {
                return Ok(());
            };
            if !backfill.reset.is_empty() {
                session.update_meta(|m| {
                    for c in &mut m.conversations {
                        if backfill.reset.contains(&c.n) {
                            c.stage = Stage::Transcribed;
                            c.clear_error();
                        }
                    }
                })?;
            }
            run_job(
                store,
                config,
                &Job::Summarize {
                    id: session.id.clone(),
                    conversation: backfill.conversation,
                    lang: None,
                    backend,
                },
            )
        },
    )
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
        // Without a summarizer every conversation is skipped, and that is no news.
        for c in &meta.conversations {
            if c.stage == Stage::Skipped
                && let Some(reason @ SkipReason::TooShort { .. }) = c.skip_reason
            {
                println!("    conversation {}: no summary, {reason}", c.n);
            }
        }
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
    // Each session's labels: it may name the others (`kute others`).
    let mut labels = std::collections::HashMap::new();
    for hit in hits {
        match hit.kind {
            HitKind::Line => {
                let labels = labels.entry(hit.session_id.clone()).or_insert_with(|| {
                    store
                        .resolve(&hit.session_id)
                        .and_then(|s| s.load_meta())
                        .map(|m| m.labels(&config.labels))
                        .unwrap_or_else(|_| config.labels.clone())
                });
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
                    render::label(labels, track),
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

    #[test]
    fn status_json_says_whether_a_call_is_being_recorded() {
        assert_eq!(
            status_document(None),
            serde_json::json!({ "running": false, "recording": false })
        );
        let mut status = Status {
            phase: Phase::Grace,
            paused: false,
            session: Some("2026-10-08T090000".into()),
            app: Some("Firefox".into()),
            queued: 1,
            retrying: 0,
            running_job: None,
        };
        let doc = status_document(Some(&status));
        assert_eq!(doc["running"], true);
        assert_eq!(doc["recording"], true);
        assert_eq!(doc["phase"], "grace");
        assert_eq!(doc["queued"], 1);
        assert_eq!(doc["session"], "2026-10-08T090000");
        status.phase = Phase::Idle;
        assert_eq!(status_document(Some(&status))["recording"], false);
    }

    fn meta(state: SessionState, stages: &[Stage]) -> Meta {
        use crate::store::{Conversation, Trigger, TriggerKind};
        let mut meta = Meta::new(
            "2026-10-19T100000".into(),
            "2026-10-19T10:00:00+02:00[+02:00]".parse().unwrap(),
            Trigger {
                kind: TriggerKind::Import,
                app_name: None,
                binary: None,
            },
            "auto".into(),
        );
        meta.state = state;
        for (i, stage) in stages.iter().enumerate() {
            let mut c = Conversation::new(i as u32 + 1, 0, 5_000);
            c.stage = *stage;
            if *stage == Stage::Skipped {
                c.skip_reason = Some(SkipReason::NoSummarizer);
            }
            meta.conversations.push(c);
        }
        meta
    }

    #[test]
    fn process_all_picks_sessions_with_work_left() {
        use Stage::*;
        assert!(needs_processing(&meta(SessionState::Processing, &[])));
        assert!(needs_processing(&meta(
            SessionState::Done,
            &[Summarized, Transcribed]
        )));
        assert!(!needs_processing(&meta(
            SessionState::Done,
            &[Summarized, Skipped]
        )));
        // The daemon is still on it.
        assert!(!needs_processing(&meta(
            SessionState::Recording,
            &[Pending]
        )));
    }

    #[test]
    fn summarize_all_picks_conversations_without_a_summary() {
        use Stage::*;
        assert_eq!(
            missing_summaries(&meta(SessionState::Done, &[Summarized])),
            None
        );
        let all = Backfill {
            reset: vec![],
            conversation: None,
        };
        let some = |reset: Vec<u32>, n| Backfill {
            reset,
            conversation: Some(n),
        };
        // All of them: the job redoes the whole session.
        assert_eq!(
            missing_summaries(&meta(SessionState::Done, &[Skipped, Skipped, Pending])),
            Some(all)
        );
        // Only some: the summaries already written stay. The skipped ones are reset, then
        // one job summarizes every waiting conversation once.
        assert_eq!(
            missing_summaries(&meta(
                SessionState::Done,
                &[Summarized, Skipped, Transcribed, Skipped]
            )),
            Some(some(vec![2, 4], 2))
        );
        assert_eq!(
            missing_summaries(&meta(
                SessionState::Done,
                &[Summarized, Transcribed, Transcribed]
            )),
            Some(some(vec![], 2))
        );
        let mut short = meta(SessionState::Done, &[Skipped]);
        short.conversations[0].skip_reason = Some(SkipReason::TooShort {
            words: 3,
            min_words: 50,
        });
        assert_eq!(missing_summaries(&short), None);
        // Skipped before kute recorded why: maybe for want of a summarizer.
        short.conversations[0].skip_reason = None;
        assert_eq!(
            missing_summaries(&short),
            Some(Backfill {
                reset: vec![],
                conversation: None
            })
        );
        assert_eq!(
            missing_summaries(&meta(SessionState::Recording, &[Skipped])),
            None
        );
    }

    #[test]
    fn all_replaces_the_session_id() {
        let parse = |args: &[&str]| Cli::try_parse_from([&["kute"], args].concat());
        assert!(parse(&["summarize", "--all"]).is_ok());
        assert!(parse(&["process", "--all"]).is_ok());
        assert!(parse(&["summarize"]).is_err());
        assert!(parse(&["process"]).is_err());
        assert!(parse(&["summarize", "x", "--all"]).is_err());
        assert!(parse(&["summarize", "--all", "--lang", "it"]).is_err());
    }

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
