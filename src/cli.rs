//! The `kute` command line.

use crate::config::{Config, SummaryBackend};
use crate::import::{self, ImportRequest};
use crate::index::{HitKind, Index};
use crate::models;
use crate::pipeline::{self, AsrProvider, Pipeline, Report, WhisperProvider};
use crate::render;
use crate::store::{Session, Store, Track};
use crate::summarize;
use crate::time;
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
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
    /// Import two recordings (your mic, everyone else) and process them
    Import {
        me: PathBuf,
        others: PathBuf,
        /// Start time, e.g. 2026-10-08T15:00:00 (default: <prefix>.start, then file birth time)
        #[arg(long)]
        start: Option<String>,
        #[arg(long)]
        title: Option<String>,
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
    },
    /// Rebuild the search index from the session folders
    Reindex,
    /// Manage Whisper and VAD models
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
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
    let mut config = Config::load(cli.config.as_deref())?;
    let store = Store::open(config.data_dir()?)?;

    match cli.command {
        Command::Import {
            me,
            others,
            start,
            title,
            no_process,
        } => {
            // Check the models first, so a missing model doesn't leave a half-done import.
            let provider = if no_process {
                None
            } else {
                Some(WhisperProvider::new(&config, &store)?)
            };
            let session = import::import(
                &store,
                &config,
                &ImportRequest {
                    me,
                    others,
                    start,
                    title,
                },
            )?;
            println!("imported {}", session.id);
            if let Some(mut provider) = provider {
                run_pipeline(
                    &store,
                    &config,
                    &session,
                    Some(&mut provider),
                    Work::Process,
                )?;
            }
        }
        Command::Process { id } => {
            let session = store.resolve(&id)?;
            let mut provider = WhisperProvider::new(&config, &store)?;
            run_pipeline(
                &store,
                &config,
                &session,
                Some(&mut provider),
                Work::Process,
            )?;
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
        Command::Title { id, title } => {
            let session = store.resolve(&id)?;
            let mut meta = session.load_meta()?;
            meta.title = Some(title.join(" "));
            session.save_meta(&meta)?;
            pipeline::write_documents(&session, &meta, &config.labels)?;
            pipeline::index_session(&Index::open(&store.index_path())?, &session)?;
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
        } => {
            let session = store.resolve(&id)?;
            if let Some(lang) = lang {
                let mut meta = session.load_meta()?;
                meta.summary.language = lang;
                session.save_meta(&meta)?;
            }
            if let Some(b) = backend {
                config.summary.backend = b.into();
            }
            run_pipeline(
                &store,
                &config,
                &session,
                None,
                Work::Summarize(conversation),
            )?;
        }
        Command::Retranscribe { id, model } => {
            let session = store.resolve(&id)?;
            if let Some(model) = model {
                config.transcribe.model = model;
            }
            let mut provider = WhisperProvider::new(&config, &store)?;
            run_pipeline(
                &store,
                &config,
                &session,
                Some(&mut provider),
                Work::Retranscribe,
            )?;
        }
        Command::Reindex => {
            let n = pipeline::reindex_all(&store, &Index::open(&store.index_path())?)?;
            println!("indexed {n} sessions");
        }
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

enum Work {
    Process,
    Retranscribe,
    Summarize(Option<u32>),
}

fn run_pipeline(
    store: &Store,
    config: &Config,
    session: &Session,
    asr: Option<&mut WhisperProvider>,
    work: Work,
) -> Result<()> {
    let index = Index::open(&store.index_path())?;
    let summarizer = summarize::from_config(&config.summary)?;
    let mut pipeline = Pipeline {
        index: &index,
        config,
        asr: asr.map(|p| p as &mut dyn AsrProvider),
        summarizer: summarizer.as_deref(),
        template: summarize::load_template(&config.summary)?,
        quiet: false,
    };
    let report = match work {
        Work::Process => pipeline.process(session)?,
        Work::Retranscribe => pipeline.retranscribe(session)?,
        Work::Summarize(only) => pipeline.resummarize(session, only)?,
    };
    print_report(session, &report)
}

fn print_report(session: &Session, report: &Report) -> Result<()> {
    if session.transcript_path().is_file() {
        println!("transcript: {}", session.transcript_path().display());
    }
    if session.summary_path().is_file() {
        println!("summary:    {}", session.summary_path().display());
    }
    if let Some(e) = &report.compression_error {
        eprintln!("kute: compression failed (audio kept as WAV): {e}");
    }
    if !report.failed.is_empty() {
        for (n, e) in &report.failed {
            eprintln!("kute: conversation {n} failed: {e}");
        }
        bail!(
            "{} conversation(s) failed; fix the cause and run `kute process {}`",
            report.failed.len(),
            session.id
        );
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
    for session in store.list()? {
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
    let hits = Index::open(&store.index_path())?.search(query, limit)?;
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
