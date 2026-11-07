//! Desktop notifications from the daemon: a call is being recorded (with Stop and
//! Discard), summaries are ready (opens `summary.md` when clicked), and something failed.

use crate::config::NotifyConfig;
use crate::jobs::Problems;
use crate::render;
use crate::store::{Meta, Session, Stage, Step, Store};
use std::collections::HashMap;
use std::process::Command;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

/// The key of the "Recording" note, withdrawn when the recording ends.
pub const RECORDING: &str = "recording";

/// What clicking a note, or one of its buttons, does. Sessions are named by ID and looked up
/// at click time, since the folder may have been renamed since.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Click {
    OpenSummary(String),
    OpenFolder(String),
    /// Stop the session, if it is still the one recording.
    Stop(String),
    /// Stop and delete the session, if it is still the one recording.
    Discard(String),
}

/// One notification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub title: String,
    /// Markup on KDE (and in the freedesktop spec): escape what isn't kute's own. The title
    /// is plain text.
    pub body: String,
    /// Clicking the note itself.
    pub default: Option<Click>,
    pub buttons: Vec<(String, Click)>,
    /// Names a note that `Notifier::withdraw` may take back.
    pub key: Option<String>,
}

/// Sends notes somewhere. `Desktop` in the daemon, `Silent` by default, a recorder in tests.
pub trait Notifier: Send {
    fn notify(&self, note: Note);
    /// Takes back the note shown with this key, if it is still up.
    fn withdraw(&self, _key: &str) {}
}

/// Drops every note.
pub struct Silent;

impl Notifier for Silent {
    fn notify(&self, _: Note) {}
}

/// "Recording: Firefox" ("Recording" for a manual session), with Stop and Discard.
pub fn recording_note(id: &str, app: Option<&str>) -> Note {
    Note {
        title: app.map_or_else(|| "Recording".into(), |app| format!("Recording: {app}")),
        body: "Stop keeps the recording and processes it; Discard deletes it.".into(),
        default: None,
        buttons: vec![
            ("Stop".into(), Click::Stop(id.into())),
            ("Discard".into(), Click::Discard(id.into())),
        ],
        key: Some(RECORDING.into()),
    }
}

/// A recording kute had to end (disk full, the capture died). `kept` is whether its session
/// is still there (a Tentative one with no audio is deleted).
pub fn recording_failed_note(id: &str, app: Option<&str>, reason: &str, kept: bool) -> Note {
    let mut body = escape(&cut(reason, FIRST_LINE_CHARS));
    if kept {
        body.push_str("\nWhat was recorded is kept.");
    }
    Note {
        title: app.map_or_else(
            || "Recording stopped".into(),
            |app| format!("Recording stopped: {app}"),
        ),
        body,
        default: kept.then(|| Click::OpenFolder(id.into())),
        buttons: Vec::new(),
        key: None,
    }
}

/// A job that failed outright (not a conversation inside it), e.g. a missing model.
pub fn job_failed_note(job: &str, error: &str) -> Note {
    Note {
        title: format!("kute: {job} failed"),
        body: escape(&cut(error, 2 * FIRST_LINE_CHARS)),
        default: None,
        buttons: Vec::new(),
        key: None,
    }
}

/// The note for what a job left failed, now that nothing more will happen on its own: a
/// transcription only after its last retry, a summary and compression at once. `None` when
/// there is nothing to say yet.
pub fn problems_note(session: &Session, meta: &Meta, problems: &Problems) -> Option<Note> {
    let mut lines = Vec::new();
    for c in problems
        .conversations
        .iter()
        .filter_map(|&n| meta.conversation(n))
        .filter(|c| c.stage == Stage::Failed)
    {
        let what = match c.failed_step {
            Some(Step::Transcribe) if crate::jobs::retries_left(c) => continue,
            Some(Step::Transcribe) => format!("transcription failed after {} tries", c.attempts),
            Some(Step::Summarize) => "the summary failed".into(),
            None => "failed".into(),
        };
        let error = c
            .error
            .as_deref()
            .map(|e| format!(": {}", cut(e, FIRST_LINE_CHARS)));
        lines.push(escape(&format!(
            "{}, {what}{}",
            render::time_range(meta, c),
            error.unwrap_or_default()
        )));
    }
    if let Some(e) = &problems.compression {
        lines.push(escape(&format!(
            "compressing the audio failed (WAVs kept): {}",
            cut(e, FIRST_LINE_CHARS)
        )));
    }
    if lines.is_empty() {
        return None;
    }
    lines.push(escape(&format!(
        "Run `kute process {}` once it is fixed.",
        meta.id
    )));
    let name = meta.title.clone().unwrap_or_else(|| meta.app_label());
    Some(Note {
        title: format!("Processing failed: {name}"),
        body: lines.join("\n"),
        default: Some(Click::OpenFolder(session.id.clone())),
        buttons: Vec::new(),
        key: None,
    })
}

/// The note for the summaries a job just wrote: details for one conversation, a count and
/// the time ranges for several. `None` when there is nothing to announce.
pub fn summary_note(session: &Session, meta: &Meta, summarized: &[u32]) -> Option<Note> {
    let conversations: Vec<_> = summarized
        .iter()
        .filter_map(|&n| meta.conversation(n))
        .collect();
    let name = meta.title.clone().unwrap_or_else(|| meta.app_label());
    let (title, body) = match conversations.as_slice() {
        [] => return None,
        [c] => {
            let mut body = render::time_range(meta, c);
            if let Some(line) = std::fs::read_to_string(session.conversation_summary_path(c.n))
                .ok()
                .and_then(|text| first_line(&text))
            {
                body = format!("{body}\n{}", escape(&line));
            }
            (format!("Summary ready: {name}"), body)
        }
        several => (
            format!("{} summaries ready: {name}", several.len()),
            several
                .iter()
                .map(|c| render::time_range(meta, c))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
    };
    let open = Click::OpenSummary(session.id.clone());
    Some(Note {
        title,
        body,
        default: Some(open.clone()),
        buttons: vec![("Open summary".into(), open)],
        key: None,
    })
}

/// Longest first line shown, in characters.
const FIRST_LINE_CHARS: usize = 140;

/// The summary's first line of prose: headings and blank lines skipped, list markers removed,
/// cut to `FIRST_LINE_CHARS`.
fn first_line(summary: &str) -> Option<String> {
    let line = summary
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))?;
    let line = line
        .strip_prefix("- ")
        .or_else(|| line.strip_prefix("* "))
        .unwrap_or(line)
        .trim();
    Some(cut(line, FIRST_LINE_CHARS))
}

/// `text` on one line, cut to `max` characters with an ellipsis.
fn cut(text: &str, max: usize) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= max {
        return text;
    }
    let cut: String = text.chars().take(max).collect();
    format!("{}…", cut.trim_end())
}

/// Notification bodies are markup on KDE (and in the freedesktop spec).
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Real desktop notifications through the freedesktop notification service. Opening a
/// summary or a folder runs `notify.open` on it; Stop and Discard go to the daemon.
pub struct Desktop {
    inner: Arc<Inner>,
}

struct Inner {
    open: Vec<String>,
    store: Store,
    clicks: Mutex<Sender<Click>>,
    /// Each keyed note not yet gone, by key.
    shown: Mutex<HashMap<String, Slot>>,
    next_token: std::sync::atomic::AtomicU64,
}

/// A keyed note: `token` tells it from a later note with the same key.
struct Slot {
    token: u64,
    state: SlotState,
}

enum SlotState {
    /// Being shown; no notification ID yet.
    Pending,
    Shown(u32),
    /// Withdrawn before it was up: close it as soon as it is.
    Withdrawn,
}

impl Desktop {
    pub fn new(config: &NotifyConfig, store: Store, clicks: Sender<Click>) -> Desktop {
        Desktop {
            inner: Arc::new(Inner {
                open: config.open.clone(),
                store,
                clicks: Mutex::new(clicks),
                shown: Mutex::new(HashMap::new()),
                next_token: std::sync::atomic::AtomicU64::new(0),
            }),
        }
    }
}

impl Notifier for Desktop {
    fn notify(&self, note: Note) {
        let inner = self.inner.clone();
        // Registered before the thread starts, so a withdraw that comes before the
        // notification is up still finds it.
        let token = inner
            .next_token
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(key) = &note.key {
            let slot = Slot {
                token,
                state: SlotState::Pending,
            };
            inner.shown.lock().unwrap().insert(key.clone(), slot);
        }
        // Waiting for a click blocks, so each note gets a thread that ends when the
        // notification is clicked, dismissed or expires.
        std::thread::spawn(move || {
            let mut notification = notify_rust::Notification::new();
            notification
                .appname("kute")
                .summary(&note.title)
                .body(&note.body)
                .icon("audio-input-microphone");
            if note.default.is_some() {
                notification.action("default", "Open");
            }
            for (i, (label, _)) in note.buttons.iter().enumerate() {
                notification.action(&format!("button-{i}"), label);
            }
            let shown = notification.show();
            let handle = match shown {
                Ok(handle) => handle,
                Err(e) => {
                    eprintln!("kute: could not show a notification: {e}");
                    if let Some(key) = &note.key {
                        inner.forget(key, token);
                    }
                    return;
                }
            };
            if let Some(key) = &note.key {
                let mut shown = inner.shown.lock().unwrap();
                match shown.get_mut(key) {
                    Some(slot) if slot.token == token => {
                        if matches!(slot.state, SlotState::Withdrawn) {
                            shown.remove(key);
                            drop(shown);
                            handle.close();
                            return;
                        }
                        slot.state = SlotState::Shown(handle.id());
                    }
                    // A later note with this key took over; this one is stale already.
                    _ => {
                        drop(shown);
                        handle.close();
                        return;
                    }
                }
            }
            let mut chosen = None;
            handle.wait_for_action(|action| chosen = Some(action.to_string()));
            // Gone from the screen either way: nothing to withdraw any more.
            if let Some(key) = &note.key {
                inner.forget(key, token);
            }
            let click = match chosen.as_deref() {
                Some("default") => note.default.clone(),
                Some(action) => action
                    .strip_prefix("button-")
                    .and_then(|i| i.parse::<usize>().ok())
                    .and_then(|i| note.buttons.get(i))
                    .map(|(_, click)| click.clone()),
                None => None,
            };
            if let Some(click) = click {
                inner.clicked(click);
            }
        });
    }

    fn withdraw(&self, key: &str) {
        let id = {
            let mut shown = self.inner.shown.lock().unwrap();
            let Some(slot) = shown.get_mut(key) else {
                return;
            };
            match slot.state {
                SlotState::Shown(id) => {
                    shown.remove(key);
                    id
                }
                SlotState::Pending | SlotState::Withdrawn => {
                    slot.state = SlotState::Withdrawn;
                    return;
                }
            }
        };
        // The spec closes a notification by ID, which notify-rust only offers on the handle
        // the waiting thread holds; replacing it with a note that is closed at once does it.
        std::thread::spawn(move || {
            match notify_rust::Notification::new()
                .appname("kute")
                .summary("kute")
                .id(id)
                .show()
            {
                Ok(handle) => handle.close(),
                Err(e) => eprintln!("kute: could not close a notification: {e}"),
            }
        });
    }
}

impl Inner {
    /// Drops the slot for `key` if it is still the note with `token`.
    fn forget(&self, key: &str, token: u64) {
        let mut shown = self.shown.lock().unwrap();
        if shown.get(key).is_some_and(|slot| slot.token == token) {
            shown.remove(key);
        }
    }

    fn clicked(&self, click: Click) {
        let (id, file) = match &click {
            Click::OpenSummary(id) => (id, true),
            Click::OpenFolder(id) => (id, false),
            Click::Stop(_) | Click::Discard(_) => {
                let _ = self.clicks.lock().unwrap().send(click);
                return;
            }
        };
        let session = match find_session(&self.store, id) {
            Ok(session) => session,
            Err(e) => {
                eprintln!("kute: cannot open {id}: {e:#}");
                return;
            }
        };
        let path = if file {
            session.summary_path()
        } else {
            session.dir.clone()
        };
        open_with(&self.open, &path);
    }
}

/// Runs the `notify.open` command on `path` and waits for it, logging a failure.
pub fn open_with(open: &[String], path: &std::path::Path) {
    let Some((program, args)) = open.split_first() else {
        return;
    };
    let mut command = Command::new(program);
    command.args(args).arg(path);
    // A service started before the desktop session has no display: take the one the
    // session gave systemd since.
    for (key, value) in graphical_env() {
        command.env(key, value);
    }
    match command.status() {
        Ok(status) if !status.success() => {
            eprintln!("kute: {program} {} failed ({status})", path.display())
        }
        Ok(_) => {}
        Err(e) => eprintln!("kute: could not run {program}: {e}"),
    }
}

/// The session a note was shown for, whose folder may have been renamed since: by folder
/// name or prefix first, then by the ID in each `meta.toml`.
fn find_session(store: &Store, id: &str) -> anyhow::Result<Session> {
    if let Ok(session) = store.resolve(id) {
        return Ok(session);
    }
    store
        .list()?
        .into_iter()
        .find(|s| s.load_meta().is_ok_and(|m| m.id == id))
        .ok_or_else(|| anyhow::anyhow!("no session {id:?} (was its folder deleted?)"))
}

/// What a program that opens a window needs to know about the desktop.
const GRAPHICAL: &[&str] = &[
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_DESKTOP",
    "XDG_SESSION_TYPE",
    "DESKTOP_SESSION",
    "KDE_FULL_SESSION",
    "KDE_SESSION_VERSION",
];

/// The desktop's variables from the systemd user manager, which the desktop session updates
/// when it starts. Empty when systemd can't be asked.
pub fn graphical_env() -> Vec<(String, String)> {
    let output = Command::new("systemctl")
        .args(["--user", "show-environment"])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            parse_graphical_env(&String::from_utf8_lossy(&out.stdout))
        }
        _ => Vec::new(),
    }
}

/// The `GRAPHICAL` variables in `systemctl --user show-environment` output. Values systemd
/// had to quote (`$'...'`) are left out: none of these needs quoting.
fn parse_graphical_env(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| line.split_once('='))
        .filter(|(key, value)| GRAPHICAL.contains(key) && !value.starts_with("$'"))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Conversation, Store, Trigger, TriggerKind};

    fn session_with(conversations: &[(u32, u64, u64, &str)]) -> (tempfile::TempDir, Session, Meta) {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let start: jiff::Zoned = "2026-10-21T15:15:21+02:00[+02:00]".parse().unwrap();
        let session = store.create_session(&start).unwrap();
        let trigger = Trigger {
            kind: TriggerKind::App,
            app_name: Some("Element".into()),
            binary: Some("element-desktop".into()),
        };
        let mut meta = Meta::new(session.id.clone(), start, trigger, "en".into());
        std::fs::create_dir_all(session.summaries_dir()).unwrap();
        for (n, start_ms, end_ms, summary) in conversations {
            meta.conversations
                .push(Conversation::new(*n, *start_ms, *end_ms));
            std::fs::write(session.conversation_summary_path(*n), summary).unwrap();
        }
        session.save_meta(&meta).unwrap();
        (tmp, session, meta)
    }

    #[test]
    fn one_summary_names_its_time_and_first_line() {
        let summary = "## Summary\n\n- Others explained a design problem in the agent's main loop.\n- More.\n";
        let (_tmp, session, meta) = session_with(&[(1, 0, 420_000, summary)]);
        let note = summary_note(&session, &meta, &[1]).unwrap();
        assert_eq!(note.title, "Summary ready: Element");
        assert_eq!(
            note.body,
            "15:15 to 15:22\nOthers explained a design problem in the agent's main loop."
        );
        assert_eq!(note.default, Some(Click::OpenSummary(session.id.clone())));
        assert_eq!(note.buttons.len(), 1);
    }

    #[test]
    fn the_title_prefers_the_session_title() {
        let (_tmp, session, mut meta) = session_with(&[(1, 0, 60_000, "Plain text.\n")]);
        meta.title = Some("Standup <weekly>".into());
        let note = summary_note(&session, &meta, &[1]).unwrap();
        assert_eq!(note.title, "Summary ready: Standup <weekly>");
        assert_eq!(note.body, "15:15 to 15:16\nPlain text.");
    }

    #[test]
    fn the_body_escapes_markup() {
        // KDE renders the body as markup, so a summary about C++ & <templates> must survive.
        let (_tmp, session, meta) = session_with(&[(1, 0, 60_000, "C++ & <templates>\n")]);
        let note = summary_note(&session, &meta, &[1]).unwrap();
        assert_eq!(note.body, "15:15 to 15:16\nC++ &amp; &lt;templates&gt;");
    }

    #[test]
    fn several_summaries_make_one_note() {
        let (_tmp, session, meta) = session_with(&[
            (1, 0, 60_000, "One.\n"),
            (2, 600_000, 660_000, "Two.\n"),
            (3, 1_200_000, 1_260_000, "Three.\n"),
        ]);
        let note = summary_note(&session, &meta, &[1, 3]).unwrap();
        assert_eq!(note.title, "2 summaries ready: Element");
        assert_eq!(note.body, "15:15 to 15:16\n15:35 to 15:36");
    }

    #[test]
    fn a_long_first_line_is_cut() {
        let long = "word ".repeat(60);
        let (_tmp, session, meta) = session_with(&[(1, 0, 60_000, &long)]);
        let body = summary_note(&session, &meta, &[1]).unwrap().body;
        let line = body.lines().nth(1).unwrap();
        assert!(line.chars().count() <= 141, "{line}");
        assert!(line.ends_with('…'));
    }

    #[test]
    fn nothing_summarized_means_no_note() {
        let (_tmp, session, meta) = session_with(&[(1, 0, 60_000, "One.\n")]);
        assert_eq!(summary_note(&session, &meta, &[]), None);
        assert_eq!(summary_note(&session, &meta, &[9]), None);
    }

    #[test]
    fn the_desktop_variables_come_from_systemd() {
        let text = "HOME=/home/alex\nDISPLAY=:1\nWAYLAND_DISPLAY=wayland-1\n\
                    XDG_CURRENT_DESKTOP=KDE\nXAUTHORITY=$'/tmp/odd\\npath'\nPATH=/usr/bin\n";
        assert_eq!(
            parse_graphical_env(text),
            [
                ("DISPLAY".to_string(), ":1".to_string()),
                ("WAYLAND_DISPLAY".to_string(), "wayland-1".to_string()),
                ("XDG_CURRENT_DESKTOP".to_string(), "KDE".to_string()),
            ]
        );
    }

    #[test]
    fn a_failed_recording_note_says_what_was_kept() {
        let note = recording_failed_note("s", Some("Firefox"), "No space left on device", true);
        assert_eq!(note.title, "Recording stopped: Firefox");
        assert_eq!(
            note.body,
            "No space left on device\nWhat was recorded is kept."
        );
        assert_eq!(note.default, Some(Click::OpenFolder("s".into())));
        let gone = recording_failed_note("s", None, "pw-record exited", false);
        assert_eq!(gone.title, "Recording stopped");
        assert_eq!(gone.default, None);
    }

    #[test]
    fn errors_are_cut_to_one_line() {
        assert_eq!(cut("a\n  b\tc", 10), "a b c");
        assert_eq!(cut("abcdef", 3), "abc…");
    }

    #[test]
    fn a_renamed_session_is_found_by_its_meta_id() {
        let (tmp, session, _meta) = session_with(&[(1, 0, 60_000, "One.\n")]);
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let renamed = store.sessions_dir().join("Standup with Jordan");
        std::fs::rename(&session.dir, &renamed).unwrap();
        let found = find_session(&store, &session.id).unwrap();
        assert_eq!(found.dir, renamed);
        assert!(find_session(&store, "2020-01-01T000000").is_err());
    }
}
