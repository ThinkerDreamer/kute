//! Desktop notifications from the daemon: a popup when new summaries are written, which
//! opens the session's `summary.md` when clicked.

use crate::config::NotifyConfig;
use crate::render;
use crate::store::{Meta, Session};
use std::path::PathBuf;

/// One notification: what it says and the file it opens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub title: String,
    pub body: String,
    pub open: PathBuf,
}

/// Sends notes somewhere. `Desktop` in the daemon, `Silent` by default, a recorder in tests.
pub trait Notifier: Send {
    fn notify(&self, note: Note);
}

/// Drops every note.
pub struct Silent;

impl Notifier for Silent {
    fn notify(&self, _: Note) {}
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
    Some(Note {
        title,
        body,
        open: session.summary_path(),
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
    if line.chars().count() <= FIRST_LINE_CHARS {
        return Some(line.to_string());
    }
    let cut: String = line.chars().take(FIRST_LINE_CHARS).collect();
    Some(format!("{}…", cut.trim_end()))
}

/// Notification bodies are markup on KDE (and in the freedesktop spec).
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Real desktop notifications through the freedesktop notification service. Clicking one
/// runs `notify.open` on the note's file.
pub struct Desktop {
    open: Vec<String>,
}

impl Desktop {
    pub fn new(config: &NotifyConfig) -> Desktop {
        Desktop {
            open: config.open.clone(),
        }
    }
}

impl Notifier for Desktop {
    fn notify(&self, note: Note) {
        let open = self.open.clone();
        // Waiting for a click blocks, so each note gets a thread that ends when the
        // notification is clicked, dismissed or expires.
        std::thread::spawn(move || {
            let shown = notify_rust::Notification::new()
                .appname("kute")
                .summary(&note.title)
                .body(&note.body)
                .icon("audio-input-microphone")
                .action("default", "Open summary")
                .action("open", "Open summary")
                .show();
            let handle = match shown {
                Ok(handle) => handle,
                Err(e) => {
                    eprintln!("kute: could not show a notification: {e}");
                    return;
                }
            };
            handle.wait_for_action(|action| {
                if (action == "default" || action == "open")
                    && let Some((program, args)) = open.split_first()
                {
                    match std::process::Command::new(program)
                        .args(args)
                        .arg(&note.open)
                        .spawn()
                    {
                        Ok(mut child) => {
                            let _ = child.wait();
                        }
                        Err(e) => eprintln!("kute: could not run {program}: {e}"),
                    }
                }
            });
        });
    }
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
        assert_eq!(note.open, session.summary_path());
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
}
