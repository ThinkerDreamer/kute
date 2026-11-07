//! Small dialogs that ask for a line or a few lines of text, for the tray: a tray menu can't
//! take text. They run `kdialog` (Plasma) or `zenity` (GNOME and others), whichever is
//! installed, so kute needs no toolkit of its own.

use anyhow::{Context, Result, bail};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The dialog program kute found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dialogs {
    kind: Kind,
    program: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Kdialog,
    Zenity,
}

/// What to ask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ask<'a> {
    pub title: &'a str,
    pub prompt: &'a str,
    /// What the field holds at first.
    pub value: &'a str,
    /// Several lines (notes) instead of one.
    pub multiline: bool,
}

impl Dialogs {
    /// kdialog first, then zenity, on `PATH`; `None` without either.
    pub fn find() -> Option<Dialogs> {
        let path = std::env::var_os("PATH")?;
        Self::find_in(&path)
    }

    pub(crate) fn find_in(path: &std::ffi::OsStr) -> Option<Dialogs> {
        [("kdialog", Kind::Kdialog), ("zenity", Kind::Zenity)]
            .into_iter()
            .find_map(|(name, kind)| {
                std::env::split_paths(path)
                    .filter(|dir| dir.is_absolute())
                    .map(|dir| dir.join(name))
                    .find(|p| is_executable(p))
                    .map(|program| Dialogs { kind, program })
            })
    }

    /// Shows the dialog and waits. `None` when it was cancelled.
    pub fn ask(&self, ask: &Ask) -> Result<Option<String>> {
        let (args, stdin) = self.ask_args(ask);
        let output = self
            .command(&args)
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                if let (Some(text), Some(mut pipe)) = (&stdin, child.stdin.take()) {
                    pipe.write_all(text.as_bytes())?;
                }
                child.wait_with_output()
            })
            .with_context(|| format!("running {}", self.program.display()))?;
        match output.status.code() {
            Some(0) => {
                let text = String::from_utf8_lossy(&output.stdout);
                Ok(Some(strip_final_newline(&text).to_string()))
            }
            // Cancel, or the window was closed.
            Some(1) | Some(-1) => Ok(None),
            _ => bail!("{} failed ({})", self.program.display(), output.status),
        }
    }

    /// Shows an error and doesn't wait for it to be closed.
    pub fn error(&self, title: &str, message: &str) {
        let args = match self.kind {
            Kind::Kdialog => vec!["--title", title, "--error", message],
            Kind::Zenity => vec!["--error", "--title", title, "--text", message],
        };
        if let Err(e) = self.command(&args).spawn() {
            eprintln!("kute: could not run {}: {e}", self.program.display());
        }
    }

    fn ask_args<'a>(&self, ask: &Ask<'a>) -> (Vec<&'a str>, Option<String>) {
        match (self.kind, ask.multiline) {
            (Kind::Kdialog, false) => (
                vec!["--title", ask.title, "--inputbox", ask.prompt, ask.value],
                None,
            ),
            (Kind::Kdialog, true) => (
                vec![
                    "--title",
                    ask.title,
                    "--textinputbox",
                    ask.prompt,
                    ask.value,
                ],
                None,
            ),
            (Kind::Zenity, false) => (
                vec![
                    "--entry",
                    "--title",
                    ask.title,
                    "--text",
                    ask.prompt,
                    "--entry-text",
                    ask.value,
                ],
                None,
            ),
            // Zenity's text box has no prompt line: the title carries it.
            (Kind::Zenity, true) => (
                vec!["--text-info", "--editable", "--title", ask.prompt],
                Some(ask.value.to_string()),
            ),
        }
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(&self.program);
        command.args(args);
        // The daemon may have started before the desktop and have no display of its own.
        for (key, value) in crate::notify::graphical_env() {
            command.env(key, value);
        }
        command
    }

    pub fn program(&self) -> &Path {
        &self.program
    }
}

/// A file someone may run: a `kdialog` without the execute bit must not hide a zenity.
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// The dialogs end what they print with one newline; the text's own stay.
fn strip_final_newline(text: &str) -> &str {
    text.strip_suffix('\n').unwrap_or(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn script(dir: &Path, name: &str, body: &str) {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn ask(multiline: bool) -> Ask<'static> {
        Ask {
            title: "kute",
            prompt: "Title",
            value: "Planning",
            multiline,
        }
    }

    #[test]
    fn kdialog_comes_before_zenity() {
        let dir = tempfile::tempdir().unwrap();
        script(dir.path(), "zenity", "");
        let d = Dialogs::find_in(dir.path().as_os_str()).unwrap();
        assert_eq!(d.kind, Kind::Zenity);
        script(dir.path(), "kdialog", "");
        let d = Dialogs::find_in(dir.path().as_os_str()).unwrap();
        assert_eq!(d.kind, Kind::Kdialog);
        // Not runnable: skipped.
        let kdialog = dir.path().join("kdialog");
        std::fs::set_permissions(&kdialog, std::fs::Permissions::from_mode(0o644)).unwrap();
        let d = Dialogs::find_in(dir.path().as_os_str()).unwrap();
        assert_eq!(d.kind, Kind::Zenity);
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(Dialogs::find_in(empty.path().as_os_str()), None);
    }

    #[test]
    fn kdialog_gets_the_value_and_answers_on_stdout() {
        let dir = tempfile::tempdir().unwrap();
        // Prints its last argument, the value, as kdialog does after an OK.
        script(
            dir.path(),
            "kdialog",
            r#"for a; do last="$a"; done; echo "$last again""#,
        );
        let d = Dialogs::find_in(dir.path().as_os_str()).unwrap();
        assert_eq!(
            d.ask(&ask(false)).unwrap().as_deref(),
            Some("Planning again")
        );
    }

    #[test]
    fn zenity_text_reads_the_value_on_stdin() {
        let dir = tempfile::tempdir().unwrap();
        script(dir.path(), "zenity", "cat; echo; echo more");
        let d = Dialogs::find_in(dir.path().as_os_str()).unwrap();
        assert_eq!(
            d.ask(&ask(true)).unwrap().as_deref(),
            Some("Planning\nmore")
        );
    }

    #[test]
    fn cancel_is_none_and_a_crash_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        script(dir.path(), "kdialog", "exit 1");
        let d = Dialogs::find_in(dir.path().as_os_str()).unwrap();
        assert_eq!(d.ask(&ask(false)).unwrap(), None);
        script(dir.path(), "kdialog", "exit 3");
        assert!(d.ask(&ask(false)).is_err());
    }
}
