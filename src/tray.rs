//! The tray icon (a StatusNotifierItem, through `ksni`): shows whether kute is listening,
//! recording or paused and what it is working on, and offers Record / Stop / Discard,
//! Detect calls on and off, and the recordings folder.
//!
//! The daemon sends it each new `Status`; its menu sends `TrayAction`s back.

use crate::detect::Phase;
use crate::ipc::Status;
use ksni::blocking::TrayMethods;
use ksni::menu::{CheckmarkItem, StandardItem};
use ksni::{MenuItem, ToolTip};
use std::path::PathBuf;
use std::sync::mpsc::Sender;

/// What the tray's menu asks the daemon to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayAction {
    Record,
    /// Stop the session the menu was shown for, if it is still the one recording.
    Stop(String),
    /// Stop and delete the session the menu was shown for, if it is still recording.
    Discard(String),
    Pause,
    Resume,
}

/// What the tray shows for a status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub icon: &'static str,
    /// The first line of the tooltip and the menu.
    pub headline: String,
    /// Further lines: the work in progress.
    pub details: Vec<String>,
    /// The session being recorded, which Stop and Discard act on.
    pub recording: Option<String>,
    pub paused: bool,
}

/// Breeze and Adwaita both have these; `media-record` is a red dot.
const ICON_LISTENING: &str = "audio-input-microphone";
const ICON_RECORDING: &str = "media-record";
const ICON_PAUSED: &str = "media-playback-pause";

/// Until the daemon's first status arrives.
fn starting() -> View {
    View {
        icon: ICON_LISTENING,
        headline: "kute is starting".into(),
        details: Vec::new(),
        recording: None,
        paused: false,
    }
}

pub fn view(status: &Status) -> View {
    // A Tentative session is not shown (spec section 5): it may be a mic check.
    let recording = matches!(status.phase, Phase::Recording | Phase::Grace);
    let (icon, headline) = if recording {
        let headline = match &status.app {
            Some(app) => format!("Recording: {app}"),
            None => "Recording".into(),
        };
        (ICON_RECORDING, headline)
    } else if status.paused {
        (ICON_PAUSED, "Paused: not detecting calls".into())
    } else {
        (ICON_LISTENING, "Listening for calls".into())
    };
    let mut details = Vec::new();
    if status.phase == Phase::Grace {
        details.push("The call seems over; stopping soon".into());
    }
    if let Some(job) = &status.running_job {
        details.push(format!("Working: {job}"));
    }
    if status.queued > 0 {
        details.push(format!("{} more queued", status.queued));
    }
    if status.retrying > 0 {
        details.push(format!("{} waiting to retry", status.retrying));
    }
    View {
        icon,
        headline,
        details,
        recording: if recording {
            status.session.clone()
        } else {
            None
        },
        paused: status.paused,
    }
}

pub struct KuteTray {
    view: View,
    actions: Sender<TrayAction>,
    sessions_dir: PathBuf,
    open: Vec<String>,
}

impl KuteTray {
    fn send(&self, action: TrayAction) {
        // The daemon is shutting down if this fails; nothing to do about it here.
        let _ = self.actions.send(action);
    }
}

impl ksni::Tray for KuteTray {
    // Left click opens the menu: there is no window to show.
    const MENU_ON_ACTIVATE: bool = true;

    fn id(&self) -> String {
        "kute".into()
    }

    fn title(&self) -> String {
        "kute".into()
    }

    fn icon_name(&self) -> String {
        self.view.icon.into()
    }

    fn status(&self) -> ksni::Status {
        if self.view.recording.is_some() {
            ksni::Status::Active
        } else {
            // Plasma tucks a passive item into the overflow; recording brings it out.
            ksni::Status::Passive
        }
    }

    fn tool_tip(&self) -> ToolTip {
        ToolTip {
            title: self.view.headline.clone(),
            description: self.view.details.join("\n"),
            ..Default::default()
        }
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        let label = |text: &str| -> MenuItem<Self> {
            StandardItem {
                label: escape_label(text),
                enabled: false,
                ..Default::default()
            }
            .into()
        };
        let mut items = vec![label(&self.view.headline)];
        items.extend(self.view.details.iter().map(|d| label(d)));
        items.push(MenuItem::Separator);
        match &self.view.recording {
            Some(id) => {
                let (stop, discard) = (id.clone(), id.clone());
                items.push(
                    StandardItem {
                        label: "Stop and keep".into(),
                        icon_name: "media-playback-stop".into(),
                        activate: Box::new(move |t: &mut Self| {
                            t.send(TrayAction::Stop(stop.clone()))
                        }),
                        ..Default::default()
                    }
                    .into(),
                );
                items.push(
                    StandardItem {
                        label: "Stop and delete".into(),
                        icon_name: "edit-delete".into(),
                        activate: Box::new(move |t: &mut Self| {
                            t.send(TrayAction::Discard(discard.clone()))
                        }),
                        ..Default::default()
                    }
                    .into(),
                );
            }
            None => items.push(
                StandardItem {
                    label: "Record now".into(),
                    icon_name: ICON_RECORDING.into(),
                    activate: Box::new(|t: &mut Self| t.send(TrayAction::Record)),
                    ..Default::default()
                }
                .into(),
            ),
        }
        items.push(
            CheckmarkItem {
                label: "Detect calls".into(),
                checked: !self.view.paused,
                activate: Box::new(|t: &mut Self| {
                    t.send(if t.view.paused {
                        TrayAction::Resume
                    } else {
                        TrayAction::Pause
                    })
                }),
                ..Default::default()
            }
            .into(),
        );
        items.push(MenuItem::Separator);
        items.push(
            StandardItem {
                label: "Open recordings folder".into(),
                icon_name: "folder-open".into(),
                activate: Box::new(|t: &mut Self| {
                    let (open, dir) = (t.open.clone(), t.sessions_dir.clone());
                    // Opening waits for the program; the menu must not.
                    std::thread::spawn(move || crate::notify::open_with(&open, &dir));
                }),
                ..Default::default()
            }
            .into(),
        );
        items
    }

    fn watcher_offline(&self, _reason: ksni::OfflineReason) -> bool {
        // The panel restarting, or not up yet at login: keep the item, and it registers
        // again when the panel is back.
        eprintln!("kute: no tray host to show the icon; it appears when the panel starts");
        true
    }

    fn watcher_online(&self) {
        eprintln!("kute: tray icon shown");
    }
}

/// Menu labels treat `_` as an access-key marker; `__` shows one underscore.
fn escape_label(text: &str) -> String {
    text.replace('_', "__")
}

/// The running tray. Dropping it leaves the icon up until the daemon exits.
pub struct Tray {
    handle: ksni::blocking::Handle<KuteTray>,
}

impl Tray {
    /// Puts the icon up. The daemon may start before the desktop (systemd's
    /// `default.target`), so a missing tray host is not an error: the icon appears when one
    /// starts. Fails only without a D-Bus session bus.
    pub fn spawn(
        actions: Sender<TrayAction>,
        sessions_dir: PathBuf,
        open: Vec<String>,
    ) -> anyhow::Result<Tray> {
        let tray = KuteTray {
            view: starting(),
            actions,
            sessions_dir,
            open,
        };
        let handle = tray
            .assume_sni_available(true)
            .spawn()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Tray { handle })
    }

    pub fn show(&self, status: &Status) {
        let view = view(status);
        self.handle.update(|t| t.view = view);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ksni::Tray as _;

    fn status(phase: Phase) -> Status {
        Status {
            phase,
            paused: false,
            session: Some("2026-11-07T101500".into()),
            app: Some("Jitsi Meet".into()),
            queued: 0,
            retrying: 0,
            running_job: None,
        }
    }

    #[test]
    fn recording_shows_the_app_and_a_red_icon() {
        let v = view(&status(Phase::Recording));
        assert_eq!(v.icon, ICON_RECORDING);
        assert_eq!(v.headline, "Recording: Jitsi Meet");
        assert_eq!(v.recording.as_deref(), Some("2026-11-07T101500"));
    }

    #[test]
    fn a_manual_recording_has_no_app() {
        let mut s = status(Phase::Recording);
        s.app = None;
        assert_eq!(view(&s).headline, "Recording");
    }

    #[test]
    fn a_tentative_session_looks_like_listening() {
        let v = view(&status(Phase::Tentative));
        assert_eq!(v.icon, ICON_LISTENING);
        assert_eq!(v.recording, None);
    }

    #[test]
    fn grace_is_still_recording_and_says_so() {
        let v = view(&status(Phase::Grace));
        assert_eq!(v.icon, ICON_RECORDING);
        assert_eq!(v.details, ["The call seems over; stopping soon"]);
    }

    #[test]
    fn paused_and_working() {
        let mut s = status(Phase::Idle);
        s.paused = true;
        s.running_job = Some("process 2026-11-06T090000".into());
        s.queued = 2;
        s.retrying = 1;
        let v = view(&s);
        assert_eq!(v.icon, ICON_PAUSED);
        assert_eq!(
            v.details,
            [
                "Working: process 2026-11-06T090000",
                "2 more queued",
                "1 waiting to retry"
            ]
        );
    }

    fn tray(status: &Status) -> (KuteTray, std::sync::mpsc::Receiver<TrayAction>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let tray = KuteTray {
            view: view(status),
            actions: tx,
            sessions_dir: PathBuf::new(),
            open: Vec::new(),
        };
        (tray, rx)
    }

    /// Activates the menu item with this label, as a click would.
    fn click(tray: &mut KuteTray, label: &str) {
        let items = tray.menu();
        for item in items {
            match item {
                MenuItem::Standard(s) if s.label == label => return (s.activate)(tray),
                MenuItem::Checkmark(c) if c.label == label => return (c.activate)(tray),
                _ => {}
            }
        }
        panic!("no menu item {label:?}");
    }

    #[test]
    fn the_menu_stops_the_session_it_was_shown_for() {
        let (mut t, rx) = tray(&status(Phase::Recording));
        click(&mut t, "Stop and keep");
        click(&mut t, "Stop and delete");
        assert_eq!(
            rx.try_iter().collect::<Vec<_>>(),
            [
                TrayAction::Stop("2026-11-07T101500".into()),
                TrayAction::Discard("2026-11-07T101500".into())
            ]
        );
    }

    #[test]
    fn the_menu_records_and_toggles_detection() {
        let (mut t, rx) = tray(&status(Phase::Idle));
        click(&mut t, "Record now");
        click(&mut t, "Detect calls");
        let mut paused = status(Phase::Idle);
        paused.paused = true;
        t.view = view(&paused);
        click(&mut t, "Detect calls");
        assert_eq!(
            rx.try_iter().collect::<Vec<_>>(),
            [TrayAction::Record, TrayAction::Pause, TrayAction::Resume]
        );
    }

    #[test]
    fn underscores_in_labels_are_shown() {
        assert_eq!(escape_label("Recording: my_app"), "Recording: my__app");
    }
}
