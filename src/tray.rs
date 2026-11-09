//! The tray icon (a StatusNotifierItem, through `ksni`): shows whether kute is listening,
//! recording or paused and what it is working on, and offers Record / Stop / Discard,
//! the call's details (title, who else is on it, languages, notes; through `dialog`),
//! Detect calls on and off, and the recordings folder.
//!
//! The daemon sends it each new `Status`; its menu sends `TrayAction`s back.

use crate::detect::Phase;
use crate::dialog::{Ask, Dialogs};
use crate::ipc::Status;
use crate::jobs::Job;
use crate::store::{LanguageChange, Meta, Store, Track};
use crate::transcribe::parse_languages;
use anyhow::Result;
use ksni::blocking::TrayMethods;
use ksni::menu::{CheckmarkItem, StandardItem};
use ksni::{MenuItem, ToolTip};
use std::sync::mpsc::Sender;

/// What the tray's menu asks the daemon to do.
#[derive(Debug, Clone, PartialEq)]
pub enum TrayAction {
    Record,
    /// Stop the session the menu was shown for, if it is still the one recording.
    Stop(String),
    /// Stop and delete the session the menu was shown for, if it is still recording.
    Discard(String),
    Pause,
    Resume,
    /// Change a session's details (title, others' name, languages, notes).
    Edit(Job),
}

/// A detail of the call the menu lets you type in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detail {
    Title,
    Others,
    Languages,
    Notes,
}

impl Detail {
    const ALL: [Detail; 4] = [
        Detail::Title,
        Detail::Others,
        Detail::Languages,
        Detail::Notes,
    ];

    fn menu_label(self) -> &'static str {
        match self {
            Detail::Title => "Title…",
            Detail::Others => "Who else is on the call…",
            Detail::Languages => "Languages…",
            Detail::Notes => "Notes for the summary…",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            Detail::Title => "edit-rename",
            Detail::Others => "system-users",
            Detail::Languages => "preferences-desktop-locale",
            Detail::Notes => "document-edit",
        }
    }

    fn prompt(self) -> &'static str {
        match self {
            Detail::Title => "The call's title:",
            Detail::Others => {
                "Who else is on the call, e.g. Jordan or Jordan, Sam.\n\
                 The transcript labels their track with it. Empty: the usual label."
            }
            Detail::Languages => {
                "Languages spoken, e.g. ru, or ru,en to pick between them; auto for any.\n\
                 Per track, yours then theirs: ru / ru,en. Empty: from the config."
            }
            Detail::Notes => {
                "Notes the summary follows: who the speakers are, what to focus on, tone."
            }
        }
    }

    /// What the dialog shows at first.
    pub fn current(self, meta: &Meta) -> String {
        match self {
            Detail::Title => meta.title.clone().unwrap_or_default(),
            Detail::Others => meta.others_name.clone().unwrap_or_default(),
            Detail::Notes => meta.notes.clone().unwrap_or_default(),
            Detail::Languages => {
                let me = meta.transcribe.for_track(Track::Me);
                let others = meta.transcribe.for_track(Track::Others);
                if me == others {
                    me.join(",")
                } else {
                    let show = |codes: &[String]| match codes {
                        [] => "default".to_string(),
                        codes => codes.join(","),
                    };
                    format!("{} / {}", show(me), show(others))
                }
            }
        }
    }

    /// The job that sets what was typed; `None` when there is nothing to change (an empty
    /// title: a session can't lose its title).
    pub fn job(self, id: &str, text: &str) -> Result<Option<Job>> {
        let id = id.to_string();
        let text = text.trim();
        let optional = (!text.is_empty()).then(|| text.to_string());
        Ok(Some(match self {
            Detail::Title => match optional {
                Some(title) => Job::Title { id, title },
                None => return Ok(None),
            },
            Detail::Others => Job::Others { id, name: optional },
            Detail::Notes => Job::Notes {
                id,
                notes: optional,
            },
            Detail::Languages => Job::Language {
                id,
                languages: language_change(text)?,
            },
        }))
    }
}

/// `ru,en` for both tracks, `ru / ru,en` for mine and theirs (`default` or nothing: the
/// config's), nothing at all for the config's on both.
fn language_change(text: &str) -> Result<LanguageChange> {
    let parse = |part: &str| match part.trim() {
        "" | "default" => Ok(Vec::new()),
        codes => parse_languages(codes),
    };
    Ok(match text.split_once('/') {
        Some((me, others)) => LanguageChange {
            all: Some(Vec::new()),
            me: Some(parse(me)?),
            others: Some(parse(others)?),
        },
        None => LanguageChange {
            all: Some(parse(text)?),
            ..LanguageChange::default()
        },
    })
}

/// What the tray shows for a status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub icon: Icon,
    /// The first line of the tooltip and the menu.
    pub headline: String,
    /// Further lines: the work in progress.
    pub details: Vec<String>,
    /// The session being recorded, which Stop and Discard act on.
    pub recording: Option<String>,
    pub paused: bool,
}

/// The tray icon's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    Listening,
    Recording,
    Paused,
}

impl Icon {
    const ALL: [Icon; 3] = [Icon::Listening, Icon::Recording, Icon::Paused];

    /// kute's own icon: an ear and a mic ("kute" is to listen, and the ear, in Toki Pona).
    /// Symbolic, so the panel recolours it to its theme; the recording mic stays red. Every
    /// shape is filled, not stroked: GTK recolours a symbolic icon by filling each shape, so
    /// a stroked line would come out as a blob. Plasma reads the `ColorScheme-*` classes,
    /// GTK the `error` class.
    fn own(self) -> (&'static str, &'static str) {
        match self {
            Icon::Listening => (
                "kute-listening-symbolic",
                include_str!("../assets/icons/kute-listening-symbolic.svg"),
            ),
            Icon::Recording => (
                "kute-recording-symbolic",
                include_str!("../assets/icons/kute-recording-symbolic.svg"),
            ),
            Icon::Paused => (
                "kute-paused-symbolic",
                include_str!("../assets/icons/kute-paused-symbolic.svg"),
            ),
        }
    }

    /// A stand-in from the desktop's theme, for when kute's own could not be written out.
    /// Breeze and Adwaita both have these; `media-record` is a red dot.
    fn themed(self) -> &'static str {
        match self {
            Icon::Listening => "audio-input-microphone",
            Icon::Recording => "media-record",
            Icon::Paused => "media-playback-pause",
        }
    }
}

/// Writes kute's icons to `dir`, which the tray then names as its icon theme path, so the
/// tray host finds them whether kute came from cargo, Nix or a distribution package.
/// Plasma takes the icon from the top of that folder (and wants the path to end in
/// `icons`); GNOME's AppIndicator extension adds it to GTK's icon search path.
fn write_icons(dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for icon in Icon::ALL {
        let (name, svg) = icon.own();
        let path = dir.join(format!("{name}.svg"));
        // Rewriting an unchanged file would only make the host reload it.
        if std::fs::read_to_string(&path).ok().as_deref() != Some(svg) {
            std::fs::write(&path, svg)?;
        }
    }
    Ok(())
}

/// Where the user's own hicolor icons go: GTK finds an icon there by name, from any theme.
fn hicolor_icons(data: &std::path::Path) -> std::path::PathBuf {
    data.join("icons/hicolor/scalable/apps")
}

/// `~/.local/share/kute/icons`, once kute's icons are written there.
///
/// They go into the user's hicolor theme as well. Cinnamon before 6.8 draws nothing for a
/// symbolic icon that is only in the tray's icon theme path: its tray host hands the panel
/// the file's path, which the panel then looks up as an icon name. One the theme has is
/// passed by name instead, and drawn.
fn own_icons() -> Option<std::path::PathBuf> {
    let data = dirs::data_dir()?;
    let dir = data.join("kute").join("icons");
    write_icons(&dir)
        .inspect_err(|e| eprintln!("kute: using the theme's tray icons: {e:#}"))
        .ok()?;
    let hicolor = hicolor_icons(&data);
    if let Err(e) = write_icons(&hicolor).and_then(|()| outdate_icon_cache(&data)) {
        eprintln!(
            "kute: can't add the tray icons to {}: {e:#}",
            hicolor.display()
        );
    }
    Some(dir)
}

/// Marks the user's hicolor icon cache, if there is one, as older than its folder.
///
/// GTK trusts `icon-theme.cache` while it is newer than the theme's folder, and finds only
/// the icons the cache lists. Writing into `scalable/apps` leaves the theme folder's time as
/// it was, so a cache made before kute's icons hides them. Touching the folder makes GTK
/// read the files instead, and tells a running GTK to look again. The cache itself is left
/// alone: it is not kute's.
fn outdate_icon_cache(data: &std::path::Path) -> Result<()> {
    let theme = data.join("icons/hicolor");
    let Ok(cache) = std::fs::metadata(theme.join("icon-theme.cache")) else {
        return Ok(());
    };
    // GTK compares whole seconds, and trusts a cache from the same second as the folder.
    let secs = |t: std::time::SystemTime| {
        t.duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    };
    let cache = cache.modified()?;
    if secs(cache) < secs(std::fs::metadata(&theme)?.modified()?) {
        return Ok(());
    }
    // Past the cache's second even when the cache claims to be from the future.
    let past_cache = std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs(cache) + 1);
    let time = std::time::SystemTime::now().max(past_cache);
    std::fs::File::open(&theme)?.set_modified(time)?;
    Ok(())
}

/// Until the daemon's first status arrives.
fn starting() -> View {
    View {
        icon: Icon::Listening,
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
        (Icon::Recording, headline)
    } else if status.paused {
        (Icon::Paused, "Paused: not detecting calls".into())
    } else {
        (Icon::Listening, "Listening for calls".into())
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
    store: Store,
    open: Vec<String>,
    /// Where to look for kdialog or zenity; `None`: `PATH`. Looked up each time the menu
    /// opens, so one installed while kute runs is used straight away.
    dialog_path: Option<std::ffi::OsString>,
    /// Mark the icon passive while idle. Plasma tucks a passive item into the overflow;
    /// other hosts, such as Cinnamon's, hide it outright, so there it stays active.
    passive_when_idle: bool,
    /// Where kute's own icons are; `None`: use the theme's.
    icons: Option<std::path::PathBuf>,
}

/// Whether the desktop tucks a passive tray item away instead of hiding it: only Plasma, as
/// far as we know. `XDG_CURRENT_DESKTOP` is a colon-separated list, e.g. `KDE` or `X-Cinnamon`.
fn passive_when_idle(current_desktop: Option<&str>) -> bool {
    current_desktop.is_some_and(|d| d.split(':').any(|d| d.eq_ignore_ascii_case("KDE")))
}

impl KuteTray {
    fn send(&self, action: TrayAction) {
        // The daemon is shutting down if this fails; nothing to do about it here.
        let _ = self.actions.send(action);
    }

    fn dialogs(&self) -> Option<Dialogs> {
        match &self.dialog_path {
            Some(path) => Dialogs::find_in(path),
            None => Dialogs::find(),
        }
    }

    /// Asks for one of the session's details in a dialog, on a thread of its own: the menu
    /// must not wait for it. The answer goes to the daemon even if the call has ended by
    /// then; it is the same session.
    fn edit(&self, id: String, detail: Detail) {
        let Some(dialogs) = self.dialogs() else {
            return;
        };
        let (store, actions) = (self.store.clone(), self.actions.clone());
        std::thread::spawn(move || {
            let meta = match store.resolve(&id).and_then(|s| s.load_meta()) {
                Ok(meta) => meta,
                Err(e) => return eprintln!("kute: can't edit {id}: {e:#}"),
            };
            let label = detail.menu_label().trim_end_matches('…');
            let title = format!("kute: {label}");
            let current = detail.current(&meta);
            let ask = Ask {
                title: &title,
                prompt: detail.prompt(),
                value: &current,
                multiline: detail == Detail::Notes,
            };
            let text = match dialogs.ask(&ask) {
                Ok(Some(text)) => text,
                Ok(None) => return,
                Err(e) => return eprintln!("kute: {e:#}"),
            };
            if text.trim() == current.trim() {
                return;
            }
            match detail.job(&id, &text) {
                Ok(Some(job)) => {
                    let _ = actions.send(TrayAction::Edit(job));
                }
                Ok(None) => {}
                Err(e) => dialogs.error(&title, &format!("{e:#}")),
            }
        });
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
        match self.icons {
            Some(_) => self.view.icon.own().0,
            None => self.view.icon.themed(),
        }
        .into()
    }

    fn icon_theme_path(&self) -> String {
        self.icons
            .as_ref()
            .map(|d| d.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    fn status(&self) -> ksni::Status {
        if self.view.recording.is_some() || !self.passive_when_idle {
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
                items.push(MenuItem::Separator);
                if self.dialogs().is_some() {
                    for detail in Detail::ALL {
                        let id = id.clone();
                        items.push(
                            StandardItem {
                                label: detail.menu_label().into(),
                                icon_name: detail.icon().into(),
                                activate: Box::new(move |t: &mut Self| t.edit(id.clone(), detail)),
                                ..Default::default()
                            }
                            .into(),
                        );
                    }
                } else {
                    items.push(label(
                        "Install kdialog or zenity to type in the call's details",
                    ));
                }
            }
            None => items.push(
                StandardItem {
                    label: "Record now".into(),
                    icon_name: Icon::Recording.themed().into(),
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
                    let (open, dir) = (t.open.clone(), t.store.sessions_dir());
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
        store: Store,
        open: Vec<String>,
    ) -> anyhow::Result<Tray> {
        let tray = KuteTray {
            view: starting(),
            actions,
            store,
            open,
            dialog_path: None,
            passive_when_idle: passive_when_idle(
                std::env::var("XDG_CURRENT_DESKTOP").ok().as_deref(),
            ),
            icons: own_icons(),
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
        assert_eq!(v.icon, Icon::Recording);
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
        assert_eq!(v.icon, Icon::Listening);
        assert_eq!(v.recording, None);
    }

    #[test]
    fn grace_is_still_recording_and_says_so() {
        let v = view(&status(Phase::Grace));
        assert_eq!(v.icon, Icon::Recording);
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
        assert_eq!(v.icon, Icon::Paused);
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
            store: Store::open(std::env::temp_dir().join("kute-tray-unused")).unwrap(),
            open: Vec::new(),
            dialog_path: Some(std::ffi::OsString::new()),
            passive_when_idle: true,
            icons: None,
        };
        (tray, rx)
    }

    #[test]
    fn only_plasma_gets_a_passive_icon_when_idle() {
        assert!(passive_when_idle(Some("KDE")));
        assert!(passive_when_idle(Some("ubuntu:KDE")));
        assert!(!passive_when_idle(Some("X-Cinnamon")));
        assert!(!passive_when_idle(Some("ubuntu:GNOME")));
        assert!(!passive_when_idle(None));
    }

    #[test]
    fn an_idle_icon_stays_active_where_passive_would_hide_it() {
        let (mut tray, _rx) = tray(&status(Phase::Idle));
        assert!(matches!(tray.status(), ksni::Status::Passive));
        tray.passive_when_idle = false;
        assert!(matches!(tray.status(), ksni::Status::Active));
    }

    #[test]
    fn a_stale_icon_cache_is_marked_outdated() {
        use std::time::{Duration, SystemTime};
        let set = |p: &std::path::Path, t: SystemTime| {
            std::fs::File::open(p).unwrap().set_modified(t).unwrap()
        };
        let modified = |p: &std::path::Path| std::fs::metadata(p).unwrap().modified().unwrap();
        let data = tempfile::tempdir().unwrap();
        // No theme folder, or no cache in it: nothing to do.
        outdate_icon_cache(data.path()).unwrap();
        let theme = data.path().join("icons/hicolor");
        write_icons(&hicolor_icons(data.path())).unwrap();
        outdate_icon_cache(data.path()).unwrap();

        // A cache newer than the folder, as after writing into scalable/apps.
        let cache = theme.join("icon-theme.cache");
        std::fs::write(&cache, b"").unwrap();
        let now = SystemTime::now();
        set(&cache, now - Duration::from_secs(60));
        set(&theme, now - Duration::from_secs(120));
        outdate_icon_cache(data.path()).unwrap();
        assert!(modified(&theme) > modified(&cache));
        assert!(cache.exists(), "the cache is not kute's to remove");

        // A cache from the future, or from the folder's own second, is still outdated.
        let ahead = now + Duration::from_secs(3600);
        set(&cache, ahead);
        set(&theme, ahead);
        outdate_icon_cache(data.path()).unwrap();
        let whole = |t: SystemTime| t.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
        assert!(whole(modified(&theme)) > whole(modified(&cache)));
    }

    #[test]
    fn writes_its_icons_and_names_them_for_the_tray_host() {
        let dir = tempfile::tempdir().unwrap();
        let icons = dir.path().join("kute/icons");
        write_icons(&icons).unwrap();
        for icon in Icon::ALL {
            let (name, svg) = icon.own();
            let written = std::fs::read_to_string(icons.join(format!("{name}.svg"))).unwrap();
            assert_eq!(written, svg);
            // Plasma recolours symbolic icons through this stylesheet.
            assert!(svg.contains("current-color-scheme"), "{name}");
            // GTK fills every shape when it recolours one, so a stroke would turn into a blob.
            assert!(!svg.contains("stroke"), "{name}");
        }
        // Again over the same files: nothing to change, and no error.
        write_icons(&icons).unwrap();

        let (mut t, _rx) = tray(&status(Phase::Recording));
        assert_eq!(t.icon_name(), "media-record");
        assert_eq!(t.icon_theme_path(), "");
        t.icons = Some(icons.clone());
        assert_eq!(t.icon_name(), "kute-recording-symbolic");
        assert_eq!(t.icon_theme_path(), icons.to_string_lossy());
    }

    fn meta() -> Meta {
        Meta::new(
            "2026-11-07T101500".into(),
            "2026-11-07T10:15:00+01:00[+01:00]".parse().unwrap(),
            crate::store::Trigger {
                kind: crate::store::TriggerKind::Manual,
                app_name: None,
                binary: None,
            },
            "auto".into(),
        )
    }

    #[test]
    fn languages_show_one_list_or_mine_and_theirs() {
        let mut m = meta();
        assert_eq!(Detail::Languages.current(&m), "");
        m.transcribe.languages = vec!["ru".into(), "en".into()];
        assert_eq!(Detail::Languages.current(&m), "ru,en");
        m.transcribe.me = vec!["ru".into()];
        assert_eq!(Detail::Languages.current(&m), "ru / ru,en");
        m.transcribe.languages.clear();
        assert_eq!(Detail::Languages.current(&m), "ru / default");
    }

    #[test]
    fn typed_languages_become_a_change() {
        let change = |text: &str| match Detail::Languages.job("s", text).unwrap() {
            Some(Job::Language { languages, .. }) => languages,
            other => panic!("{other:?}"),
        };
        let codes = |c: &[&str]| Some(c.iter().map(|c| c.to_string()).collect::<Vec<_>>());
        assert_eq!(
            change("ru, en"),
            LanguageChange {
                all: codes(&["ru", "en"]),
                ..Default::default()
            }
        );
        assert_eq!(
            change("ru / default"),
            LanguageChange {
                all: codes(&[]),
                me: codes(&["ru"]),
                others: codes(&[]),
            }
        );
        // Round trip: what `current` shows sets the same languages again.
        let mut m = meta();
        change("ru / ru,en").apply(&mut m.transcribe);
        assert_eq!(Detail::Languages.current(&m), "ru / ru,en");
        assert_eq!(
            change(""),
            LanguageChange {
                all: codes(&[]),
                ..Default::default()
            }
        );
        assert!(Detail::Languages.job("s", "klingon").is_err());
    }

    #[test]
    fn an_empty_answer_clears_all_but_the_title() {
        assert_eq!(Detail::Title.job("s", "  ").unwrap(), None);
        assert_eq!(
            Detail::Title.job("s", " Standup ").unwrap(),
            Some(Job::Title {
                id: "s".into(),
                title: "Standup".into()
            })
        );
        assert_eq!(
            Detail::Others.job("s", "").unwrap(),
            Some(Job::Others {
                id: "s".into(),
                name: None
            })
        );
        assert_eq!(
            Detail::Notes.job("s", "").unwrap(),
            Some(Job::Notes {
                id: "s".into(),
                notes: None
            })
        );
    }

    #[test]
    fn the_menu_asks_for_a_detail_and_sends_the_edit() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let mut m = meta();
        let session = store.create_session(&m.started_at).unwrap();
        m.id = session.id.clone();
        m.others_name = Some("Jordan".into());
        session.save_meta(&m).unwrap();
        // A kdialog that appends to the name it was shown.
        let bin = tmp.path().join("bin");
        std::fs::create_dir(&bin).unwrap();
        let kdialog = bin.join("kdialog");
        std::fs::write(
            &kdialog,
            "#!/bin/sh\nfor a; do last=\"$a\"; done; echo \"$last, Sam\"\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&kdialog, std::fs::Permissions::from_mode(0o755)).unwrap();

        let mut s = status(Phase::Recording);
        s.session = Some(session.id.clone());
        let (mut t, rx) = tray(&s);
        t.store = store;
        let hint = "Install kdialog or zenity to type in the call's details";
        assert!(
            t.menu()
                .iter()
                .any(|i| matches!(i, MenuItem::Standard(s) if s.label == hint))
        );
        t.dialog_path = Some(bin.clone().into_os_string());
        click(&mut t, "Who else is on the call…");
        let got = rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
        assert_eq!(
            got,
            TrayAction::Edit(Job::Others {
                id: session.id.clone(),
                name: Some("Jordan, Sam".into())
            })
        );
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
