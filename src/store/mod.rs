//! Session folders on disk: the source of truth for everything kute knows.

mod meta;
mod tracks;

pub use meta::*;
pub use tracks::*;

use crate::time;
use anyhow::{Context, Result, bail};
use jiff::Zoned;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Serializes `Session::update_meta` within the process, so the recorder and the worker
/// can't overwrite each other's changes to `meta.toml`.
static META_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Track {
    Me,
    Others,
}

impl Track {
    pub const BOTH: [Track; 2] = [Track::Me, Track::Others];

    pub fn as_str(self) -> &'static str {
        match self {
            Track::Me => "me",
            Track::Others => "others",
        }
    }
}

pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Opens (creating if needed) the storage root. Root, `sessions/` and `models/` are 0700.
    pub fn open(root: impl Into<PathBuf>) -> Result<Store> {
        let root = root.into();
        create_private_dir(&root)?;
        create_private_dir(&root.join("sessions"))?;
        create_private_dir(&root.join("models"))?;
        Ok(Store { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub fn index_path(&self) -> PathBuf {
        self.root.join("index.sqlite")
    }

    /// Creates an empty session folder named after `started_at`, adding `-2`, `-3`, ...
    /// if that name is taken.
    pub fn create_session(&self, started_at: &Zoned) -> Result<Session> {
        let base = time::session_id(started_at);
        for i in 1u32.. {
            let id = if i == 1 {
                base.clone()
            } else {
                format!("{base}-{i}")
            };
            let dir = self.sessions_dir().join(&id);
            match fs::create_dir(&dir) {
                Ok(()) => {
                    set_private(&dir)?;
                    return Ok(Session { id, dir });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).with_context(|| format!("creating {}", dir.display())),
            }
        }
        unreachable!("u32 range exhausted")
    }

    pub fn session(&self, id: &str) -> Result<Session> {
        let s = Session {
            id: id.to_string(),
            dir: self.sessions_dir().join(id),
        };
        if !s.meta_path().is_file() {
            bail!("no session {id:?}");
        }
        Ok(s)
    }

    /// All sessions (folders with a `meta.toml`), oldest first.
    pub fn list(&self) -> Result<Vec<Session>> {
        let mut out = Vec::new();
        let dir = self.sessions_dir();
        for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry?;
            let Some(id) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let s = Session {
                id,
                dir: entry.path(),
            };
            if s.meta_path().is_file() {
                out.push(s);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// Finds a session by full ID, unique prefix, or `last`.
    pub fn resolve(&self, query: &str) -> Result<Session> {
        let all = self.list()?;
        if query == "last" {
            return all.into_iter().last().context("there are no sessions yet");
        }
        if let Some(s) = all.iter().find(|s| s.id == query) {
            return Ok(s.clone());
        }
        let mut matches: Vec<Session> = all
            .into_iter()
            .filter(|s| s.id.starts_with(query))
            .collect();
        match matches.len() {
            0 => bail!("no session matches {query:?}"),
            1 => Ok(matches.remove(0)),
            n => bail!("{query:?} matches {n} sessions; use a longer prefix"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub dir: PathBuf,
}

impl Session {
    pub fn meta_path(&self) -> PathBuf {
        self.dir.join("meta.toml")
    }

    pub fn load_meta(&self) -> Result<Meta> {
        let path = self.meta_path();
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save_meta(&self, meta: &Meta) -> Result<()> {
        write_atomic(&self.meta_path(), toml::to_string_pretty(meta)?.as_bytes())
    }

    /// Loads the meta, applies `f` and saves it, holding a process-wide lock for the whole
    /// cycle. Returns the saved meta.
    pub fn update_meta(&self, f: impl FnOnce(&mut Meta)) -> Result<Meta> {
        let _guard = META_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut meta = self.load_meta()?;
        f(&mut meta);
        self.save_meta(&meta)?;
        Ok(meta)
    }

    /// Removes the session folder and everything in it.
    pub fn delete(self) -> Result<()> {
        fs::remove_dir_all(&self.dir).with_context(|| format!("removing {}", self.dir.display()))
    }

    pub fn wav_path(&self, track: Track) -> PathBuf {
        self.dir.join(format!("{}.wav", track.as_str()))
    }

    pub fn opus_path(&self, track: Track) -> PathBuf {
        self.dir.join(format!("{}.opus", track.as_str()))
    }

    pub fn json_path(&self, track: Track) -> PathBuf {
        self.dir.join(format!("{}.json", track.as_str()))
    }

    pub fn transcript_path(&self) -> PathBuf {
        self.dir.join("transcript.md")
    }

    pub fn summary_path(&self) -> PathBuf {
        self.dir.join("summary.md")
    }

    pub fn summaries_dir(&self) -> PathBuf {
        self.dir.join("summaries")
    }

    pub fn conversation_summary_path(&self, n: u32) -> PathBuf {
        self.summaries_dir().join(format!("{n:02}.md"))
    }

    pub fn load_track(&self, track: Track) -> Result<TrackTranscript> {
        let path = self.json_path(track);
        if !path.is_file() {
            return Ok(TrackTranscript::default());
        }
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save_track(&self, track: Track, transcript: &TrackTranscript) -> Result<()> {
        write_atomic(
            &self.json_path(track),
            serde_json::to_string_pretty(transcript)?.as_bytes(),
        )
    }
}

/// Writes `bytes` to a temp file next to `path`, syncs it, then renames it into place.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().context("path has no parent directory")?;
    let name = path
        .file_name()
        .context("path has no file name")?
        .to_string_lossy();
    let tmp = dir.join(format!(".{name}.tmp"));
    {
        let mut f =
            fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))
}

fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;
    set_private(path)
}

fn set_private(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("setting permissions on {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn at(s: &str) -> Zoned {
        s.parse().unwrap()
    }

    fn store() -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        (tmp, store)
    }

    fn sample_meta(id: &str, started: &Zoned) -> Meta {
        let mut m = Meta::new(
            id.to_string(),
            started.clone(),
            Trigger {
                kind: TriggerKind::Import,
                app_name: None,
                binary: None,
            },
            "auto".into(),
        );
        m.state = SessionState::Processing;
        m.conversations.push(Conversation::new(1, 0, 60_000));
        m
    }

    fn create_with_meta(store: &Store, t: &str) -> Session {
        let z = at(t);
        let s = store.create_session(&z).unwrap();
        s.save_meta(&sample_meta(&s.id, &z)).unwrap();
        s
    }

    #[test]
    fn create_session_uses_start_time_and_suffixes_collisions() {
        let (_tmp, store) = store();
        let t = at("2026-10-08T09:03:12.418+02:00[+02:00]");
        let a = store.create_session(&t).unwrap();
        let b = store.create_session(&t).unwrap();
        assert_eq!(a.id, "2026-10-08T090312");
        assert_eq!(b.id, "2026-10-08T090312-2");
        assert!(a.dir.is_dir() && b.dir.is_dir());
    }

    #[test]
    fn store_and_session_dirs_are_private() {
        let (_tmp, store) = store();
        let s = store
            .create_session(&at("2026-10-08T09:00:00+02:00[+02:00]"))
            .unwrap();
        for p in [
            store.root().to_path_buf(),
            store.sessions_dir(),
            s.dir.clone(),
        ] {
            let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "{}", p.display());
        }
    }

    #[test]
    fn meta_round_trips_through_toml() {
        let (_tmp, store) = store();
        let z = at("2026-10-08T09:03:12.418+02:00[+02:00]");
        let s = store.create_session(&z).unwrap();
        let mut m = sample_meta(&s.id, &z);
        m.title = Some("Planning".into());
        m.conversations[0].fail(Step::Summarize, &anyhow::anyhow!("boom"));
        s.save_meta(&m).unwrap();
        assert_eq!(s.load_meta().unwrap(), m);
        let text = std::fs::read_to_string(s.meta_path()).unwrap();
        assert!(text.contains("stage = \"failed\""), "{text}");
        assert!(text.contains("failed_step = \"summarize\""), "{text}");
    }

    #[test]
    fn resolve_by_exact_prefix_and_last() {
        let (_tmp, store) = store();
        create_with_meta(&store, "2026-10-07T10:00:00+02:00[+02:00]");
        create_with_meta(&store, "2026-10-08T09:00:00+02:00[+02:00]");
        create_with_meta(&store, "2026-10-08T15:30:00+02:00[+02:00]");
        // A folder without meta.toml is not a session.
        std::fs::create_dir(store.sessions_dir().join("2026-10-09T000000")).unwrap();

        assert_eq!(store.list().unwrap().len(), 3);
        assert_eq!(store.resolve("last").unwrap().id, "2026-10-08T153000");
        assert_eq!(store.resolve("2026-10-07").unwrap().id, "2026-10-07T100000");
        assert_eq!(
            store.resolve("2026-10-08T090000").unwrap().id,
            "2026-10-08T090000"
        );
        assert!(format!("{:#}", store.resolve("2026-10-08").unwrap_err()).contains("matches 2"));
        assert!(store.resolve("2025").is_err());
    }

    #[test]
    fn write_atomic_replaces_and_leaves_no_temp_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.txt");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "two");
        let names: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from("a.txt")]);
    }

    #[test]
    fn tracks_load_empty_when_missing_and_round_trip() {
        let (_tmp, store) = store();
        let s = store
            .create_session(&at("2026-10-08T09:00:00+02:00[+02:00]"))
            .unwrap();
        assert!(s.load_track(Track::Me).unwrap().segments.is_empty());
        let mut t = TrackTranscript::default();
        t.replace_conversation(
            2,
            vec![Segment {
                conversation: 2,
                start_ms: 5,
                end_ms: 9,
                text: " hi".into(),
                words: vec![],
            }],
        );
        s.save_track(Track::Me, &t).unwrap();
        assert_eq!(s.load_track(Track::Me).unwrap(), t);
    }

    fn session_with_meta() -> (tempfile::TempDir, Session) {
        let (tmp, store) = store();
        let s = create_with_meta(&store, "2026-10-19T10:00:00+02:00[+02:00]");
        (tmp, s)
    }

    #[test]
    fn update_meta_serializes_concurrent_writers() {
        // Two threads each append 50 conversations through update_meta; none may be lost.
        let (_tmp, session) = session_with_meta();
        std::thread::scope(|s| {
            for t in 0..2u32 {
                let session = &session;
                s.spawn(move || {
                    for i in 0..50u32 {
                        session
                            .update_meta(|m| {
                                m.conversations
                                    .push(Conversation::new(100 + t * 100 + i, 0, 1))
                            })
                            .unwrap();
                    }
                });
            }
        });
        // The sample meta already holds conversation 1.
        assert_eq!(session.load_meta().unwrap().conversations.len(), 101);
    }

    #[test]
    fn update_meta_returns_the_saved_meta() {
        let (_tmp, session) = session_with_meta();
        let saved = session
            .update_meta(|m| m.title = Some("Planning".into()))
            .unwrap();
        assert_eq!(saved, session.load_meta().unwrap());
        assert_eq!(saved.title.as_deref(), Some("Planning"));
    }

    #[test]
    fn delete_removes_the_folder() {
        let (_tmp, session) = session_with_meta();
        let dir = session.dir.clone();
        assert!(dir.is_dir());
        session.delete().unwrap();
        assert!(!dir.exists());
    }
}
