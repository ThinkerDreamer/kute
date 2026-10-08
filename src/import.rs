//! `kute import`: brings in two recordings made elsewhere (for example by `record-call`).

use crate::audio;
use crate::config::Config;
use crate::store::{Conversation, Meta, Session, SessionState, Store, Track, Trigger, TriggerKind};
use crate::time;
use anyhow::{Context, Result, bail};
use jiff::{Timestamp, Zoned, tz::TimeZone};
use std::fs;
use std::path::{Path, PathBuf};

pub struct ImportRequest {
    pub me: PathBuf,
    pub others: PathBuf,
    pub start: Option<String>,
    pub title: Option<String>,
}

pub fn import(store: &Store, config: &Config, req: &ImportRequest) -> Result<Session> {
    for path in [&req.me, &req.others] {
        if !path.is_file() {
            bail!("{} does not exist", path.display());
        }
    }
    let started_at = resolve_start(req)?;
    let session = store.create_session(&started_at)?;
    match fill_session(&session, config, req, started_at) {
        Ok(()) => Ok(session),
        Err(e) => {
            let _ = fs::remove_dir_all(&session.dir);
            Err(e)
        }
    }
}

fn fill_session(
    session: &Session,
    config: &Config,
    req: &ImportRequest,
    started_at: Zoned,
) -> Result<()> {
    audio::normalize_to_wav(&req.me, &session.wav_path(Track::Me))
        .with_context(|| format!("converting {}", req.me.display()))?;
    audio::normalize_to_wav(&req.others, &session.wav_path(Track::Others))
        .with_context(|| format!("converting {}", req.others.display()))?;
    let total = audio::wav_duration_ms(&session.wav_path(Track::Me))?
        .max(audio::wav_duration_ms(&session.wav_path(Track::Others))?);

    let mut meta = Meta::new(
        session.id.clone(),
        started_at.clone(),
        Trigger {
            kind: TriggerKind::Import,
            app_name: None,
            binary: None,
        },
        config.summary.language.clone(),
    );
    meta.title = req.title.clone();
    meta.ended_at = Some(time::add_ms(&started_at, total));
    meta.state = SessionState::Processing;
    if total > 0 {
        meta.conversations.push(Conversation::new(1, 0, total));
    }
    session.save_meta(&meta)
}

/// `--start`, else `<prefix>.start` next to the "me" file, else the file's birth time.
pub fn resolve_start(req: &ImportRequest) -> Result<Zoned> {
    if let Some(s) = &req.start {
        return time::parse_start(s);
    }
    if let Some(path) = start_file_for(&req.me).filter(|p| p.is_file()) {
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        return time::parse_start(&text).with_context(|| format!("parsing {}", path.display()));
    }
    let created = fs::metadata(&req.me)
        .and_then(|m| m.created())
        .with_context(|| format!("no start time for {}; pass --start", req.me.display()))?;
    Ok(time::truncate_ms(
        &Timestamp::try_from(created)?.to_zoned(TimeZone::system()),
    ))
}

/// `<prefix>-me.wav` -> `<prefix>.start`.
pub fn start_file_for(me: &Path) -> Option<PathBuf> {
    let name = me.file_name()?.to_str()?;
    let prefix = name.strip_suffix("-me.wav")?;
    Some(me.with_file_name(format!("{prefix}.start")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::write_test_wav;
    use crate::store::{SessionState, TriggerKind};

    fn req(dir: &Path, start: Option<&str>) -> ImportRequest {
        ImportRequest {
            me: dir.join("2026-10-08_1500_sync-me.wav"),
            others: dir.join("2026-10-08_1500_sync-others.wav"),
            start: start.map(String::from),
            title: Some("Sync".into()),
        }
    }

    #[test]
    fn start_file_naming() {
        assert_eq!(
            start_file_for(Path::new("/r/2026-10-08_1500_sync-me.wav")),
            Some(PathBuf::from("/r/2026-10-08_1500_sync.start"))
        );
        assert_eq!(start_file_for(Path::new("/r/me.wav")), None);
    }

    #[test]
    fn start_from_flag_then_start_file() {
        let tmp = tempfile::tempdir().unwrap();
        let r = req(tmp.path(), Some("2026-10-08T15:00:00+02:00[+02:00]"));
        assert_eq!(
            resolve_start(&r).unwrap().strftime("%H:%M").to_string(),
            "15:00"
        );

        std::fs::write(
            tmp.path().join("2026-10-08_1500_sync.start"),
            "2026-10-08T15:00:12\n",
        )
        .unwrap();
        let r = req(tmp.path(), None);
        assert_eq!(
            resolve_start(&r).unwrap().datetime().to_string(),
            "2026-10-08T15:00:12"
        );
    }

    #[test]
    fn import_creates_a_processing_session() {
        let tmp = tempfile::tempdir().unwrap();
        let r = req(tmp.path(), Some("2026-10-08T15:00:00+02:00[+02:00]"));
        write_test_wav(&r.me, 3.0, 1_000);
        write_test_wav(&r.others, 2.0, 1_000);
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let session = import(&store, &Config::default(), &r).unwrap();

        assert_eq!(session.id, "2026-10-08T150000");
        let meta = session.load_meta().unwrap();
        assert_eq!(meta.state, SessionState::Processing);
        assert_eq!(meta.trigger.kind, TriggerKind::Import);
        assert_eq!(meta.title.as_deref(), Some("Sync"));
        assert_eq!(meta.summary.language, "auto");
        assert_eq!(meta.duration_ms(), Some(3_000));
        assert_eq!(meta.conversations.len(), 1);
        assert_eq!(
            (meta.conversations[0].start_ms, meta.conversations[0].end_ms),
            (0, 3_000)
        );
        assert!(session.wav_path(Track::Me).is_file() && session.wav_path(Track::Others).is_file());
    }

    #[test]
    fn failed_import_leaves_no_session_behind() {
        let tmp = tempfile::tempdir().unwrap();
        let r = req(tmp.path(), Some("2026-10-08T15:00:00+02:00[+02:00]"));
        write_test_wav(&r.me, 1.0, 1_000);
        std::fs::write(&r.others, b"not audio").unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        assert!(import(&store, &Config::default(), &r).is_err());
        assert!(store.list().unwrap().is_empty());
        assert_eq!(std::fs::read_dir(store.sessions_dir()).unwrap().count(), 0);
    }

    #[test]
    fn missing_input_is_reported_before_anything_is_created() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open(tmp.path().join("kute")).unwrap();
        let err = import(
            &store,
            &Config::default(),
            &req(tmp.path(), Some("2026-10-08T15:00:00")),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("does not exist"), "{err:#}");
        assert_eq!(std::fs::read_dir(store.sessions_dir()).unwrap().count(), 0);
    }
}
