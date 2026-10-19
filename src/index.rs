//! Full-text search over transcripts and summaries. Derived from the session folders and
//! rebuildable at any time.

use crate::merge::Line;
use crate::store::Meta;
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::Path;
use std::time::Duration;

/// Bump on any change to `SCHEMA`. An index with another version is dropped and rebuilt from the
/// session folders, so there are no migrations.
const SCHEMA_VERSION: i64 = 1;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    app TEXT NOT NULL,
    conversation_count INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS lines USING fts5(
    text, session_id UNINDEXED, conversation UNINDEXED, t_ms UNINDEXED, track UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS summaries USING fts5(
    text, session_id UNINDEXED, conversation UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
";

pub struct Index {
    conn: Connection,
    recreated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitKind {
    Line,
    Summary,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub kind: HitKind,
    pub session_id: String,
    pub title: Option<String>,
    /// RFC 9557, as stored in `meta.toml`.
    pub started_at: String,
    pub conversation: u32,
    pub t_ms: Option<u64>,
    pub track: Option<String>,
    pub text: String,
}

impl Index {
    pub fn open(path: &Path) -> Result<Index> {
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // The daemon's worker and a foreground command can write at the same time.
        conn.busy_timeout(Duration::from_secs(5))?;
        Index::init(conn)
    }

    pub fn open_in_memory() -> Result<Index> {
        Index::init(Connection::open_in_memory()?)
    }

    /// Creates the tables, first dropping them if they were made by another schema version.
    fn init(conn: Connection) -> Result<Index> {
        let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        let recreated = version != SCHEMA_VERSION;
        if recreated {
            conn.execute_batch(
                "DROP TABLE IF EXISTS lines; DROP TABLE IF EXISTS summaries;
                 DROP TABLE IF EXISTS sessions;",
            )?;
        }
        conn.execute_batch(SCHEMA)?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(Index { conn, recreated })
    }

    /// Whether `open` found no index, or one from another schema version, and started empty.
    /// The caller should then reindex every session (`pipeline::open_index` does).
    pub fn recreated(&self) -> bool {
        self.recreated
    }

    pub fn upsert_session(&self, meta: &Meta) -> Result<()> {
        self.conn.execute(
            "INSERT INTO sessions (id, title, started_at, ended_at, app, conversation_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET title = ?2, started_at = ?3, ended_at = ?4, app = ?5,
                 conversation_count = ?6",
            params![
                meta.id,
                meta.title,
                meta.started_at.to_string(),
                meta.ended_at.as_ref().map(|z| z.to_string()),
                meta.app_label(),
                meta.conversations.len() as i64,
            ],
        )?;
        Ok(())
    }

    pub fn replace_lines(&self, session_id: &str, conversation: u32, lines: &[Line]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM lines WHERE session_id = ?1 AND conversation = ?2",
            params![session_id, conversation],
        )?;
        {
            let mut insert = tx.prepare(
                "INSERT INTO lines (text, session_id, conversation, t_ms, track) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for l in lines {
                insert.execute(params![
                    l.text,
                    session_id,
                    conversation,
                    l.start_ms as i64,
                    l.track.as_str()
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn replace_summary(
        &self,
        session_id: &str,
        conversation: u32,
        text: Option<&str>,
    ) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM summaries WHERE session_id = ?1 AND conversation = ?2",
            params![session_id, conversation],
        )?;
        if let Some(text) = text {
            tx.execute(
                "INSERT INTO summaries (text, session_id, conversation) VALUES (?1, ?2, ?3)",
                params![text, session_id, conversation],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM lines WHERE session_id = ?1",
            params![session_id],
        )?;
        tx.execute(
            "DELETE FROM summaries WHERE session_id = ?1",
            params![session_id],
        )?;
        tx.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
        tx.commit()?;
        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM lines; DELETE FROM summaries; DELETE FROM sessions;")?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let Some(fts) = fts_query(query) else {
            return Ok(Vec::new());
        };
        let mut stmt = self.conn.prepare(
            "SELECT 'line', lines.session_id, lines.conversation, lines.t_ms, lines.track, lines.text,
                    s.started_at, s.title, bm25(lines) AS rank
               FROM lines JOIN sessions s ON s.id = lines.session_id
              WHERE lines MATCH ?1
             UNION ALL
             SELECT 'summary', summaries.session_id, summaries.conversation, NULL, NULL,
                    snippet(summaries, 0, '[', ']', '…', 12),
                    s.started_at, s.title, bm25(summaries) AS rank
               FROM summaries JOIN sessions s ON s.id = summaries.session_id
              WHERE summaries MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![fts, limit as i64], |r| {
            let kind: String = r.get(0)?;
            Ok(Hit {
                kind: if kind == "line" {
                    HitKind::Line
                } else {
                    HitKind::Summary
                },
                session_id: r.get(1)?,
                conversation: r.get::<_, i64>(2)? as u32,
                t_ms: r.get::<_, Option<i64>>(3)?.map(|v| v as u64),
                track: r.get(4)?,
                text: r.get(5)?,
                started_at: r.get(6)?,
                title: r.get(7)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
}

/// Turns free text into an FTS5 query where every term is a quoted literal.
pub fn fts_query(input: &str) -> Option<String> {
    let terms: Vec<String> = input
        .split_whitespace()
        .filter(|t| t.chars().any(char::is_alphanumeric))
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Track, Trigger, TriggerKind};

    fn meta(id: &str) -> Meta {
        let mut m = Meta::new(
            id.into(),
            "2026-10-08T09:00:00+02:00[+02:00]".parse().unwrap(),
            Trigger {
                kind: TriggerKind::Import,
                app_name: None,
                binary: None,
            },
            "auto".into(),
        );
        m.title = Some("Sync".into());
        m
    }

    fn line(start_ms: u64, track: Track, text: &str) -> Line {
        Line {
            start_ms,
            track,
            text: text.into(),
        }
    }

    fn seeded() -> Index {
        let idx = Index::open_in_memory().unwrap();
        idx.upsert_session(&meta("s1")).unwrap();
        idx.replace_lines(
            "s1",
            1,
            &[
                line(1_000, Track::Me, "Perché l'ho fatto in C++?"),
                line(2_000, Track::Others, "the differential download is ready"),
            ],
        )
        .unwrap();
        idx.replace_summary(
            "s1",
            1,
            Some("## Decisions\n\nShip the differential download."),
        )
        .unwrap();
        idx
    }

    #[test]
    fn fts_query_quotes_every_term() {
        assert_eq!(fts_query("  "), None);
        assert_eq!(fts_query("\" -- ?"), None);
        assert_eq!(fts_query("a \"b\""), Some("\"a\" \"\"\"b\"\"\"".into()));
    }

    #[test]
    fn finds_lines_and_summaries() {
        let idx = seeded();
        let hits = idx.search("differential", 10).unwrap();
        assert_eq!(hits.len(), 2);
        let line_hit = hits.iter().find(|h| h.kind == HitKind::Line).unwrap();
        assert_eq!(line_hit.t_ms, Some(2_000));
        assert_eq!(line_hit.track.as_deref(), Some("others"));
        assert_eq!(line_hit.title.as_deref(), Some("Sync"));
        assert!(
            hits.iter()
                .any(|h| h.kind == HitKind::Summary && h.text.contains("differential"))
        );
    }

    #[test]
    fn search_handles_fts_syntax_literally() {
        let idx = seeded();
        for q in [
            "l'ho",
            "C++",
            "perche",
            "\"",
            "-x",
            "AND",
            "differential AND",
        ] {
            let result = idx.search(q, 10);
            assert!(result.is_ok(), "{q:?}: {:?}", result.err());
        }
        assert_eq!(idx.search("perche", 10).unwrap().len(), 1);
        assert_eq!(idx.search("l'ho", 10).unwrap().len(), 1);
        assert_eq!(idx.search("C++", 10).unwrap().len(), 1);
        assert!(idx.search("AND", 10).unwrap().is_empty());
    }

    #[test]
    fn replace_and_delete() {
        let idx = seeded();
        idx.replace_lines("s1", 1, &[line(0, Track::Me, "something else")])
            .unwrap();
        assert!(
            idx.search("differential", 10)
                .unwrap()
                .iter()
                .all(|h| h.kind == HitKind::Summary)
        );
        idx.replace_summary("s1", 1, None).unwrap();
        assert!(idx.search("differential", 10).unwrap().is_empty());
        idx.delete_session("s1").unwrap();
        assert!(idx.search("something", 10).unwrap().is_empty());
    }

    #[test]
    fn an_index_from_another_schema_version_is_recreated() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.sqlite");
        // A table with the right name and the wrong columns, and no version: an older kute.
        Connection::open(&path)
            .unwrap()
            .execute_batch("CREATE TABLE sessions (id TEXT PRIMARY KEY);")
            .unwrap();

        let idx = Index::open(&path).unwrap();
        assert!(idx.recreated());
        idx.upsert_session(&meta("s1")).unwrap();
        drop(idx);

        let idx = Index::open(&path).unwrap();
        assert!(!idx.recreated());
        let version: i64 = idx
            .conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }
}
