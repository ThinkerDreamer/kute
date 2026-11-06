//! `meta.toml`: everything about a session except audio and text.

use super::Track;
use crate::time;
use jiff::Zoned;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Free text for the summarizer, e.g. who the other speakers are (`kute notes`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    pub kute_version: String,
    pub started_at: Zoned,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<Zoned>,
    pub state: SessionState,
    #[serde(default)]
    pub interrupted: bool,
    pub trigger: Trigger,
    #[serde(default)]
    pub tracks: Tracks,
    #[serde(default, skip_serializing_if = "TranscribeMeta::is_empty")]
    pub transcribe: TranscribeMeta,
    pub summary: SummaryMeta,
    #[serde(default)]
    pub conversations: Vec<Conversation>,
    /// Start of the conversation still being recorded, in session time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_conversation_ms: Option<u64>,
    #[serde(default)]
    pub compression: Compression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    Recording,
    Processing,
    Done,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trigger {
    pub kind: TriggerKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TriggerKind {
    App,
    Manual,
    Import,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tracks {
    #[serde(default)]
    pub me: TrackMeta,
    #[serde(default)]
    pub others: TrackMeta,
}

impl Tracks {
    pub fn get(&self, track: Track) -> &TrackMeta {
        match track {
            Track::Me => &self.me,
            Track::Others => &self.others,
        }
    }

    pub fn get_mut(&mut self, track: Track) -> &mut TrackMeta {
        match track {
            Track::Me => &mut self.me,
            Track::Others => &mut self.others,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrackMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// How much later than session time 0 this track's first sample arrived.
    #[serde(default)]
    pub offset_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TranscribeMeta {
    /// The languages spoken in this session (`kute language`): one is forced, several are
    /// detected among, `["auto"]` detects any. Empty: `transcribe.language` from the config.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub languages: Vec<String>,
    /// The same for one track, in place of `languages`; empty: `languages`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub me: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub others: Vec<String>,
}

impl TranscribeMeta {
    fn is_empty(&self) -> bool {
        self.languages.is_empty() && self.me.is_empty() && self.others.is_empty()
    }

    /// The session's languages for `track`; empty when the session sets none.
    pub fn for_track(&self, track: Track) -> &[String] {
        let own = match track {
            Track::Me => &self.me,
            Track::Others => &self.others,
        };
        if own.is_empty() { &self.languages } else { own }
    }
}

/// A change to a session's languages (`kute language`, `--lang`). `Some(empty)` removes a
/// setting.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LanguageChange {
    /// Both tracks; replaces any per-track languages too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub me: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub others: Option<Vec<String>>,
}

impl LanguageChange {
    pub fn is_empty(&self) -> bool {
        self.all.is_none() && self.me.is_none() && self.others.is_none()
    }

    pub fn apply(&self, t: &mut TranscribeMeta) {
        if let Some(all) = &self.all {
            t.languages = all.clone();
            t.me.clear();
            t.others.clear();
        }
        if let Some(me) = &self.me {
            t.me = me.clone();
        }
        if let Some(others) = &self.others {
            t.others = others.clone();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryMeta {
    pub language: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    pub n: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub stage: Stage,
    #[serde(default)]
    pub languages: Languages,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whisper_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summarizer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_step: Option<Step>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub attempts: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Pending,
    Transcribed,
    Summarized,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Step {
    Transcribe,
    Summarize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Languages {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub me: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub others: Option<String>,
}

impl Languages {
    pub fn set(&mut self, track: Track, language: Option<String>) {
        match track {
            Track::Me => self.me = language,
            Track::Others => self.others = language,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Compression {
    pub stage: CompressionStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompressionStage {
    #[default]
    Pending,
    Done,
    Failed,
}

impl Meta {
    pub fn new(id: String, started_at: Zoned, trigger: Trigger, summary_language: String) -> Meta {
        Meta {
            id,
            title: None,
            notes: None,
            kute_version: env!("CARGO_PKG_VERSION").to_string(),
            started_at,
            ended_at: None,
            state: SessionState::Recording,
            interrupted: false,
            trigger,
            tracks: Tracks::default(),
            transcribe: TranscribeMeta::default(),
            summary: SummaryMeta {
                language: summary_language,
            },
            conversations: Vec::new(),
            open_conversation_ms: None,
            compression: Compression::default(),
        }
    }

    pub fn duration_ms(&self) -> Option<u64> {
        self.ended_at
            .as_ref()
            .map(|end| time::ms_between(&self.started_at, end))
    }

    /// What triggered the session, for headers and listings.
    pub fn app_label(&self) -> String {
        match self.trigger.kind {
            TriggerKind::App => self
                .trigger
                .app_name
                .clone()
                .or_else(|| self.trigger.binary.clone())
                .unwrap_or_else(|| "app".into()),
            TriggerKind::Manual => "manual".into(),
            TriggerKind::Import => "import".into(),
        }
    }

    pub fn conversation(&self, n: u32) -> Option<&Conversation> {
        self.conversations.iter().find(|c| c.n == n)
    }

    pub fn conversation_mut(&mut self, n: u32) -> Option<&mut Conversation> {
        self.conversations.iter_mut().find(|c| c.n == n)
    }
}

impl Conversation {
    pub fn new(n: u32, start_ms: u64, end_ms: u64) -> Conversation {
        Conversation {
            n,
            start_ms,
            end_ms,
            stage: Stage::Pending,
            languages: Languages::default(),
            whisper_model: None,
            summarizer: None,
            failed_step: None,
            error: None,
            attempts: 0,
        }
    }

    pub fn fail(&mut self, step: Step, err: &anyhow::Error) {
        self.stage = Stage::Failed;
        self.failed_step = Some(step);
        self.error = Some(format!("{err:#}"));
        self.attempts += 1;
    }

    pub fn clear_error(&mut self) {
        self.failed_step = None;
        self.error = None;
    }

    pub fn needs_transcription(&self) -> bool {
        self.stage == Stage::Pending
            || (self.stage == Stage::Failed && self.failed_step == Some(Step::Transcribe))
    }

    pub fn needs_summary(&self) -> bool {
        self.stage == Stage::Transcribed
            || (self.stage == Stage::Failed && self.failed_step == Some(Step::Summarize))
    }

    /// True once nothing more will happen to this conversation without a user command.
    /// A failed transcription is not settled: its retry needs the original WAV, so the
    /// session must not be compressed yet.
    pub fn is_settled(&self) -> bool {
        match self.stage {
            Stage::Summarized | Stage::Skipped => true,
            Stage::Failed => self.failed_step == Some(Step::Summarize),
            Stage::Pending | Stage::Transcribed => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_with(trigger: Trigger) -> Meta {
        Meta::new(
            "2026-10-19T100000".into(),
            "2026-10-19T10:00:00+02:00[+02:00]".parse().unwrap(),
            trigger,
            "auto".into(),
        )
    }

    #[test]
    fn app_label_prefers_app_name() {
        let mut t = Trigger {
            kind: TriggerKind::App,
            app_name: Some("Jitsi Meet".into()),
            binary: Some("electron".into()),
        };
        assert_eq!(meta_with(t.clone()).app_label(), "Jitsi Meet");
        t.app_name = None;
        assert_eq!(meta_with(t).app_label(), "electron");
    }

    #[test]
    fn open_conversation_ms_round_trips() {
        let mut m = meta_with(Trigger {
            kind: TriggerKind::Manual,
            app_name: None,
            binary: None,
        });
        assert!(
            !toml::to_string_pretty(&m)
                .unwrap()
                .contains("open_conversation_ms")
        );
        m.open_conversation_ms = Some(12_000);
        let text = toml::to_string_pretty(&m).unwrap();
        assert!(text.contains("open_conversation_ms = 12000"), "{text}");
        assert_eq!(toml::from_str::<Meta>(&text).unwrap(), m);
    }
}
