//! Markdown rendering of transcripts and summaries. Both files are derived and rewritten
//! whenever the underlying data changes.

use crate::config::Labels;
use crate::merge::{Line, merge};
use crate::store::{Conversation, Meta, Track, TrackTranscript};
use crate::time::{add_ms, format_duration};

/// `# Title. 2026-10-08 09:03 (42 min, firefox)`; title and duration are optional.
pub fn header(meta: &Meta) -> String {
    let mut s = String::from("# ");
    if let Some(title) = &meta.title {
        s.push_str(title);
        s.push_str(". ");
    }
    s.push_str(&meta.started_at.strftime("%Y-%m-%d %H:%M").to_string());
    let mut extras = Vec::new();
    if let Some(ms) = meta.duration_ms() {
        extras.push(format_duration(ms));
    }
    extras.push(meta.app_label());
    s.push_str(&format!(" ({})", extras.join(", ")));
    s
}

pub fn time_range(meta: &Meta, c: &Conversation) -> String {
    format!(
        "{} to {}",
        add_ms(&meta.started_at, c.start_ms).strftime("%H:%M"),
        add_ms(&meta.started_at, c.end_ms).strftime("%H:%M")
    )
}

pub fn label(labels: &Labels, track: Track) -> &str {
    match track {
        Track::Me => &labels.me,
        Track::Others => &labels.others,
    }
}

pub fn conversation_lines(
    c: &Conversation,
    me: &TrackTranscript,
    others: &TrackTranscript,
) -> Vec<Line> {
    merge(me.conversation(c.n), others.conversation(c.n))
}

/// `[HH:MM:SS] Label: text` per line.
pub fn render_lines(meta: &Meta, labels: &Labels, lines: &[Line]) -> String {
    let mut out = String::new();
    for l in lines {
        let at = add_ms(&meta.started_at, l.start_ms).strftime("%H:%M:%S");
        out.push_str(&format!("[{at}] {}: {}\n", label(labels, l.track), l.text));
    }
    out
}

/// The whole transcript: header, then one section per transcribed conversation.
pub fn render_transcript(
    meta: &Meta,
    labels: &Labels,
    me: &TrackTranscript,
    others: &TrackTranscript,
) -> String {
    let mut out = header(meta);
    out.push('\n');
    for c in meta
        .conversations
        .iter()
        .filter(|c| !c.needs_transcription())
    {
        out.push_str(&format!("\n## {}\n\n", time_range(meta, c)));
        let lines = conversation_lines(c, me, others);
        if lines.is_empty() {
            out.push_str("_No speech detected._\n");
        } else {
            out.push_str(&render_lines(meta, labels, &lines));
        }
    }
    out
}

/// Header, then each conversation's summary under its time range.
pub fn render_summary(meta: &Meta, summaries: &[(&Conversation, String)]) -> String {
    let mut out = header(meta);
    out.push('\n');
    for (c, text) in summaries {
        out.push_str(&format!(
            "\n## {}\n\n{}\n",
            time_range(meta, c),
            text.trim()
        ));
    }
    out
}

pub fn word_count(lines: &[Line]) -> usize {
    lines
        .iter()
        .map(|l| l.text.split_whitespace().count())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Segment, Stage, Trigger, TriggerKind, Word};

    fn meta() -> Meta {
        let start: jiff::Zoned = "2026-10-08T09:03:12+02:00[+02:00]".parse().unwrap();
        let mut m = Meta::new(
            "2026-10-08T090312".into(),
            start.clone(),
            Trigger {
                kind: TriggerKind::App,
                app_name: Some("Firefox".into()),
                binary: Some("firefox".into()),
            },
            "en".into(),
        );
        m.title = Some("Coworking".into());
        m.ended_at = Some(crate::time::add_ms(&start, 31_020_000));
        let mut c1 = Conversation::new(1, 2_211_000, 3_034_500);
        c1.stage = Stage::Transcribed;
        let mut c2 = Conversation::new(2, 7_000_000, 7_100_000);
        c2.stage = Stage::Transcribed;
        let c3 = Conversation::new(3, 9_000_000, 9_100_000); // still pending
        m.conversations = vec![c1, c2, c3];
        m
    }

    fn one(conversation: u32, start_ms: u64, text: &str) -> TrackTranscript {
        TrackTranscript {
            segments: vec![Segment {
                conversation,
                start_ms,
                end_ms: start_ms + 500,
                text: text.into(),
                words: vec![Word {
                    start_ms,
                    end_ms: start_ms + 500,
                    text: text.into(),
                }],
            }],
        }
    }

    #[test]
    fn header_variants() {
        let mut m = meta();
        assert_eq!(
            header(&m),
            "# Coworking. 2026-10-08 09:03 (8 h 37 min, firefox)"
        );
        m.title = None;
        m.ended_at = None;
        m.trigger.kind = TriggerKind::Import;
        assert_eq!(header(&m), "# 2026-10-08 09:03 (import)");
    }

    #[test]
    fn transcript_layout() {
        let m = meta();
        let me = one(1, 2_211_500, " Ciao! Hai visto la MR?");
        let others = one(1, 2_214_000, " Sì, I left a couple of comments.");
        let labels = Labels {
            me: "Alex".into(),
            others: "Others".into(),
        };
        let got = render_transcript(&m, &labels, &me, &others);
        let want = "\
# Coworking. 2026-10-08 09:03 (8 h 37 min, firefox)

## 09:40 to 09:53

[09:40:03] Alex: Ciao! Hai visto la MR?
[09:40:06] Others: Sì, I left a couple of comments.

## 10:59 to 11:01

_No speech detected._
";
        assert_eq!(got, want);
    }

    #[test]
    fn summary_layout_and_word_count() {
        let m = meta();
        let c1 = &m.conversations[0];
        let got = render_summary(&m, &[(c1, "## Summary\n\nWe met.\n\n".into())]);
        assert_eq!(
            got,
            "# Coworking. 2026-10-08 09:03 (8 h 37 min, firefox)\n\n## 09:40 to 09:53\n\n## Summary\n\nWe met.\n"
        );
        let lines = conversation_lines(
            c1,
            &one(1, 2_211_500, " one two"),
            &one(1, 2_212_000, " three"),
        );
        assert_eq!(word_count(&lines), 3);
    }
}
