//! `me.json` / `others.json`: Whisper output in session time, tagged by conversation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrackTranscript {
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub conversation: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default)]
    pub words: Vec<Word>,
}

/// A word as Whisper produced it. `text` keeps Whisper's leading space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

impl TrackTranscript {
    pub fn replace_conversation(&mut self, n: u32, segments: Vec<Segment>) {
        self.segments.retain(|s| s.conversation != n);
        self.segments.extend(segments);
        self.segments.sort_by_key(|s| (s.start_ms, s.conversation));
    }

    pub fn conversation(&self, n: u32) -> Vec<&Segment> {
        self.segments
            .iter()
            .filter(|s| s.conversation == n)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(conversation: u32, start_ms: u64) -> Segment {
        Segment {
            conversation,
            start_ms,
            end_ms: start_ms + 10,
            text: format!("c{conversation}"),
            words: vec![],
        }
    }

    #[test]
    fn replace_conversation_swaps_only_that_conversation_and_keeps_order() {
        let mut t = TrackTranscript::default();
        t.replace_conversation(1, vec![seg(1, 100), seg(1, 0)]);
        t.replace_conversation(2, vec![seg(2, 500)]);
        t.replace_conversation(1, vec![seg(1, 50)]);
        let starts: Vec<_> = t
            .segments
            .iter()
            .map(|s| (s.conversation, s.start_ms))
            .collect();
        assert_eq!(starts, vec![(1, 50), (2, 500)]);
        assert_eq!(t.conversation(2).len(), 1);
        assert!(t.conversation(3).is_empty());
    }
}
