//! Turns per-track Whisper segments into one time-ordered list of lines.

use crate::store::{Segment, Track, Word};

/// Words further apart than this start a new line.
pub const SPLIT_GAP_MS: u64 = 1_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    pub start_ms: u64,
    pub track: Track,
    pub text: String,
}

pub fn track_lines<'a>(segments: impl IntoIterator<Item = &'a Segment>, track: Track) -> Vec<Line> {
    let mut out = Vec::new();
    for seg in segments {
        let Some((first, rest)) = seg.words.split_first() else {
            push_line(&mut out, seg.start_ms, track, &seg.text);
            continue;
        };
        let mut chunks: Vec<Vec<Word>> = vec![vec![first.clone()]];
        for word in rest {
            let prev_end = chunks.last().and_then(|c| c.last()).map_or(0, |w| w.end_ms);
            if word.start_ms.saturating_sub(prev_end) > SPLIT_GAP_MS {
                chunks.push(Vec::new());
            }
            chunks
                .last_mut()
                .expect("chunks is never empty")
                .push(word.clone());
        }
        // Whisper often pins a segment's first word to the segment start, well before
        // the rest: a lone leading word goes with the words that follow it.
        if chunks.len() > 1 && chunks[0].len() == 1 {
            let lone = chunks.remove(0).remove(0);
            let start_ms = chunks[0][0].start_ms;
            chunks[0].insert(0, Word { start_ms, ..lone });
        }
        for chunk in chunks {
            let text: String = chunk.iter().map(|w| w.text.as_str()).collect();
            push_line(&mut out, chunk[0].start_ms, track, &text);
        }
    }
    out
}

/// Both tracks as one list, ordered by start time; on equal times "me" comes first.
pub fn merge<'a>(
    me: impl IntoIterator<Item = &'a Segment>,
    others: impl IntoIterator<Item = &'a Segment>,
) -> Vec<Line> {
    let mut lines = track_lines(me, Track::Me);
    lines.extend(track_lines(others, Track::Others));
    lines.sort_by_key(|l| l.start_ms);
    lines
}

fn push_line(out: &mut Vec<Line>, start_ms: u64, track: Track, text: &str) {
    let text = text.trim();
    if !text.is_empty() {
        out.push(Line {
            start_ms,
            track,
            text: text.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(start_ms: u64, end_ms: u64, text: &str) -> Word {
        Word {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    fn seg(start_ms: u64, text: &str, words: Vec<Word>) -> Segment {
        Segment {
            conversation: 1,
            start_ms,
            end_ms: start_ms + 1,
            text: text.into(),
            words,
        }
    }

    fn texts(lines: &[Line]) -> Vec<(u64, &str)> {
        lines
            .iter()
            .map(|l| (l.start_ms, l.text.as_str()))
            .collect()
    }

    #[test]
    fn segment_without_words_becomes_one_line() {
        let s = [seg(500, "  Ciao a tutti ", vec![])];
        assert_eq!(
            texts(&track_lines(&s, Track::Me)),
            vec![(500, "Ciao a tutti")]
        );
    }

    #[test]
    fn splits_on_gaps_longer_than_one_second() {
        let s = [seg(
            0,
            "",
            vec![
                w(0, 400, " Hello"),
                w(450, 900, " there."),
                w(3_000, 3_300, " Anyway"),
            ],
        )];
        assert_eq!(
            texts(&track_lines(&s, Track::Me)),
            vec![(0, "Hello there."), (3_000, "Anyway")]
        );
    }

    #[test]
    fn a_gap_of_exactly_one_second_does_not_split() {
        let s = [seg(0, "", vec![w(0, 400, " a"), w(1_400, 1_500, " b")])];
        assert_eq!(texts(&track_lines(&s, Track::Me)), vec![(0, "a b")]);
    }

    #[test]
    fn lone_first_word_moves_to_the_next_chunk() {
        let s = [seg(
            0,
            "",
            vec![
                w(0, 200, " So"),
                w(5_000, 5_300, " the"),
                w(5_300, 5_600, " plan"),
            ],
        )];
        assert_eq!(
            texts(&track_lines(&s, Track::Me)),
            vec![(5_000, "So the plan")]
        );
    }

    #[test]
    fn lone_word_alone_in_segment_stays_put() {
        let s = [seg(0, "", vec![w(700, 900, " Yes.")])];
        assert_eq!(texts(&track_lines(&s, Track::Me)), vec![(700, "Yes.")]);
    }

    #[test]
    fn whitespace_only_text_is_dropped() {
        let s = [seg(0, "   ", vec![]), seg(10, "", vec![w(10, 20, " ")])];
        assert!(track_lines(&s, Track::Me).is_empty());
    }

    #[test]
    fn merge_sorts_by_time_and_puts_me_first_on_ties() {
        let me = [seg(1_000, "mine", vec![]), seg(5_000, "later", vec![])];
        let others = [seg(1_000, "theirs", vec![]), seg(2_000, "middle", vec![])];
        let lines = merge(&me, &others);
        let got: Vec<_> = lines
            .iter()
            .map(|l| (l.start_ms, l.track, l.text.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                (1_000, Track::Me, "mine"),
                (1_000, Track::Others, "theirs"),
                (2_000, Track::Others, "middle"),
                (5_000, Track::Me, "later"),
            ]
        );
    }
}
