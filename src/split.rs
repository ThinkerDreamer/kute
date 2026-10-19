//! Splits a live session into conversations from VAD speech intervals (spec section 5).
//!
//! Pure: no I/O. All times are session-time milliseconds.

/// Silence kept on each side of a conversation.
pub const PAD_MS: u64 = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitEvent {
    Opened { start_ms: u64 },
    Closed { start_ms: u64, end_ms: u64 },
}

pub struct Splitter {
    gap_ms: u64,
    /// First speech start and latest speech end of the open conversation.
    open: Option<(u64, u64)>,
}

impl Splitter {
    pub fn new(gap_ms: u64) -> Splitter {
        Splitter { gap_ms, open: None }
    }

    /// Speech on either track, in session time.
    pub fn speech(&mut self, start_ms: u64, end_ms: u64) -> Option<SplitEvent> {
        match &mut self.open {
            Some((_, last_end)) => {
                // The slower track reports late, so only ever move the end forward.
                *last_end = (*last_end).max(end_ms);
                None
            }
            None => {
                self.open = Some((start_ms, end_ms));
                Some(SplitEvent::Opened {
                    start_ms: start_ms.saturating_sub(PAD_MS),
                })
            }
        }
    }

    /// Both tracks have been analysed up to `analysed_ms` (session time).
    pub fn advance(&mut self, analysed_ms: u64) -> Option<SplitEvent> {
        let (first, last_end) = self.open?;
        if analysed_ms < last_end + self.gap_ms {
            return None;
        }
        self.open = None;
        Some(SplitEvent::Closed {
            start_ms: first.saturating_sub(PAD_MS),
            end_ms: last_end + PAD_MS,
        })
    }

    /// The session ended at `end_ms`.
    pub fn finish(&mut self, end_ms: u64) -> Option<SplitEvent> {
        let (first, last_end) = self.open.take()?;
        Some(SplitEvent::Closed {
            start_ms: first.saturating_sub(PAD_MS),
            end_ms: (last_end + PAD_MS).min(end_ms),
        })
    }

    /// Padded start of the open conversation.
    pub fn open_start_ms(&self) -> Option<u64> {
        self.open.map(|(first, _)| first.saturating_sub(PAD_MS))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAP: u64 = 300_000;

    #[test]
    fn two_chats_far_apart_are_two_conversations() {
        let mut s = Splitter::new(GAP);
        assert_eq!(
            s.speech(600_000, 610_000),
            Some(SplitEvent::Opened { start_ms: 598_000 })
        );
        assert_eq!(s.speech(620_000, 630_000), None);
        assert_eq!(s.advance(929_999), None);
        assert_eq!(
            s.advance(930_000),
            Some(SplitEvent::Closed {
                start_ms: 598_000,
                end_ms: 632_000
            })
        );
        assert_eq!(
            s.speech(2_000_000, 2_001_000),
            Some(SplitEvent::Opened {
                start_ms: 1_998_000
            })
        );
    }

    #[test]
    fn pauses_shorter_than_the_gap_stay_in_one() {
        let mut s = Splitter::new(GAP);
        assert_eq!(s.speech(0, 1_000), Some(SplitEvent::Opened { start_ms: 0 }));
        assert_eq!(s.advance(100_000), None);
        assert_eq!(s.speech(200_000, 201_000), None);
        assert_eq!(s.advance(500_999), None);
        assert_eq!(
            s.advance(501_000),
            Some(SplitEvent::Closed {
                start_ms: 0,
                end_ms: 203_000
            })
        );
    }

    #[test]
    fn padding_clamps_at_zero_and_session_end() {
        let mut s = Splitter::new(GAP);
        assert_eq!(
            s.speech(500, 1_000),
            Some(SplitEvent::Opened { start_ms: 0 })
        );
        assert_eq!(s.open_start_ms(), Some(0));
        assert_eq!(
            s.finish(1_500),
            Some(SplitEvent::Closed {
                start_ms: 0,
                end_ms: 1_500
            })
        );
        assert_eq!(s.open_start_ms(), None);
    }

    #[test]
    fn no_speech_no_conversation() {
        let mut s = Splitter::new(GAP);
        assert_eq!(s.advance(10_000_000), None);
        assert_eq!(s.finish(10_000_000), None);
    }

    #[test]
    fn late_speech_from_the_slower_track_extends_the_open_conversation() {
        let mut s = Splitter::new(GAP);
        s.speech(0, 1_000);
        assert_eq!(s.advance(40_000), None);
        assert_eq!(s.speech(50_000, 60_000), None);
        assert_eq!(s.advance(301_000), None);
        assert_eq!(
            s.advance(360_000),
            Some(SplitEvent::Closed {
                start_ms: 0,
                end_ms: 62_000
            })
        );
    }
}
