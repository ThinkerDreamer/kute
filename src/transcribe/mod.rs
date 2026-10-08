//! Speech detection, chunking, and per-track transcription. The engines sit behind
//! traits so the orchestration is testable and other backends can be added.

pub mod whisper;

use crate::audio::{ms_to_samples, samples_to_ms};
use crate::store::{Segment, Word};
use anyhow::Result;
use std::collections::BTreeMap;

pub const CHUNK_JOIN_GAP_MS: u64 = 2_000;
pub const CHUNK_MAX_MS: u64 = 120_000;
pub const CHUNK_PAD_MS: u64 = 200;
pub const MIN_DETECT_MS: u64 = 3_000;
pub const DETECT_WINDOW_MS: u64 = 30_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start_ms: u64,
    pub end_ms: u64,
}

impl Span {
    pub fn len_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A Whisper segment with times relative to the audio that was passed in.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub words: Vec<Word>,
}

pub trait SpeechDetector {
    fn speech_regions(&mut self, samples: &[f32]) -> Result<Vec<Span>>;
}

pub trait Transcriber {
    fn model_name(&self) -> &str;
    fn detect_language(&mut self, samples: &[f32]) -> Result<String>;
    fn transcribe(&mut self, samples: &[f32], language: &str) -> Result<Vec<RawSegment>>;
}

/// Pads speech regions, joins close ones, and caps chunk length. Chunks never overlap.
pub fn plan_chunks(regions: &[Span], total_ms: u64) -> Vec<Span> {
    let mut sorted = regions.to_vec();
    sorted.sort_by_key(|r| r.start_ms);
    let mut chunks: Vec<Span> = Vec::new();
    for r in sorted {
        let padded = Span {
            start_ms: r.start_ms.saturating_sub(CHUNK_PAD_MS),
            end_ms: (r.end_ms + CHUNK_PAD_MS).min(total_ms),
        };
        if padded.end_ms <= padded.start_ms {
            continue;
        }
        match chunks.last_mut() {
            Some(c)
                if padded.start_ms <= c.end_ms + CHUNK_JOIN_GAP_MS
                    && padded.end_ms.saturating_sub(c.start_ms) <= CHUNK_MAX_MS =>
            {
                c.end_ms = c.end_ms.max(padded.end_ms);
            }
            Some(c) => {
                let start_ms = padded.start_ms.max(c.end_ms);
                if padded.end_ms > start_ms {
                    chunks.push(Span {
                        start_ms,
                        end_ms: padded.end_ms,
                    });
                }
            }
            None => chunks.push(padded),
        }
    }
    chunks
}

pub struct TrackResult {
    pub language: Option<String>,
    pub segments: Vec<Segment>,
}

/// Transcribes one track's audio for one conversation. `base_ms` is the session time of
/// `samples[0]`; returned segments are in session time.
pub fn transcribe_track(
    samples: &[f32],
    base_ms: u64,
    conversation: u32,
    forced_language: Option<&str>,
    vad: &mut dyn SpeechDetector,
    asr: &mut dyn Transcriber,
) -> Result<TrackResult> {
    let total_ms = samples_to_ms(samples.len());
    let chunks = plan_chunks(&vad.speech_regions(samples)?, total_ms);
    let languages = assign_languages(&chunks, samples, forced_language, asr)?;

    let mut segments = Vec::new();
    let mut coverage: BTreeMap<String, u64> = BTreeMap::new();
    for (chunk, language) in chunks.iter().zip(&languages) {
        *coverage.entry(language.clone()).or_default() += chunk.len_ms();
        let offset = base_ms + chunk.start_ms;
        for raw in asr.transcribe(slice(samples, *chunk), language)? {
            segments.push(Segment {
                conversation,
                start_ms: offset + raw.start_ms,
                end_ms: offset + raw.end_ms,
                text: raw.text,
                words: raw
                    .words
                    .into_iter()
                    .map(|w| Word {
                        start_ms: offset + w.start_ms,
                        end_ms: offset + w.end_ms,
                        text: w.text,
                    })
                    .collect(),
            });
        }
    }

    let mut language: Option<(String, u64)> = None;
    for (lang, ms) in coverage {
        if language.as_ref().is_none_or(|(_, best)| ms > *best) {
            language = Some((lang, ms));
        }
    }
    Ok(TrackResult {
        language: language.map(|(l, _)| l),
        segments,
    })
}

fn assign_languages(
    chunks: &[Span],
    samples: &[f32],
    forced: Option<&str>,
    asr: &mut dyn Transcriber,
) -> Result<Vec<String>> {
    if let Some(lang) = forced {
        return Ok(vec![lang.to_string(); chunks.len()]);
    }
    let mut detected: Vec<Option<String>> = vec![None; chunks.len()];
    for (i, c) in chunks.iter().enumerate() {
        if c.len_ms() >= MIN_DETECT_MS {
            detected[i] = Some(asr.detect_language(slice(samples, *c))?);
        }
    }
    let known: Vec<(usize, String)> = detected
        .iter()
        .enumerate()
        .filter_map(|(i, l)| l.clone().map(|l| (i, l)))
        .collect();
    if known.is_empty() {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let mut joined = Vec::new();
        for c in chunks {
            joined.extend_from_slice(slice(samples, *c));
            if joined.len() >= ms_to_samples(DETECT_WINDOW_MS) {
                break;
            }
        }
        let lang = asr.detect_language(&joined)?;
        return Ok(vec![lang; chunks.len()]);
    }
    Ok((0..chunks.len())
        .map(|i| {
            detected[i].clone().unwrap_or_else(|| {
                let nearest = known.iter().min_by_key(|(j, _)| (i.abs_diff(*j), *j > i));
                nearest.expect("known is not empty").1.clone()
            })
        })
        .collect())
}

fn slice(samples: &[f32], span: Span) -> &[f32] {
    let start = ms_to_samples(span.start_ms).min(samples.len());
    let end = ms_to_samples(span.end_ms).min(samples.len()).max(start);
    &samples[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FakeVad(Vec<Span>);
    impl SpeechDetector for FakeVad {
        fn speech_regions(&mut self, _samples: &[f32]) -> Result<Vec<Span>> {
            Ok(self.0.clone())
        }
    }

    #[derive(Default)]
    struct FakeAsr {
        detect: VecDeque<String>,
        detect_calls: Vec<usize>,
        calls: Vec<(usize, String)>,
    }
    impl Transcriber for FakeAsr {
        fn model_name(&self) -> &str {
            "fake"
        }
        fn detect_language(&mut self, samples: &[f32]) -> Result<String> {
            self.detect_calls.push(samples.len());
            Ok(self.detect.pop_front().unwrap_or_else(|| "en".into()))
        }
        fn transcribe(&mut self, samples: &[f32], language: &str) -> Result<Vec<RawSegment>> {
            self.calls.push((samples.len(), language.to_string()));
            Ok(vec![RawSegment {
                start_ms: 100,
                end_ms: 600,
                text: " hi".into(),
                words: vec![Word {
                    start_ms: 100,
                    end_ms: 600,
                    text: " hi".into(),
                }],
            }])
        }
    }

    fn span(a: u64, b: u64) -> Span {
        Span {
            start_ms: a,
            end_ms: b,
        }
    }

    fn silence(ms: u64) -> Vec<f32> {
        vec![0.0; crate::audio::ms_to_samples(ms)]
    }

    #[test]
    fn plan_pads_and_clamps() {
        assert_eq!(
            plan_chunks(&[span(100, 1_000)], 1_100),
            vec![span(0, 1_100)]
        );
    }

    #[test]
    fn plan_joins_small_gaps_and_keeps_big_ones() {
        assert_eq!(
            plan_chunks(&[span(0, 1_000), span(3_000, 4_000)], 10_000),
            vec![span(0, 4_200)]
        );
        assert_eq!(
            plan_chunks(&[span(0, 1_000), span(5_000, 6_000)], 10_000),
            vec![span(0, 1_200), span(4_800, 6_200)]
        );
    }

    #[test]
    fn plan_caps_chunk_length_without_overlap() {
        let regions: Vec<Span> = (0..200).map(|i| span(i * 1_000, i * 1_000 + 500)).collect();
        let chunks = plan_chunks(&regions, 300_000);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.len_ms() <= CHUNK_MAX_MS, "{c:?}");
        }
        for pair in chunks.windows(2) {
            assert!(pair[1].start_ms >= pair[0].end_ms, "{pair:?}");
        }
    }

    #[test]
    fn no_speech_means_no_calls() {
        let mut asr = FakeAsr::default();
        let r =
            transcribe_track(&silence(10_000), 0, 1, None, &mut FakeVad(vec![]), &mut asr).unwrap();
        assert_eq!(r.language, None);
        assert!(r.segments.is_empty());
        assert!(asr.calls.is_empty() && asr.detect_calls.is_empty());
    }

    #[test]
    fn times_are_shifted_into_session_time() {
        let mut asr = FakeAsr::default();
        let r = transcribe_track(
            &silence(20_000),
            60_000,
            3,
            None,
            &mut FakeVad(vec![span(1_000, 5_000)]),
            &mut asr,
        )
        .unwrap();
        // chunk starts at 1_000 - 200 pad; segment at +100 inside it
        assert_eq!(r.segments.len(), 1);
        assert_eq!(r.segments[0].conversation, 3);
        assert_eq!(r.segments[0].start_ms, 60_000 + 800 + 100);
        assert_eq!(r.segments[0].words[0].end_ms, 60_000 + 800 + 600);
        assert_eq!(r.language.as_deref(), Some("en"));
    }

    #[test]
    fn forced_language_skips_detection() {
        let mut asr = FakeAsr::default();
        transcribe_track(
            &silence(10_000),
            0,
            1,
            Some("it"),
            &mut FakeVad(vec![span(0, 4_000)]),
            &mut asr,
        )
        .unwrap();
        assert!(asr.detect_calls.is_empty());
        assert_eq!(asr.calls[0].1, "it");
    }

    #[test]
    fn short_chunk_takes_nearest_long_chunk_language() {
        let mut asr = FakeAsr {
            detect: VecDeque::from(vec!["it".to_string()]),
            ..Default::default()
        };
        let vad = &mut FakeVad(vec![span(0, 5_000), span(10_000, 11_000)]);
        transcribe_track(&silence(20_000), 0, 1, None, vad, &mut asr).unwrap();
        assert_eq!(asr.detect_calls.len(), 1);
        assert_eq!(
            asr.calls.iter().map(|c| c.1.as_str()).collect::<Vec<_>>(),
            vec!["it", "it"]
        );
    }

    #[test]
    fn all_short_chunks_share_one_detection() {
        let mut asr = FakeAsr {
            detect: VecDeque::from(vec!["it".to_string()]),
            ..Default::default()
        };
        let vad = &mut FakeVad(vec![span(0, 1_000), span(5_000, 6_000)]);
        let r = transcribe_track(&silence(10_000), 0, 1, None, vad, &mut asr).unwrap();
        assert_eq!(asr.detect_calls.len(), 1);
        // [0, 1_200] (padding clamped at 0) + [4_800, 6_200]
        assert_eq!(
            asr.detect_calls[0],
            crate::audio::ms_to_samples(1_200 + 1_400)
        );
        assert_eq!(r.language.as_deref(), Some("it"));
    }

    #[test]
    fn track_language_is_the_dominant_one() {
        let mut asr = FakeAsr {
            detect: VecDeque::from(vec!["en".to_string(), "it".to_string()]),
            ..Default::default()
        };
        let vad = &mut FakeVad(vec![span(0, 6_000), span(20_000, 24_000)]);
        let r = transcribe_track(&silence(30_000), 0, 1, None, vad, &mut asr).unwrap();
        assert_eq!(r.language.as_deref(), Some("en"));
        assert_eq!(
            asr.calls.iter().map(|c| c.1.as_str()).collect::<Vec<_>>(),
            vec!["en", "it"]
        );
    }
}
