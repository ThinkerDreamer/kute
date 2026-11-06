//! whisper.cpp (via whisper-rs) as the `Transcriber`, and its Silero VAD as the
//! `SpeechDetector`. whisper.cpp's own VAD option is not used: it remaps segment times but
//! not token times, which would break word timestamps.

use super::{RawSegment, Span, SpeechDetector, Transcriber};
use crate::store::Word;
use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadContext,
    WhisperVadContextParams, WhisperVadParams,
};

pub struct SileroVad {
    ctx: WhisperVadContext,
}

impl SileroVad {
    pub fn load(path: &Path) -> Result<SileroVad> {
        let mut params = WhisperVadContextParams::default();
        params.set_use_gpu(false);
        params.set_n_threads(2);
        let path = path.to_str().context("VAD model path is not valid UTF-8")?;
        let ctx = WhisperVadContext::new(path, params)
            .map_err(|e| anyhow!("loading VAD model {path}: {e}"))?;
        Ok(SileroVad { ctx })
    }
}

impl SpeechDetector for SileroVad {
    fn speech_regions(&mut self, samples: &[f32]) -> Result<Vec<Span>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let segments = self
            .ctx
            .segments_from_samples(WhisperVadParams::new(), samples)
            .map_err(|e| anyhow!("running VAD: {e}"))?;
        // VAD timestamps are in centiseconds.
        Ok(segments
            .map(|s| Span {
                start_ms: (s.start.max(0.0) * 10.0) as u64,
                end_ms: (s.end.max(0.0) * 10.0) as u64,
            })
            .collect())
    }
}

pub struct WhisperTranscriber {
    ctx: WhisperContext,
    name: String,
    threads: usize,
}

impl WhisperTranscriber {
    pub fn load(path: &Path, name: &str, use_gpu: bool) -> Result<WhisperTranscriber> {
        let mut params = WhisperContextParameters::default();
        params.use_gpu(use_gpu);
        let path = path.to_str().context("model path is not valid UTF-8")?;
        let ctx = WhisperContext::new_with_params(path, params)
            .map_err(|e| anyhow!("loading Whisper model {path}: {e}"))?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8);
        Ok(WhisperTranscriber {
            ctx,
            name: name.to_string(),
            threads,
        })
    }
}

impl Transcriber for WhisperTranscriber {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn detect_language(&mut self, samples: &[f32], candidates: &[String]) -> Result<String> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow!("creating Whisper state: {e}"))?;
        state
            .pcm_to_mel(samples, self.threads)
            .map_err(|e| anyhow!("computing mel spectrogram: {e}"))?;
        let (best, probs) = state
            .lang_detect(0, self.threads)
            .map_err(|e| anyhow!("detecting language: {e}"))?;
        let id = most_likely(best, &probs, candidates)?;
        match whisper_rs::get_lang_str(id) {
            Some(lang) => Ok(lang.to_string()),
            None => bail!("Whisper returned unknown language id {id}"),
        }
    }

    fn transcribe(&mut self, samples: &[f32], language: &str) -> Result<Vec<RawSegment>> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow!("creating Whisper state: {e}"))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(language));
        params.set_n_threads(self.threads as i32);
        params.set_token_timestamps(true);
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        state
            .full(params, samples)
            .map_err(|e| anyhow!("Whisper failed: {e}"))?;

        let eot = self.ctx.token_eot();
        let mut out = Vec::new();
        for segment in state.as_iter() {
            let mut pieces = Vec::new();
            for i in 0..segment.n_tokens() {
                let Some(token) = segment.get_token(i) else {
                    continue;
                };
                let data = token.token_data();
                pieces.push(TokenPiece {
                    bytes: token.to_bytes().map(<[u8]>::to_vec).unwrap_or_default(),
                    t0_cs: data.t0,
                    t1_cs: data.t1,
                    special: data.id >= eot,
                });
            }
            out.push(RawSegment {
                start_ms: (segment.start_timestamp().max(0) as u64) * 10,
                end_ms: (segment.end_timestamp().max(0) as u64) * 10,
                text: segment
                    .to_str_lossy()
                    .map(|s| s.into_owned())
                    .unwrap_or_default(),
                words: tokens_to_words(&pieces),
            });
        }
        Ok(out)
    }
}

/// Whisper's code for a language given by code or English name, e.g. `ru` for `ru` or
/// `russian`; `None` when Whisper doesn't know it.
pub fn language_code(name: &str) -> Option<&'static str> {
    whisper_rs::get_lang_id(name).and_then(whisper_rs::get_lang_str)
}

/// The most likely of `candidates` by Whisper's per-language probabilities `probs`, or
/// `best` (Whisper's own pick) when there are no candidates.
fn most_likely(best: i32, probs: &[f32], candidates: &[String]) -> Result<i32> {
    if candidates.is_empty() {
        return Ok(best);
    }
    let mut pick: Option<(i32, f32)> = None;
    for code in candidates {
        let id = whisper_rs::get_lang_id(code)
            .with_context(|| format!("Whisper doesn't know the language {code:?}"))?;
        let p = probs.get(id as usize).copied().unwrap_or(0.0);
        if pick.is_none_or(|(_, best_p)| p > best_p) {
            pick = Some((id, p));
        }
    }
    Ok(pick.expect("candidates is not empty").0)
}

/// One Whisper token: raw bytes (possibly part of a UTF-8 character) and its times in
/// centiseconds.
pub struct TokenPiece {
    pub bytes: Vec<u8>,
    pub t0_cs: i64,
    pub t1_cs: i64,
    pub special: bool,
}

/// Groups tokens into words: a token starting with a space starts a new word, any other
/// token continues the previous one. Bytes are joined before decoding, so characters
/// split across tokens survive.
pub fn tokens_to_words(tokens: &[TokenPiece]) -> Vec<Word> {
    let mut words: Vec<(u64, u64, Vec<u8>)> = Vec::new();
    for t in tokens {
        if t.special || t.bytes.starts_with(b"[_") {
            continue;
        }
        let start = t.t0_cs.max(0) as u64 * 10;
        let end = t.t1_cs.max(0) as u64 * 10;
        match words.last_mut() {
            Some(word) if !t.bytes.starts_with(b" ") => {
                word.1 = word.1.max(end);
                word.2.extend_from_slice(&t.bytes);
            }
            _ => words.push((start, end.max(start), t.bytes.clone())),
        }
    }
    words
        .into_iter()
        .map(|(start_ms, end_ms, bytes)| Word {
            start_ms,
            end_ms,
            text: String::from_utf8_lossy(&bytes).into_owned(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(bytes: &[u8], t0: i64, t1: i64) -> TokenPiece {
        TokenPiece {
            bytes: bytes.to_vec(),
            t0_cs: t0,
            t1_cs: t1,
            special: false,
        }
    }

    #[test]
    fn tokens_to_words_groups_on_leading_space() {
        let words = tokens_to_words(&[
            TokenPiece {
                bytes: b"[_BEG_]".to_vec(),
                t0_cs: 0,
                t1_cs: 0,
                special: true,
            },
            piece(b" Hello", 5, 46),
            piece(b" every", 46, 70),
            piece(b"one", 70, 97),
            piece(b"!", 132, 132),
            TokenPiece {
                bytes: b"[_TT_500]".to_vec(),
                t0_cs: 1000,
                t1_cs: 1000,
                special: true,
            },
        ]);
        let got: Vec<_> = words
            .iter()
            .map(|w| (w.start_ms, w.end_ms, w.text.as_str()))
            .collect();
        assert_eq!(got, vec![(50, 460, " Hello"), (460, 1_320, " everyone!")]);
    }

    #[test]
    fn tokens_to_words_rejoins_split_utf8() {
        // "perché" with the é (C3 A9) split across two tokens, then "città".
        let words = tokens_to_words(&[
            piece(b" perch", 0, 10),
            piece(&[0xC3], 10, 11),
            piece(&[0xA9], 11, 12),
            piece(b"?", 12, 12),
            piece(b" citt", 20, 30),
            piece(&[0xC3, 0xA0], 30, 31),
        ]);
        let texts: Vec<_> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec![" perché?", " città"]);
        assert!(words.iter().all(|w| !w.text.contains('\u{FFFD}')));
    }

    #[test]
    fn tokens_to_words_skips_bracket_markers_and_negative_times() {
        let words = tokens_to_words(&[piece(b"[_SOT_]", -1, -1), piece(b" ok", -3, 4)]);
        assert_eq!(
            words,
            vec![Word {
                start_ms: 0,
                end_ms: 40,
                text: " ok".into()
            }]
        );
    }

    #[test]
    fn most_likely_picks_among_the_candidates() {
        let id = |code| whisper_rs::get_lang_id(code).unwrap();
        let mut probs = vec![0.0; whisper_rs::get_lang_max_id() as usize + 1];
        probs[id("uk") as usize] = 0.5;
        probs[id("ru") as usize] = 0.3;
        probs[id("en") as usize] = 0.1;
        let langs = |codes: &[&str]| codes.iter().map(|c| c.to_string()).collect::<Vec<_>>();
        assert_eq!(most_likely(id("uk"), &probs, &[]).unwrap(), id("uk"));
        assert_eq!(
            most_likely(id("uk"), &probs, &langs(&["en", "ru"])).unwrap(),
            id("ru")
        );
        assert!(most_likely(id("uk"), &probs, &langs(&["xx"])).is_err());
    }

    /// Needs KUTE_TEST_MODELS with ggml-tiny.bin and ggml-silero-v6.2.0.bin.
    #[test]
    #[ignore]
    fn real_models_transcribe_the_fixture() {
        let dir = std::path::PathBuf::from(
            std::env::var("KUTE_TEST_MODELS").expect("set KUTE_TEST_MODELS"),
        );
        let fixture =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/me.wav");
        let samples = crate::audio::read_wav_range(&fixture, 0, 60_000).unwrap();
        let mut vad = SileroVad::load(&dir.join("ggml-silero-v6.2.0.bin")).unwrap();
        let regions = vad.speech_regions(&samples).unwrap();
        assert!(!regions.is_empty());
        assert!(regions[0].start_ms < 1_000, "{regions:?}");
        let mut asr = WhisperTranscriber::load(&dir.join("ggml-tiny.bin"), "tiny", false).unwrap();
        assert_eq!(asr.detect_language(&samples, &[]).unwrap(), "en");
        let among = ["it".to_string(), "en".to_string()];
        assert_eq!(asr.detect_language(&samples, &among).unwrap(), "en");
        let text: String = asr
            .transcribe(&samples, "en")
            .unwrap()
            .iter()
            .map(|s| s.text.clone())
            .collect();
        assert!(text.to_lowercase().contains("differential"), "{text}");
    }
}
