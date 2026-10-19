//! Test fakes shared by the unit tests and the integration tests in `tests/`, which
//! can't see `#[cfg(test)]` items. Not part of kute's API.

use crate::recorder::VadFactory;
use crate::transcribe::{Span, SpeechDetector};
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

/// Speech is any 10 ms frame with a non-zero sample: one span per run of such frames.
pub struct RunVad;

impl SpeechDetector for RunVad {
    fn speech_regions(&mut self, samples: &[f32]) -> Result<Vec<Span>> {
        const FRAME: usize = 160;
        let total_ms = (samples.len() * 10).div_ceil(FRAME) as u64;
        let mut spans: Vec<Span> = Vec::new();
        for (i, frame) in samples.chunks(FRAME).enumerate() {
            if frame.iter().all(|&s| s == 0.0) {
                continue;
            }
            let start_ms = i as u64 * 10;
            let end_ms = (start_ms + 10).min(total_ms);
            match spans.last_mut() {
                Some(last) if last.end_ms == start_ms => last.end_ms = end_ms,
                _ => spans.push(Span { start_ms, end_ms }),
            }
        }
        Ok(spans)
    }
}

pub fn run_vad() -> VadFactory {
    Arc::new(|| Some(Box::new(RunVad) as Box<dyn SpeechDetector>))
}

/// Zeroes the RIFF and `data` chunk sizes of a finished WAV, which is what a crash before
/// the writer's first header flush leaves on disk.
pub fn crash_wav(path: &Path) {
    let mut bytes = std::fs::read(path).unwrap();
    let data = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("no data chunk");
    bytes[4..8].fill(0);
    bytes[data + 4..data + 8].fill(0);
    std::fs::write(path, bytes).unwrap();
}
