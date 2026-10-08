//! Audio files: kute works on 16 kHz mono 16-bit WAV and archives as Ogg Opus.
//! ffmpeg/ffprobe do the format conversions.

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

pub const SAMPLE_RATE: u32 = 16_000;

pub fn ms_to_samples(ms: u64) -> usize {
    (ms * SAMPLE_RATE as u64 / 1_000) as usize
}

pub fn samples_to_ms(n: usize) -> u64 {
    n as u64 * 1_000 / SAMPLE_RATE as u64
}

pub fn wav_duration_ms(path: &Path) -> Result<u64> {
    Ok(samples_to_ms(open_checked(path)?.duration() as usize))
}

/// Samples in `[start_ms, end_ms)` as f32 in [-1, 1]. Ranges past the end of the file are
/// clamped, so a short or empty result is normal.
pub fn read_wav_range(path: &Path, start_ms: u64, end_ms: u64) -> Result<Vec<f32>> {
    let mut reader = open_checked(path)?;
    let total = reader.duration() as usize;
    let start = ms_to_samples(start_ms).min(total);
    let end = ms_to_samples(end_ms).min(total).max(start);
    reader
        .seek(start as u32)
        .with_context(|| format!("seeking in {}", path.display()))?;
    reader
        .samples::<i16>()
        .take(end - start)
        .map(|s| s.map(|v| v as f32 / 32768.0).map_err(anyhow::Error::from))
        .collect::<Result<Vec<f32>>>()
        .with_context(|| format!("reading {}", path.display()))
}

fn open_checked(path: &Path) -> Result<hound::WavReader<BufReader<File>>> {
    let reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        bail!("{} is not 16 kHz mono 16-bit PCM", path.display());
    }
    Ok(reader)
}

/// Converts any audio ffmpeg can read into kute's working WAV format.
pub fn normalize_to_wav(src: &Path, dst: &Path) -> Result<()> {
    let mut cmd = ffmpeg();
    cmd.arg("-i")
        .arg(src)
        .args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
        ])
        .arg(dst);
    run(cmd, "ffmpeg")
}

/// Encodes a WAV as Ogg Opus at speech quality. `dst` may have any extension.
pub fn encode_opus(wav: &Path, dst: &Path) -> Result<()> {
    let mut cmd = ffmpeg();
    cmd.arg("-i")
        .arg(wav)
        .args([
            "-c:a",
            "libopus",
            "-b:a",
            "24k",
            "-application",
            "voip",
            "-f",
            "opus",
        ])
        .arg(dst);
    run(cmd, "ffmpeg")
}

pub fn probe_duration_ms(path: &Path) -> Result<u64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(path)
        .output()
        .context("running ffprobe (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!(
            "ffprobe failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let secs: f64 = text
        .trim()
        .parse()
        .with_context(|| format!("ffprobe printed {text:?}"))?;
    Ok((secs * 1_000.0).round() as u64)
}

fn ffmpeg() -> Command {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y"]);
    cmd
}

fn run(mut cmd: Command, name: &str) -> Result<()> {
    let out = cmd
        .output()
        .with_context(|| format!("running {name} (is ffmpeg installed?)"))?;
    if !out.status.success() {
        bail!(
            "{name} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// Test helper: writes `seconds` of a 200 Hz square wave of amplitude `value` (first
/// sample is `+value`) as a working-format WAV. A square wave, not a constant, so it
/// survives Opus' high-pass filter in round trips.
#[cfg(test)]
pub fn write_test_wav(path: &Path, seconds: f32, value: i16) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..(seconds * SAMPLE_RATE as f32) as usize {
        w.write_sample(if (i / 40) % 2 == 0 { value } else { -value })
            .unwrap();
    }
    w.finalize().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(path: &Path, n: usize) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..n {
            w.write_sample((i % 1000) as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    #[test]
    fn conversions() {
        assert_eq!(ms_to_samples(1_000), 16_000);
        assert_eq!(samples_to_ms(8_000), 500);
    }

    #[test]
    fn read_range_returns_exact_samples() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.wav");
        ramp(&p, 32_000); // 2 s
        assert_eq!(wav_duration_ms(&p).unwrap(), 2_000);
        let s = read_wav_range(&p, 510, 610).unwrap();
        assert_eq!(s.len(), 1_600);
        assert_eq!(s[0], (8_160 % 1000) as f32 / 32768.0);
    }

    #[test]
    fn read_range_clamps_past_end() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.wav");
        ramp(&p, 16_000); // 1 s
        assert_eq!(read_wav_range(&p, 500, 5_000).unwrap().len(), 8_000);
        assert!(read_wav_range(&p, 3_000, 5_000).unwrap().is_empty());
        assert!(read_wav_range(&p, 800, 200).unwrap().is_empty());
    }

    #[test]
    fn rejects_wrong_format() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        w.write_sample(0i16).unwrap();
        w.write_sample(0i16).unwrap();
        w.finalize().unwrap();
        let err = read_wav_range(&p, 0, 10).unwrap_err();
        assert!(format!("{err:#}").contains("16 kHz mono"), "{err:#}");
    }

    #[test]
    fn ffmpeg_normalize_and_opus_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&src, spec).unwrap();
        for i in 0..(44_100 * 2 * 3) {
            w.write_sample(((i % 200) as i16 - 100) * 50).unwrap();
        }
        w.finalize().unwrap();

        let wav = tmp.path().join("norm.wav");
        normalize_to_wav(&src, &wav).unwrap();
        assert!((wav_duration_ms(&wav).unwrap() as i64 - 3_000).abs() < 50);

        let opus = tmp.path().join("out.opus.tmp");
        encode_opus(&wav, &opus).unwrap();
        assert!((probe_duration_ms(&opus).unwrap() as i64 - 3_000).abs() < 500);
        assert!(std::fs::metadata(&opus).unwrap().len() < std::fs::metadata(&wav).unwrap().len());
    }

    #[test]
    fn ffmpeg_errors_carry_stderr() {
        let tmp = tempfile::tempdir().unwrap();
        let err = normalize_to_wav(&tmp.path().join("missing.wav"), &tmp.path().join("x.wav"))
            .unwrap_err();
        assert!(format!("{err:#}").contains("ffmpeg failed"), "{err:#}");
    }
}
