//! Records one track: reads PCM from a capture, writes a crash-safe WAV, and runs the
//! speech detector on 10 s blocks so the daemon can split conversations live.

use crate::audio::{SAMPLE_RATE, ms_to_samples, samples_to_ms};
use crate::capture::Capture;
use crate::store::Track;
use crate::time;
use crate::transcribe::SpeechDetector;
use anyhow::{Context, Result, bail};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// The WAV header is rewritten every this many samples (5 s), so a crash loses at most
/// that much of the header's count; `repair_wav_header` recovers the rest.
pub const FLUSH_EVERY: u32 = 80_000;
pub const VAD_BLOCK_MS: u64 = 10_000;
/// 100 ms of S16LE.
const READ_BYTES: usize = 3_200;
/// Gap fill kicks in when the track falls this far behind the wall clock.
const GAP_FILL_SAMPLES: u64 = 8_000;
const SILENCE: [i16; 1_600] = [0; 1_600];

/// A 16 kHz mono 16-bit WAV that keeps a valid header while it grows.
pub struct TrackWriter {
    wav: hound::WavWriter<BufWriter<File>>,
    samples: u64,
    since_flush: u32,
}

impl TrackWriter {
    pub fn create(path: &Path) -> Result<TrackWriter> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let wav = hound::WavWriter::create(path, spec)
            .with_context(|| format!("creating {}", path.display()))?;
        Ok(TrackWriter {
            wav,
            samples: 0,
            since_flush: 0,
        })
    }

    pub fn write(&mut self, samples: &[i16]) -> Result<()> {
        for &s in samples {
            self.wav.write_sample(s)?;
            self.samples += 1;
            self.since_flush += 1;
            if self.since_flush == FLUSH_EVERY {
                self.wav.flush()?;
                self.since_flush = 0;
            }
        }
        Ok(())
    }

    /// Appends silence until the track is `total_samples` long.
    pub fn pad_to(&mut self, total_samples: u64) -> Result<()> {
        while self.samples < total_samples {
            let n = (total_samples - self.samples).min(SILENCE.len() as u64) as usize;
            self.write(&SILENCE[..n])?;
        }
        Ok(())
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }

    pub fn finish(self) -> Result<u64> {
        self.wav.finalize()?;
        Ok(self.samples)
    }
}

/// Fixes the sizes in the header of a WAV whose writer died: everything from the start
/// of the `data` payload to EOF is audio. A trailing partial sample is cut off. Returns
/// the number of samples.
pub fn repair_wav_header(path: &Path) -> Result<u64> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let len = file.metadata()?.len();
    let mut riff = [0u8; 12];
    file.read_exact(&mut riff)
        .with_context(|| format!("{} is too short for a WAV", path.display()))?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        bail!("{} is not a WAV file", path.display());
    }
    // Walk the chunks before `data`; their sizes were written when the file was created.
    let mut pos = 12;
    let mut block_align = None;
    let data_start = loop {
        let mut head = [0u8; 8];
        if pos + 8 > len {
            bail!("{} has no data chunk", path.display());
        }
        file.seek(SeekFrom::Start(pos))?;
        file.read_exact(&mut head)?;
        let size = u32::from_le_bytes(head[4..8].try_into().unwrap()) as u64;
        match &head[0..4] {
            b"data" => break pos + 8,
            b"fmt " => {
                let mut fmt = [0u8; 16];
                file.read_exact(&mut fmt)
                    .with_context(|| format!("{} has a short fmt chunk", path.display()))?;
                block_align = Some(u16::from_le_bytes([fmt[12], fmt[13]]) as u64);
            }
            _ => {}
        }
        // Chunks are padded to an even size.
        pos += 8 + size + size % 2;
    };
    let Some(block_align) = block_align.filter(|&b| b > 0) else {
        bail!("{} has no fmt chunk before its data", path.display());
    };
    let samples = (len - data_start) / block_align;
    let data_bytes = samples * block_align;
    let riff_size = u32::try_from(data_start + data_bytes - 8)
        .with_context(|| format!("{} is too large for a WAV", path.display()))?;
    file.set_len(data_start + data_bytes)?;
    file.seek(SeekFrom::Start(4))?;
    file.write_all(&riff_size.to_le_bytes())?;
    file.seek(SeekFrom::Start(data_start - 4))?;
    file.write_all(&(data_bytes as u32).to_le_bytes())?;
    file.sync_all()?;
    Ok(samples)
}

#[derive(Debug)]
pub enum RecorderEvent {
    FirstSample {
        track: Track,
        at: jiff::Zoned,
    },
    /// Track time.
    Speech {
        track: Track,
        start_ms: u64,
        end_ms: u64,
    },
    /// Speech detection has covered the track up to here (track time).
    Analysed {
        track: Track,
        until_ms: u64,
    },
    Ended {
        track: Track,
        samples: u64,
    },
    Failed {
        track: Track,
        error: String,
    },
}

/// Makes a speech detector inside the track thread (whisper.cpp contexts are not `Send`).
/// `None`: record without live speech detection.
pub type VadFactory = Arc<dyn Fn() -> Option<Box<dyn SpeechDetector>> + Send + Sync>;

pub struct TrackOptions {
    /// Pad with silence when the capture falls behind the wall clock (a stalled device),
    /// so both tracks keep the same time base.
    pub gap_fill: bool,
}

/// Records `capture` into `path` on its own thread until the capture reaches EOF, then
/// stops it. Ends with `Ended`, or `Failed` on any error.
pub fn spawn_track(
    track: Track,
    capture: Capture,
    path: PathBuf,
    vad: VadFactory,
    options: TrackOptions,
    tx: Sender<RecorderEvent>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let stop = capture.stop_handle();
        let mut reader = capture.reader;
        let mut live = LiveVad {
            track,
            detector: vad(),
            block: Vec::new(),
            block_start: 0,
        };
        let result = record(track, &mut reader, &path, &mut live, &options, &tx);
        // Drop the reader first, so a capture still writing gets EPIPE rather than
        // blocking on a full pipe.
        drop(reader);
        stop.stop();
        let _ = tx.send(match result {
            Ok(samples) => RecorderEvent::Ended { track, samples },
            Err(e) => RecorderEvent::Failed {
                track,
                error: format!("{e:#}"),
            },
        });
    })
}

/// Where the track should be after `elapsed_ms` since its first sample, if it has fallen
/// more than 500 ms behind `written` samples.
fn gap_fill_target(elapsed_ms: u64, written: u64) -> Option<u64> {
    let expected = elapsed_ms * SAMPLE_RATE as u64 / 1_000;
    (expected > written + GAP_FILL_SAMPLES).then_some(expected)
}

fn record(
    track: Track,
    reader: &mut dyn Read,
    path: &Path,
    live: &mut LiveVad,
    options: &TrackOptions,
    tx: &Sender<RecorderEvent>,
) -> Result<u64> {
    let mut writer = TrackWriter::create(path)?;
    let mut buf = [0u8; READ_BYTES];
    // Bytes at the start of `buf` left over from the last read: an odd byte at most.
    let mut carry = 0;
    let mut first: Option<Instant> = None;
    let mut samples = Vec::with_capacity(READ_BYTES / 2);
    loop {
        let n = match reader.read(&mut buf[carry..]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e).context("reading the capture"),
        };
        let first = *first.get_or_insert_with(|| {
            let _ = tx.send(RecorderEvent::FirstSample {
                track,
                at: time::now(),
            });
            Instant::now()
        });
        let len = carry + n;
        let whole = len / 2 * 2;
        samples.clear();
        samples.extend(
            buf[..whole]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| i16::from_le_bytes(b)),
        );
        carry = len - whole;
        if carry == 1 {
            buf[0] = buf[whole];
        }
        if options.gap_fill {
            let elapsed_ms = first.elapsed().as_millis() as u64;
            if let Some(target) = gap_fill_target(elapsed_ms, writer.samples()) {
                let before = writer.samples();
                writer.pad_to(target)?;
                live.feed_silence(target - before, tx);
            }
        }
        writer.write(&samples)?;
        live.feed(&samples, tx);
    }
    live.run(tx);
    writer.finish()
}

/// Runs the detector on each `VAD_BLOCK_MS` of written audio.
struct LiveVad {
    track: Track,
    /// Dropped after an error: no more speech events for this track.
    detector: Option<Box<dyn SpeechDetector>>,
    block: Vec<f32>,
    /// Track time of `block[0]`, in samples.
    block_start: u64,
}

impl LiveVad {
    fn feed(&mut self, mut samples: &[i16], tx: &Sender<RecorderEvent>) {
        if self.detector.is_none() {
            return;
        }
        let block_len = ms_to_samples(VAD_BLOCK_MS);
        while !samples.is_empty() {
            let take = (block_len - self.block.len()).min(samples.len());
            self.block
                .extend(samples[..take].iter().map(|&s| s as f32 / 32768.0));
            samples = &samples[take..];
            if self.block.len() == block_len {
                self.run(tx);
            }
        }
    }

    fn feed_silence(&mut self, mut n: u64, tx: &Sender<RecorderEvent>) {
        while n > 0 && self.detector.is_some() {
            let take = n.min(SILENCE.len() as u64);
            self.feed(&SILENCE[..take as usize], tx);
            n -= take;
        }
    }

    /// Analyses the pending block, however short.
    fn run(&mut self, tx: &Sender<RecorderEvent>) {
        let Some(detector) = self.detector.as_mut() else {
            return;
        };
        if self.block.is_empty() {
            return;
        }
        let offset = samples_to_ms(self.block_start as usize);
        match detector.speech_regions(&self.block) {
            Ok(spans) => {
                for span in spans {
                    let _ = tx.send(RecorderEvent::Speech {
                        track: self.track,
                        start_ms: offset + span.start_ms,
                        end_ms: offset + span.end_ms,
                    });
                }
                self.block_start += self.block.len() as u64;
                let _ = tx.send(RecorderEvent::Analysed {
                    track: self.track,
                    until_ms: samples_to_ms(self.block_start as usize),
                });
            }
            Err(e) => {
                eprintln!(
                    "kute: speech detection failed on the {} track, continuing without it: {e:#}",
                    self.track.as_str()
                );
                self.detector = None;
            }
        }
        self.block.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{Capture, CaptureSource, PwRecord};
    use crate::testing::{crash_wav, run_vad};
    use std::io::Write;
    use std::sync::mpsc;

    fn wav_samples(path: &Path) -> u32 {
        hound::WavReader::open(path).unwrap().duration()
    }

    #[test]
    fn writer_keeps_a_valid_header_while_writing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("me.wav");
        let mut w = TrackWriter::create(&path).unwrap();
        w.write(&vec![7; 100_000]).unwrap();
        assert_eq!(w.samples(), 100_000);
        assert!(wav_samples(&path) >= FLUSH_EVERY);
        assert_eq!(w.finish().unwrap(), 100_000);
        assert_eq!(wav_samples(&path), 100_000);
    }

    #[test]
    fn pad_to_appends_zeros() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("me.wav");
        let mut w = TrackWriter::create(&path).unwrap();
        w.write(&[5; 10]).unwrap();
        w.pad_to(30).unwrap();
        w.pad_to(20).unwrap();
        assert_eq!(w.finish().unwrap(), 30);
        let samples: Vec<i16> = hound::WavReader::open(&path)
            .unwrap()
            .samples::<i16>()
            .map(|s| s.unwrap())
            .collect();
        assert_eq!(&samples[..10], &[5; 10]);
        assert_eq!(&samples[10..], &[0; 20]);
    }

    #[test]
    fn repair_wav_header_fixes_a_crashed_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("me.wav");
        let mut w = TrackWriter::create(&path).unwrap();
        w.write(&vec![7; 50_000]).unwrap();
        w.finish().unwrap();
        crash_wav(&path);
        assert_eq!(repair_wav_header(&path).unwrap(), 50_000);
        assert_eq!(wav_samples(&path), 50_000);
    }

    #[test]
    fn repair_wav_header_drops_a_partial_sample() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("me.wav");
        let mut w = TrackWriter::create(&path).unwrap();
        w.write(&[7; 10]).unwrap();
        w.finish().unwrap();
        crash_wav(&path);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[1])
            .unwrap();
        assert_eq!(repair_wav_header(&path).unwrap(), 10);
        assert_eq!(wav_samples(&path), 10);
    }

    #[test]
    fn repair_wav_header_rejects_a_file_without_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("me.wav");
        std::fs::write(&path, b"RIFF\0\0\0\0WAVEjunk\x04\0\0\0abcd").unwrap();
        let err = repair_wav_header(&path).unwrap_err();
        assert!(format!("{err:#}").contains("no data chunk"), "{err:#}");
        std::fs::write(&path, b"RIFF\0\0\0\0WAVEdata\0\0\0\0abcd").unwrap();
        let err = repair_wav_header(&path).unwrap_err();
        assert!(format!("{err:#}").contains("no fmt chunk"), "{err:#}");
        std::fs::write(&path, b"nope").unwrap();
        assert!(repair_wav_header(&path).is_err());
    }

    #[test]
    fn gap_fill_keeps_track_time_aligned() {
        assert_eq!(gap_fill_target(1_000, 16_000), None);
        assert_eq!(gap_fill_target(2_000, 16_000), Some(32_000));
        assert_eq!(gap_fill_target(1_400, 16_000), None);
    }

    fn pcm(seconds: u32, value: i16) -> Vec<u8> {
        (0..seconds * 16_000)
            .flat_map(|_| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn track_thread_writes_wav_and_reports_speech() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("me.wav");
        let (reader, mut writer) = std::io::pipe().unwrap();
        let (tx, rx) = mpsc::channel();
        let handle = spawn_track(
            Track::Me,
            Capture::new(Box::new(reader), || {}),
            path.clone(),
            run_vad(),
            TrackOptions { gap_fill: false },
            tx,
        );
        // An odd-sized first write splits a sample across reads.
        let mut bytes = pcm(3, 0);
        bytes.extend(pcm(2, 1_000));
        bytes.extend(pcm(20, 0));
        writer.write_all(&bytes[..3]).unwrap();
        writer.write_all(&bytes[3..]).unwrap();
        drop(writer);
        handle.join().unwrap();

        let events: Vec<RecorderEvent> = rx.iter().collect();
        let first = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    RecorderEvent::FirstSample {
                        track: Track::Me,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(first, 1, "{events:?}");
        let speech: Vec<(u64, u64)> = events
            .iter()
            .filter_map(|e| match e {
                RecorderEvent::Speech {
                    track: Track::Me,
                    start_ms,
                    end_ms,
                } => Some((*start_ms, *end_ms)),
                _ => None,
            })
            .collect();
        assert_eq!(speech.len(), 1, "{events:?}");
        assert!(speech[0].0.abs_diff(3_000) <= 10, "{speech:?}");
        assert!(speech[0].1.abs_diff(5_000) <= 10, "{speech:?}");
        let analysed: Vec<u64> = events
            .iter()
            .filter_map(|e| match e {
                RecorderEvent::Analysed {
                    track: Track::Me,
                    until_ms,
                } => Some(*until_ms),
                _ => None,
            })
            .collect();
        assert_eq!(analysed, [10_000, 20_000, 25_000]);
        assert!(
            matches!(
                events.last(),
                Some(RecorderEvent::Ended {
                    track: Track::Me,
                    samples: 400_000
                })
            ),
            "{events:?}"
        );
        assert_eq!(wav_samples(&path), 400_000);
    }

    #[test]
    fn track_thread_without_vad_sends_no_speech_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("others.wav");
        let (tx, rx) = mpsc::channel();
        let bytes = pcm(12, 1_000);
        let handle = spawn_track(
            Track::Others,
            Capture::new(Box::new(std::io::Cursor::new(bytes)), || {}),
            path.clone(),
            Arc::new(|| None),
            TrackOptions { gap_fill: false },
            tx,
        );
        handle.join().unwrap();
        let events: Vec<RecorderEvent> = rx.iter().collect();
        assert_eq!(events.len(), 2, "{events:?}");
        assert!(matches!(
            events[0],
            RecorderEvent::FirstSample {
                track: Track::Others,
                ..
            }
        ));
        assert!(matches!(
            events[1],
            RecorderEvent::Ended {
                track: Track::Others,
                samples: 192_000
            }
        ));
    }

    #[test]
    fn track_thread_reports_a_failed_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("me.wav");
        let (tx, rx) = mpsc::channel();
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stopped.clone();
        let handle = spawn_track(
            Track::Me,
            Capture::new(Box::new(std::io::Cursor::new(pcm(1, 0))), move || {
                s.store(true, std::sync::atomic::Ordering::SeqCst)
            }),
            path,
            run_vad(),
            TrackOptions { gap_fill: true },
            tx,
        );
        handle.join().unwrap();
        let events: Vec<RecorderEvent> = rx.iter().collect();
        assert!(
            matches!(
                &events[..],
                [RecorderEvent::Failed {
                    track: Track::Me,
                    ..
                }]
            ),
            "{events:?}"
        );
        assert!(stopped.load(std::sync::atomic::Ordering::SeqCst));
    }

    /// Needs PipeWire: records 3 s from the default devices.
    #[test]
    #[ignore]
    fn pw_record_captures_both_tracks() {
        let dir = tempfile::tempdir().unwrap();
        let (tx, rx) = mpsc::channel();
        let mut running = Vec::new();
        for track in Track::BOTH {
            let capture = PwRecord.open(track, None).unwrap();
            let stop = capture.stop_handle();
            let path = dir.path().join(format!("{}.wav", track.as_str()));
            let handle = spawn_track(
                track,
                capture,
                path.clone(),
                Arc::new(|| None),
                TrackOptions { gap_fill: true },
                tx.clone(),
            );
            running.push((stop, handle, path));
        }
        drop(tx);
        std::thread::sleep(std::time::Duration::from_secs(3));
        for (stop, handle, path) in running {
            stop.stop();
            handle.join().unwrap();
            let n = wav_samples(&path);
            assert!((44_000..=50_000).contains(&n), "{}: {n}", path.display());
        }
        let events: Vec<RecorderEvent> = rx.iter().collect();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, RecorderEvent::Failed { .. })),
            "{events:?}"
        );
    }
}
