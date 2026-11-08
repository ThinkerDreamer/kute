//! `kute doctor --bench`: times the configured Whisper model on the GPU and on the CPU with a
//! short sample, and offers to set `transcribe.gpu` to the faster one.
//!
//! An integrated GPU can lose to a fast desktop CPU, and whisper.cpp quietly runs on the CPU
//! when it finds no usable GPU, so the only way to know is to measure.

use crate::audio;
use crate::config::Config;
use crate::ipc::{self, Request, Response};
use crate::models;
use crate::store::write_atomic;
use crate::transcribe::Transcriber;
use crate::transcribe::whisper::WhisperTranscriber;
use anyhow::{Context, Result, bail};
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Synthetic English speech, the same clip the tests use. About 10 s, so the sample repeats it.
const BUILTIN_SAMPLE: &[u8] = include_bytes!("../tests/fixtures/me.wav");
/// How much audio a run transcribes: one Whisper window.
const SAMPLE_MS: u64 = 30_000;
/// Timed runs per device, after one untimed warm-up run. The fastest counts.
const RUNS: usize = 2;
/// The CPU must be this much faster to win: on a tie the GPU leaves the CPU free for the call
/// that is being recorded.
const CPU_MARGIN: f64 = 0.9;

pub struct Options<'a> {
    pub config_path: Option<&'a Path>,
    /// A recording to time instead of the built-in clip; any format ffmpeg reads.
    pub sample: Option<&'a Path>,
    /// Write the result to the config without asking.
    pub save: bool,
}

pub fn run(opts: Options) -> Result<()> {
    let config = Config::load(opts.config_path)?;
    let Some(backend) = gpu_backend() else {
        bail!("this kute is built without GPU support, so there is nothing to compare");
    };
    refuse_while_busy()?;
    let models_dir = config.data_dir()?.join("models");
    let (model_path, model_name) = models::resolve_whisper(&models_dir, &config.transcribe.model)?;

    let (samples, language) = match opts.sample {
        Some(path) => (
            read_sample(path)?,
            single_language(&config.transcribe.language).unwrap_or_else(|| "auto".into()),
        ),
        None => (builtin_sample()?, "en".to_string()),
    };
    let language = language.as_str();
    let audio_secs = samples.len() as f64 / audio::SAMPLE_RATE as f64;
    println!(
        "Timing {model_name} on {:.0} s of {}, {RUNS} runs each after a warm-up.",
        audio_secs,
        match opts.sample {
            Some(p) => p.display().to_string(),
            None => "built-in speech".into(),
        }
    );

    let Some(device) = gpu_device() else {
        println!(
            "{backend} finds no GPU, so transcription runs on the CPU whatever transcribe.gpu \
             says; nothing to compare."
        );
        return Ok(());
    };
    let gpu_label = format!("GPU ({backend}, {device})");
    let gpu = time_device(
        &gpu_label,
        &model_path,
        &model_name,
        true,
        &samples,
        language,
    )?;
    let cpu = time_device("CPU", &model_path, &model_name, false, &samples, language)?;

    let use_gpu = pick_gpu(gpu, cpu);
    let ratio = gpu.as_secs_f64() / cpu.as_secs_f64();
    println!();
    if use_gpu {
        if ratio <= 1.0 {
            println!("The GPU is {:.1}x as fast as the CPU.", 1.0 / ratio);
        } else {
            println!(
                "The CPU is only {:.0}% faster; the GPU stays the pick, so the CPU stays free \
                 during calls.",
                (ratio - 1.0) * 100.0
            );
        }
    } else {
        println!("The CPU is {ratio:.1}x as fast as the GPU.");
    }
    let best = if use_gpu { "GPU" } else { "CPU" };
    if config.transcribe.gpu == use_gpu {
        println!("transcribe.gpu = {use_gpu} already uses the {best}; nothing to change.");
        return Ok(());
    }

    let path = config_path(opts.config_path)?;
    let save = opts.save
        || (std::io::stdin().is_terminal()
            && ask(&format!(
                "Set transcribe.gpu = {use_gpu} in {}? [y/N] ",
                path.display()
            ))?);
    if !save {
        println!(
            "To use the {best}, set transcribe.gpu = {use_gpu} in {}, or run \
             `kute doctor --bench --save`.",
            path.display()
        );
        return Ok(());
    }
    write_gpu_setting(&path, use_gpu)?;
    println!("Set transcribe.gpu = {use_gpu} in {}.", path.display());
    if daemon_running() {
        println!(
            "The daemon reads the config at start: `systemctl --user restart kute` once no call \
             is being recorded."
        );
    }
    Ok(())
}

fn gpu_backend() -> Option<&'static str> {
    if cfg!(feature = "cuda") {
        Some("CUDA")
    } else if cfg!(feature = "vulkan") {
        Some("Vulkan")
    } else {
        None
    }
}

/// The device whisper.cpp uses for the GPU (the first one), when the backend can name it.
#[cfg(feature = "vulkan")]
fn gpu_device() -> Option<String> {
    whisper_rs::vulkan::list_devices()
        .into_iter()
        .next()
        .map(|d| d.name)
}

#[cfg(not(feature = "vulkan"))]
fn gpu_device() -> Option<String> {
    Path::new("/dev/nvidiactl")
        .exists()
        .then(|| "NVIDIA".to_string())
}

/// A benchmark during a call or a transcription would slow both and time neither fairly.
fn refuse_while_busy() -> Result<()> {
    let Some(Response::Status { status }) = daemon_status() else {
        return Ok(());
    };
    if status.phase != crate::detect::Phase::Idle {
        bail!("the daemon is recording a call; run the benchmark once it is over");
    }
    if status.running_job.is_some() || status.queued > 0 {
        bail!("the daemon is transcribing; run the benchmark once `kute status` shows it idle");
    }
    Ok(())
}

fn daemon_status() -> Option<Response> {
    let path = ipc::socket_path().ok()?;
    ipc::request(&path, &Request::Status).ok().flatten()
}

fn daemon_running() -> bool {
    daemon_status().is_some()
}

fn time_device(
    label: &str,
    model_path: &Path,
    model_name: &str,
    use_gpu: bool,
    samples: &[f32],
    language: &str,
) -> Result<Duration> {
    print!("{label}: loading");
    std::io::stdout().flush().ok();
    let started = Instant::now();
    let mut asr = WhisperTranscriber::load(model_path, model_name, use_gpu)?;
    print!(" ({:.1} s), warming up", started.elapsed().as_secs_f64());
    std::io::stdout().flush().ok();
    asr.transcribe(samples, language)?;
    let mut best = Duration::MAX;
    for _ in 0..RUNS {
        print!(".");
        std::io::stdout().flush().ok();
        let started = Instant::now();
        asr.transcribe(samples, language)?;
        best = best.min(started.elapsed());
    }
    let audio_secs = samples.len() as f64 / audio::SAMPLE_RATE as f64;
    println!(
        " {:.2} s ({:.0}x real time)",
        best.as_secs_f64(),
        audio_secs / best.as_secs_f64()
    );
    Ok(best)
}

/// Whether the GPU should be the default, given the best time on each.
fn pick_gpu(gpu: Duration, cpu: Duration) -> bool {
    cpu.as_secs_f64() >= gpu.as_secs_f64() * CPU_MARGIN
}

/// The built-in clip repeated to fill one sample.
fn builtin_sample() -> Result<Vec<f32>> {
    let reader = hound::WavReader::new(std::io::Cursor::new(BUILTIN_SAMPLE))?;
    let clip: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;
    let want = audio::ms_to_samples(SAMPLE_MS);
    Ok(clip.iter().copied().cycle().take(want).collect())
}

/// The first 30 s of `path`, converted with ffmpeg.
fn read_sample(path: &Path) -> Result<Vec<f32>> {
    // A private folder: the sample may be a recorded call.
    let dir = tempfile::tempdir().context("creating a temporary folder")?;
    let wav = dir.path().join("sample.wav");
    audio::normalize_to_wav_limited(path, &wav, Some(SAMPLE_MS))?;
    let samples = audio::read_wav_range(&wav, 0, SAMPLE_MS)?;
    if samples.len() < audio::ms_to_samples(5_000) {
        bail!(
            "{} is shorter than 5 s; give a longer recording",
            path.display()
        );
    }
    Ok(samples)
}

/// The language to force for a user sample: only when the config names exactly one.
fn single_language(setting: &str) -> Option<String> {
    let mut languages = crate::transcribe::parse_languages(setting).ok()?;
    if languages.len() != 1 || languages[0] == crate::transcribe::ANY_LANGUAGE {
        return None;
    }
    languages.pop()
}

fn config_path(explicit: Option<&Path>) -> Result<PathBuf> {
    match explicit {
        Some(p) => Ok(p.to_path_buf()),
        None => Config::default_path().context("no config folder (is HOME set?)"),
    }
}

fn ask(question: &str) -> Result<bool> {
    print!("{question}");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "Yes"))
}

/// Sets `[transcribe] gpu` in the config file, keeping its comments and layout, and creates
/// the file when there is none.
pub fn write_gpu_setting(path: &Path, gpu: bool) -> Result<()> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let updated = set_gpu(&text, gpu).with_context(|| format!("editing {}", path.display()))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    write_atomic(path, updated.as_bytes())
}

fn set_gpu(text: &str, gpu: bool) -> Result<String> {
    let mut doc: toml_edit::DocumentMut = text.parse()?;
    let table = doc
        .entry("transcribe")
        .or_insert_with(toml_edit::table)
        .as_table_like_mut()
        .context("`transcribe` is not a table")?;
    table.insert("gpu", toml_edit::value(gpu));
    let updated = doc.to_string();
    // Never leave behind a config that kute can't read.
    Config::parse(&updated)?;
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gpu_wins_ties() {
        let ms = Duration::from_millis;
        assert!(pick_gpu(ms(1000), ms(2000)));
        assert!(
            pick_gpu(ms(1000), ms(950)),
            "5% faster is not enough for the CPU"
        );
        assert!(!pick_gpu(ms(1000), ms(800)));
    }

    #[test]
    fn set_gpu_keeps_comments_and_other_keys() {
        let text = "# my config\n[transcribe]\nmodel = \"small\" # fast\ngpu = true\n\n\
                    [summary]\nbackend = \"none\"\n";
        let out = set_gpu(text, false).unwrap();
        assert_eq!(
            out,
            "# my config\n[transcribe]\nmodel = \"small\" # fast\ngpu = false\n\n\
             [summary]\nbackend = \"none\"\n"
        );
    }

    #[test]
    fn set_gpu_adds_the_key_and_the_table() {
        let out = set_gpu("[transcribe]\nmodel = \"small\"\n", false).unwrap();
        assert!(!Config::parse(&out).unwrap().transcribe.gpu);
        assert_eq!(Config::parse(&out).unwrap().transcribe.model, "small");

        let out = set_gpu("[user]\nname = \"Alex\"\n", false).unwrap();
        let config = Config::parse(&out).unwrap();
        assert!(!config.transcribe.gpu);
        assert_eq!(config.user.name, "Alex");

        let out = set_gpu("", true).unwrap();
        assert!(Config::parse(&out).unwrap().transcribe.gpu);
    }

    #[test]
    fn set_gpu_works_on_an_inline_table() {
        let out = set_gpu("transcribe = { model = \"small\" }\n", false).unwrap();
        let config = Config::parse(&out).unwrap();
        assert!(!config.transcribe.gpu);
        assert_eq!(config.transcribe.model, "small");
    }

    #[test]
    fn write_gpu_setting_creates_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kute").join("config.toml");
        write_gpu_setting(&path, false).unwrap();
        let config = Config::load(Some(&path)).unwrap();
        assert!(!config.transcribe.gpu);
    }

    #[test]
    fn the_builtin_sample_fills_one_window() {
        let samples = builtin_sample().unwrap();
        assert_eq!(samples.len(), audio::ms_to_samples(SAMPLE_MS));
        assert!(samples.iter().any(|s| s.abs() > 0.01), "has speech");
    }

    #[test]
    fn only_one_configured_language_is_forced() {
        assert_eq!(single_language("ru").as_deref(), Some("ru"));
        assert_eq!(single_language("auto"), None);
        assert_eq!(single_language("ru,en"), None);
        assert_eq!(single_language("ru en"), None);
        assert_eq!(single_language(" ru "), Some("ru".into()));
    }
}
