//! Known Whisper and VAD models, where they live, and checksum-verified downloads.

use crate::config::expand_tilde;
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct ModelInfo {
    pub name: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

pub const WHISPER_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "tiny",
        file: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        size: 77_691_713,
    },
    ModelInfo {
        name: "base",
        file: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        size: 147_951_465,
    },
    ModelInfo {
        name: "small",
        file: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
        size: 487_601_967,
    },
    ModelInfo {
        name: "medium",
        file: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
        size: 1_533_763_059,
    },
    ModelInfo {
        name: "large-v3",
        file: "ggml-large-v3.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
        size: 3_095_033_483,
    },
    ModelInfo {
        name: "large-v3-turbo",
        file: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        size: 1_624_555_275,
    },
    ModelInfo {
        name: "large-v3-turbo-q5_0",
        file: "ggml-large-v3-turbo-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        size: 574_041_195,
    },
];

pub const VAD_MODEL: ModelInfo = ModelInfo {
    name: "silero-v6.2.0",
    file: "ggml-silero-v6.2.0.bin",
    url: "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
    sha256: "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987",
    size: 885_098,
};

pub fn find_whisper(name: &str) -> Option<&'static ModelInfo> {
    WHISPER_MODELS.iter().find(|m| m.name == name)
}

pub fn path_for(models_dir: &Path, info: &ModelInfo) -> PathBuf {
    models_dir.join(info.file)
}

/// `model` is a known name or a path to a ggml `.bin` file. Returns the file and a name
/// to record in `meta.toml`.
pub fn resolve_whisper(models_dir: &Path, model: &str) -> Result<(PathBuf, String)> {
    let as_path = expand_tilde(Path::new(model));
    if as_path.extension().is_some_and(|e| e == "bin") && as_path.is_file() {
        let name = as_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        return Ok((as_path, name));
    }
    let Some(info) = find_whisper(model) else {
        let known: Vec<_> = WHISPER_MODELS.iter().map(|m| m.name).collect();
        bail!(
            "unknown Whisper model {model:?}; use one of {} or a path to a ggml .bin file",
            known.join(", ")
        );
    };
    let path = path_for(models_dir, info);
    if !path.is_file() {
        bail!("the Whisper model {model} is not downloaded yet; run `kute model download {model}`");
    }
    Ok((path, model.to_string()))
}

pub fn resolve_vad(models_dir: &Path) -> Result<PathBuf> {
    let path = path_for(models_dir, &VAD_MODEL);
    if !path.is_file() {
        bail!("the VAD model is not downloaded yet; run `kute model download`");
    }
    Ok(path)
}

/// Downloads `info` into `models_dir` unless an intact copy is already there.
pub fn download(
    models_dir: &Path,
    info: &ModelInfo,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf> {
    let dest = path_for(models_dir, info);
    if dest.is_file() && sha256_file(&dest)? == info.sha256 {
        progress(info.size, info.size);
        return Ok(dest);
    }
    download_from(info.url, &dest, info.sha256, info.size, progress)?;
    Ok(dest)
}

/// Streams `url` to `<dest>.part`, checks the SHA-256, then renames to `dest`.
pub(crate) fn download_from(
    url: &str,
    dest: &Path,
    sha256: &str,
    size: u64,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<()> {
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("downloading {url}"))?;
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut done = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("downloading {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        done += n as u64;
        progress(done, size);
    }
    file.sync_all()?;
    drop(file);
    let got = hex(&hasher.finalize());
    if got != sha256 {
        let _ = fs::remove_file(&part);
        bail!("checksum mismatch for {url}: expected {sha256}, got {got}");
    }
    fs::rename(&part, dest).with_context(|| format!("moving download to {}", dest.display()))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO_SHA256: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn registry_lookup() {
        assert_eq!(
            find_whisper("large-v3-turbo").unwrap().file,
            "ggml-large-v3-turbo.bin"
        );
        assert!(find_whisper("huge").is_none());
        assert_eq!(VAD_MODEL.file, "ggml-silero-v6.2.0.bin");
    }

    #[test]
    fn resolve_whisper_by_name_path_and_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let err = resolve_whisper(tmp.path(), "tiny").unwrap_err();
        assert!(
            format!("{err:#}").contains("kute model download tiny"),
            "{err:#}"
        );
        assert!(
            format!("{:#}", resolve_whisper(tmp.path(), "huge").unwrap_err()).contains("unknown")
        );

        std::fs::write(tmp.path().join("ggml-tiny.bin"), b"x").unwrap();
        let (path, name) = resolve_whisper(tmp.path(), "tiny").unwrap();
        assert_eq!(
            (path, name.as_str()),
            (tmp.path().join("ggml-tiny.bin"), "tiny")
        );

        let custom = tmp.path().join("my-finetune.bin");
        std::fs::write(&custom, b"x").unwrap();
        let (path, name) = resolve_whisper(tmp.path(), custom.to_str().unwrap()).unwrap();
        assert_eq!((path, name.as_str()), (custom.clone(), "my-finetune"));
    }

    #[test]
    fn sha256_of_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("f");
        std::fs::write(&p, b"hello").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), HELLO_SHA256);
    }

    #[test]
    fn download_verifies_checksum_and_reports_progress() {
        let mut server = mockito::Server::new();
        server.mock("GET", "/good.bin").with_body("hello").create();
        server
            .mock("GET", "/bad.bin")
            .with_body("tampered")
            .create();
        let tmp = tempfile::tempdir().unwrap();

        let dest = tmp.path().join("good.bin");
        let mut seen = Vec::new();
        download_from(
            &format!("{}/good.bin", server.url()),
            &dest,
            HELLO_SHA256,
            5,
            &mut |d, t| seen.push((d, t)),
        )
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello");
        assert_eq!(seen.last(), Some(&(5, 5)));

        let bad = tmp.path().join("bad.bin");
        let err = download_from(
            &format!("{}/bad.bin", server.url()),
            &bad,
            HELLO_SHA256,
            5,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("checksum"), "{err:#}");
        assert!(!bad.exists());
        assert!(!tmp.path().join("bad.bin.part").exists());
    }
}
