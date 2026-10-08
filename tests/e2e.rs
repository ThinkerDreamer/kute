//! Full import -> transcript -> summary -> Opus -> search with the tiny Whisper model.
//! Needs KUTE_TEST_MODELS with ggml-tiny.bin and ggml-silero-v6.2.0.bin:
//!   KUTE_TEST_MODELS=/path/to/models cargo test --test e2e -- --ignored

use std::path::Path;
use std::process::Command;

#[test]
#[ignore]
fn import_transcribes_summarizes_and_compresses() {
    let models = std::env::var("KUTE_TEST_MODELS").expect("set KUTE_TEST_MODELS");
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(data.join("models")).unwrap();
    for file in ["ggml-tiny.bin", "ggml-silero-v6.2.0.bin"] {
        std::os::unix::fs::symlink(
            Path::new(&models).join(file),
            data.join("models").join(file),
        )
        .unwrap();
    }
    let config = tmp.path().join("config.toml");
    std::fs::write(
        &config,
        format!(
            r###"
[storage]
dir = {data:?}
[transcribe]
model = "tiny"
gpu = false
[summary]
backend = "command"
command = ["sh", "-c", 'cat > /dev/null; printf "## Summary\n\nFixture summary.\n"']
min_words = 1
"###
        ),
    )
    .unwrap();

    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let kute = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_kute"))
            .arg("--config")
            .arg(&config)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "kute {args:?} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    kute(&[
        "import",
        fixtures.join("me.wav").to_str().unwrap(),
        fixtures.join("others.wav").to_str().unwrap(),
        "--start",
        "2026-10-08T15:00:00",
        "--title",
        "Fixture call",
    ]);

    let transcript = kute(&["show", "last"]);
    assert!(
        transcript.starts_with("# Fixture call. 2026-10-08 15:00"),
        "{transcript}"
    );
    assert!(
        transcript.to_lowercase().contains("differential"),
        "{transcript}"
    );
    assert!(
        transcript.contains("Others: ") && transcript.to_lowercase().contains("ciao"),
        "{transcript}"
    );
    let me_line = transcript.lines().position(|l| l.contains("Me: ")).unwrap();
    let others_line = transcript
        .lines()
        .position(|l| l.contains("Others: "))
        .unwrap();
    assert!(
        me_line < others_line,
        "English (0-5 s) should come before Italian (6 s+)"
    );

    assert!(kute(&["show", "last", "--summary"]).contains("Fixture summary."));
    assert!(kute(&["search", "differential"]).contains("2026-10-08T150000"));
    assert!(kute(&["list"]).contains("Fixture call"));

    let session = data.join("sessions/2026-10-08T150000");
    assert!(session.join("me.opus").is_file() && !session.join("me.wav").exists());
    let meta = std::fs::read_to_string(session.join("meta.toml")).unwrap();
    assert!(meta.contains("state = \"done\""), "{meta}");
    assert!(
        meta.contains("me = \"en\"") && meta.contains("others = \"it\""),
        "{meta}"
    );

    // Re-transcribing from Opus works and keeps the result.
    kute(&["retranscribe", "last"]);
    assert!(
        kute(&["show", "last"])
            .to_lowercase()
            .contains("differential")
    );
}
