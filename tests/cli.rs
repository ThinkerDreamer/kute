//! Black-box checks of the CLI that need no models.

use std::process::Command;

fn kute(config: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kute"))
        .arg("--config")
        .arg(config)
        .args(args)
        .output()
        .unwrap()
}

fn setup() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp.path().join("config.toml");
    std::fs::write(
        &config,
        format!("[storage]\ndir = {:?}\n", tmp.path().join("data")),
    )
    .unwrap();
    (tmp, config)
}

#[test]
fn help_lists_commands() {
    let out = Command::new(env!("CARGO_BIN_EXE_kute"))
        .arg("--help")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    for cmd in [
        "import",
        "process",
        "list",
        "show",
        "search",
        "title",
        "summarize",
        "retranscribe",
        "reindex",
        "model",
    ] {
        assert!(text.contains(cmd), "missing {cmd} in:\n{text}");
    }
}

#[test]
fn empty_archive() {
    let (_tmp, config) = setup();
    let out = kute(&config, &["list"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("no sessions"));
    let out = kute(&config, &["search", "anything"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("no matches"));
    assert!(!kute(&config, &["show", "last"]).status.success());
}

#[test]
fn model_list_marks_missing_models() {
    let (_tmp, config) = setup();
    let out = kute(&config, &["model", "list"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("large-v3-turbo")
            && text.contains("(default)")
            && text.contains("silero-v6.2.0"),
        "{text}"
    );
}

#[test]
fn import_without_models_explains_what_to_do() {
    let (tmp, config) = setup();
    let me = tmp.path().join("me.wav");
    std::fs::copy(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/me.wav"),
        &me,
    )
    .unwrap();
    let out = kute(
        &config,
        &[
            "import",
            me.to_str().unwrap(),
            me.to_str().unwrap(),
            "--start",
            "2026-10-08T15:00:00",
        ],
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("kute model download"));
    // Nothing was imported, so nothing is half-processed.
    assert!(String::from_utf8_lossy(&kute(&config, &["list"]).stdout).contains("no sessions"));
}
