# Contributing to kute

Thanks for your interest. Bug reports, fixes and ideas are all welcome. For anything bigger than
a small fix, open an issue first so we can agree on the approach before you spend time on it.

## Building

See [Installing](docs/install.md#requirements) for the system packages. Then:

```bash
cargo build                               # Vulkan (the default)
cargo build --no-default-features         # CPU only: quicker to build, enough for most work
```

The first build compiles whisper.cpp and takes a few minutes.

## Checks

CI runs these, so run them before you push:

```bash
cargo fmt --check
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --no-default-features
```

The tests need `ffmpeg` but no microphone, PipeWire or GPU: captures, speech detection and
Whisper have fakes (`src/testing.rs`), and the daemon is tested end to end with made-up PipeWire
graph snapshots and a fake clock (`tests/daemon.rs`).

A few tests run the real models (the `tiny` Whisper model and Silero VAD). They're marked
`#[ignore]`; point `KUTE_TEST_MODELS` at a folder that has `ggml-tiny.bin` and
`ggml-silero-v6.2.0.bin` (e.g. `~/.local/share/kute/models` after `kute model download tiny`):

```bash
KUTE_TEST_MODELS=~/.local/share/kute/models cargo test --no-default-features -- --ignored
```

One other ignored test, `pw_record_captures_both_tracks`, records 3 s from your real devices and
needs PipeWire. On a machine without it, add `--skip pw_record` after `--ignored`.

Building a binary for another machine? Set `GGML_NATIVE=OFF`, or whisper.cpp is compiled for
your CPU only (CI does this too, see `.github/workflows/ci.yml`).

## Trying the daemon

Only one daemon can run per user. If you have kute installed as a service, stop it
(`systemctl --user stop kute`) before running `cargo run -- daemon`, and start it again
afterwards. To keep test calls apart from your real ones, give the development build its own
config with a scratch `[storage] dir` and pass it with `--config`:
`cargo run -- --config dev.toml daemon`.

## Detector traces

`tests/traces.rs` replays PipeWire traces through the call detector. A trace made with
`kute debug trace` on a real call (see [Troubleshooting](docs/troubleshooting.md#reporting-a-bug))
is the best fixture for a detection bug: put it in `tests/fixtures/traces/`, check it for anything
private, and add a test that asserts what should have happened. `kute debug replay <trace>` shows
what the detector does with it now. Traces named `synthetic-*` are written by hand in the same
format.

## Where things are

- `src/` is one crate. Each module opens with a comment saying what it's for.
- [`docs/superpowers/specs/2026-10-08-kute-design.md`](docs/superpowers/specs/2026-10-08-kute-design.md)
  is the design: goals, the storage format, how detection and conversation splitting work, and
  the results of the PipeWire experiments behind them. Read the relevant section before
  changing behavior, and update it when you change a decision it records.
- `docs/superpowers/plans/` holds the implementation plans the code was built from, task by task.
  They're history: useful to see why something is the way it is, but not kept up to date.
- [`docs/TODO.md`](docs/TODO.md) lists known issues and deferred findings. Good first issues are
  usually there.

## Style

- Format with `cargo fmt`; clippy must pass with `-D warnings`.
- Dates and times use [jiff](https://docs.rs/jiff), not chrono.
- No async runtime: the daemon uses threads and channels (see the design, section 3).
- Prefer extending what's there over adding a parallel version of it.
- Tests and docs use gender-neutral placeholder names (Alex, Jordan) and never real people's
  voices, names or machine details. Test audio is synthesized.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  ([cheatsheet](https://kapeli.com/cheat_sheets/Conventional_Commits.docset/Contents/Resources/Documents/index)):
  a short summary prefixed with a type such as `feat:`, `fix:` or `docs:`, as in `git log`.

## License

By contributing, you agree that your contributions are licensed under the project's terms:
either the [MIT license](LICENSE-MIT) or the [Apache License, Version 2.0](LICENSE-APACHE), at
the user's option.
