# Installing kute

kute runs on Linux desktops with PipeWire. It is one binary, `kute`, plus two model files you
download once with `kute model download` before the first transcription (see [Models](#models)).

## Requirements

- Linux with PipeWire. The daemon uses `pw-dump` and `pw-record`, which most distributions ship
  in the `pipewire` or `pipewire-bin` package.
- `ffmpeg` and `ffprobe`, for decoding imports and compressing audio to the `opus` format.
- A desktop session: the daemon talks to the CLI over a socket in `$XDG_RUNTIME_DIR`, and sends
  desktop notifications.
- To build: Rust 1.89 or newer, a C/C++ toolchain, `cmake` and `clang` (whisper.cpp is compiled
  from source and bound with bindgen).
- For GPU transcription, one of:
  - **Vulkan** (the default build): Vulkan drivers for your GPU plus the build-time headers and
    shader compiler, usually `vulkan-headers` and `shaderc` (Debian/Ubuntu: `libvulkan-dev` and
    `glslc`).
  - **CUDA**: the CUDA toolkit, and build with `--no-default-features --features cuda`.

  Without a GPU backend it runs on the CPU. That works, but a long call takes a while; pick a
  smaller model (see [Models](#models)).

## From crates.io

```bash
cargo install kute                                           # Vulkan
# cargo install kute --no-default-features --features cuda   # CUDA
# cargo install kute --no-default-features                   # CPU only
```

## From source

To build the latest `main` instead of a release:

```bash
git clone https://github.com/ThinkerDreamer/kute
cd kute
cargo install --path .         # same feature flags as above
```

Either way, this puts `kute` in `~/.cargo/bin`. A local build compiles whisper.cpp for your own
CPU (`-march=native`), which is the fastest option. If you build a binary to run on other
machines, set `GGML_NATIVE=OFF` first, or it may crash with "illegal instruction" on a different
CPU.

## With Nix

*Coming soon.* A `flake.nix` is in progress; this section will cover `nix run`, `nix profile
install` and a NixOS/Home Manager snippet once it lands.

## Models

```bash
kute model download          # large-v3-turbo (1.6 GB) and the Silero VAD model (1 MB)
kute model list              # what's available and what's installed
kute model download small    # another Whisper model
```

Models are stored in the `models` folder of kute's data directory (`~/.local/share/kute/models`
unless you set `XDG_DATA_HOME` or `storage.dir`) and checked against known SHA-256 sums. Pick
one in the config with `transcribe.model`; a path to any ggml `.bin` file works too.

| Model | Size | Notes |
|---|---|---|
| `large-v3-turbo` | 1.6 GB | The default. Good accuracy, fast on a GPU. |
| `large-v3-turbo-q5_0` | 574 MB | Quantized turbo, for GPUs with less memory. |
| `large-v3` | 3.1 GB | Larger and slower than turbo. |
| `medium`, `small`, `base`, `tiny` | 1.5 GB to 78 MB | For CPU-only machines; accuracy drops with size. |

## Running the daemon at login

```bash
kute service install
```

writes `~/.config/systemd/user/kute.service`, enables it and starts it. The unit runs the `kute`
binary you ran it with, by the path you ran it from: `~/.cargo/bin/kute` after `cargo install`,
`~/.nix-profile/bin/kute` from a Nix profile, a store path from `nix run`. Its `PATH` is
`~/.local/bin`, `~/.cargo/bin` and the binary's folder, then the `PATH` of the shell you ran the
install from, so the daemon finds `pw-record`, `ffmpeg` and your summarizer command where your
shell does. `kute service install --print` shows the unit without installing it.

Run it again after moving the binary or changing your `PATH`. If the daemon is idle, it restarts
with the new unit; if it is recording or processing, it tells you to restart it yourself
(`systemctl --user restart kute`) once it's done, since a restart mid-call splits the call into
two sessions.

If you wrote a `kute.service` by hand, `kute service install` won't touch it. `--force` replaces
it and keeps yours as `kute.service.bak`.

A systemd user service doesn't see the environment of your shell, so an API key for a
summarizer (e.g. `ANTHROPIC_API_KEY`) goes in `~/.config/kute/env`, which the unit reads if it
exists. Keep it private (`chmod 600`):

```bash
ANTHROPIC_API_KEY=...
```

Run only one daemon at a time. To try `kute daemon` in a terminal, stop the service first.

## Updating

```bash
cargo install kute                     # or, in a clone: git pull && cargo install --path .
systemctl --user restart kute
```

Sessions and the config are kept. If a release changes the search index format, kute rebuilds
the index from the session folders on its own.

## Uninstalling

```bash
kute service uninstall     # stops and disables the service, removes the unit kute wrote
cargo uninstall kute
```

Your recordings, transcripts and models stay in `~/.local/share/kute` (or
`$XDG_DATA_HOME/kute`, or your `storage.dir`)
and the config in `~/.config/kute`. Delete those folders to remove everything.
