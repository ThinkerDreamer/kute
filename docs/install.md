# Installing kute

kute runs on Linux desktops with PipeWire. It is one binary, `kute`, plus two model files you
download once with `kute model download` before the first transcription (see [Models](#models)).

## Requirements

- Linux with PipeWire. The daemon uses `pw-dump` and `pw-record`, which most distributions ship
  in the `pipewire` or `pipewire-bin` package.
- `ffmpeg` and `ffprobe`, for decoding imports and compressing audio to the `opus` format.
- A desktop session: the daemon talks to the CLI over a socket in `$XDG_RUNTIME_DIR`, and sends
  desktop notifications.
- Optional: `xdg-open` (usually in `xdg-utils`) to open summaries and folders from
  notifications, and `kdialog` or `zenity` for the tray's dialogs (title, names, languages,
  notes).
- To build: Rust 1.89 or newer, a C/C++ toolchain, `cmake` and `clang` (whisper.cpp is compiled
  from source and bound with bindgen).
- For GPU transcription, one of:
  - **Vulkan** (the default build): Vulkan drivers for your GPU plus the build-time headers and
    shader compiler, usually `vulkan-headers` and `shaderc` (Debian/Ubuntu: `libvulkan-dev` and
    `glslc`).
  - **CUDA**: the CUDA toolkit, and build with `--no-default-features --features cuda`.

  Without a GPU backend it runs on the CPU. That works, but a long call takes a while; pick a
  smaller model (see [Models](#models)). A GPU build also uses the CPU by default when the only
  GPU is integrated, since a desktop CPU is often faster than it; `kute doctor --bench` measures
  both, and `transcribe.gpu` in the [config](configuration.md#transcribe) overrides it.

## Prebuilt binary

Each [release](https://github.com/ThinkerDreamer/kute/releases) has a tarball,
`kute-<version>-x86_64-ubuntu24.04.tar.gz`, built on Ubuntu 24.04 with Vulkan and for any x86-64
CPU with AVX2. It runs on Ubuntu 24.04 and newer and on systems based on them, such as Linux
Mint 22, but not on older ones (their glibc is too old). Install the runtime packages, then the
binary:

```bash
sudo apt install pipewire-bin ffmpeg libvulkan1 mesa-vulkan-drivers xdg-utils zenity
tar xzf kute-<version>-x86_64-ubuntu24.04.tar.gz
cd kute-<version>-x86_64-ubuntu24.04
sha256sum -c SHA256SUMS
install -D -m 755 kute ~/.local/bin/kute
```

`libvulkan1` is needed for the binary to start at all, even when it transcribes on the CPU. If
`kute` isn't found afterwards, log out and back in: some distributions add `~/.local/bin` to
`PATH` only when it exists at login.

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

The repository is a flake with two packages: `kute` (the default, with Vulkan) and `kute-cpu`.
Both build with `GGML_NATIVE=OFF` and find `ffmpeg` and the PipeWire tools on their own.

```bash
nix run github:ThinkerDreamer/kute -- doctor
nix profile install github:ThinkerDreamer/kute             # or ...#kute-cpu
```

Then download the models and install the service as below; the unit runs
`~/.nix-profile/bin/kute`, so it keeps working after `nix profile upgrade`. In a NixOS or Home
Manager config that has this flake as an input named `kute`, add
`inputs.kute.packages.${pkgs.system}.default` to `environment.systemPackages` or
`home.packages` instead. `kute doctor` says whether the build found your GPU.

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
cargo uninstall kute       # or delete ~/.local/bin/kute, or nix profile remove kute
```

Your recordings, transcripts and models stay in `~/.local/share/kute` (or
`$XDG_DATA_HOME/kute`, or your `storage.dir`)
and the config in `~/.config/kute`. Delete those folders to remove everything. The tray icons
kute added to `~/.local/share/icons/hicolor/scalable/apps` (`kute-*-symbolic.svg`) can go too.
