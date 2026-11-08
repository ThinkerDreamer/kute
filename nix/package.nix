{
  lib,
  rustPlatform,
  cmake,
  makeWrapper,
  shaderc,
  vulkan-headers,
  vulkan-loader,
  ffmpeg,
  pipewire,
  tzdata,
  vulkanSupport ? true,
}:

rustPlatform.buildRustPackage {
  pname = "kute";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../src
      # The tray icons, built into the binary.
      ../assets
      ../tests
    ];
  };

  cargoLock.lockFile = ../Cargo.lock;

  buildNoDefaultFeatures = !vulkanSupport;

  nativeBuildInputs = [
    cmake
    makeWrapper
    # whisper-rs-sys runs bindgen, which needs libclang.
    rustPlatform.bindgenHook
  ]
  ++ lib.optionals vulkanSupport [ shaderc ];

  buildInputs = lib.optionals vulkanSupport [
    vulkan-headers
    vulkan-loader
  ];

  # cmake is only here for whisper-rs-sys's build script, not for the top-level build.
  dontUseCmakeConfigure = true;

  # Without this whisper.cpp builds with -march=native and the binary dies with SIGILL on any CPU
  # that lacks the build machine's extensions.
  env.GGML_NATIVE = "OFF";

  nativeCheckInputs = [ ffmpeg ];

  # The sandbox has no time zone database, and tests read back times saved as `...[UTC]`.
  preCheck = ''
    export TZDIR=${tzdata}/share/zoneinfo
  '';

  # kute shells out to ffmpeg/ffprobe for audio and to pw-dump/pw-record for PipeWire.
  postInstall = ''
    wrapProgram $out/bin/kute \
      --prefix PATH : ${
        lib.makeBinPath [
          ffmpeg
          pipewire
        ]
      }
  '';

  meta = {
    description = "Records, transcribes and summarizes your calls, locally";
    homepage = "https://github.com/ThinkerDreamer/kute";
    license = with lib.licenses; [
      mit
      asl20
    ];
    mainProgram = "kute";
    platforms = lib.platforms.linux;
  };
}
