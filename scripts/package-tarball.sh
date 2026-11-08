#!/usr/bin/env bash
# Builds a portable release binary and packs it as
#   dist/kute-<version>-x86_64-<distro>.tar.gz
# with a SHA256SUMS file inside. On a commit that isn't tagged v<version>, the short commit hash
# goes after the version (kute-0.2.0-1a2b3c4-...), so a test build is never mistaken for a release.
#
# Usage: scripts/package-tarball.sh [distro-label]   (default: ubuntu24.04)
set -euo pipefail

cd "$(dirname "$0")/.."
distro="${1:-ubuntu24.04}"
arch="$(uname -m)"

version="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n1)"
[ -n "$version" ] || { echo "no version in Cargo.toml" >&2; exit 1; }

label="$version"
if [ "$(git describe --tags --exact-match 2>/dev/null)" != "v$version" ] || [ -n "$(git status --porcelain)" ]; then
  label="$version-$(git rev-parse --short=7 HEAD)"
fi

# Without GGML_NATIVE=OFF whisper.cpp builds for this machine's CPU, and the binary dies with
# SIGILL elsewhere.
GGML_NATIVE=OFF cargo build --release --locked

name="kute-$label-$arch-$distro"
rm -rf "dist/$name" "dist/$name.tar.gz"
mkdir -p "dist/$name"
install -m 755 target/release/kute "dist/$name/kute"
cp README.md CHANGELOG.md LICENSE-MIT LICENSE-APACHE "dist/$name/"
(cd "dist/$name" && sha256sum kute > SHA256SUMS)
tar -C dist -czf "dist/$name.tar.gz" "$name"
echo "dist/$name.tar.gz"
