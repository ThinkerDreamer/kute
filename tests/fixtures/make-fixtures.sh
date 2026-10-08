#!/usr/bin/env bash
# Regenerates the synthetic speech fixtures (no real voices). Needs espeak-ng and ffmpeg.
# me.wav: English, then silence. others.wav: silence, then Italian.
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ff() { ffmpeg -nostdin -hide_banner -loglevel error -y "$@"; }
espeak-ng -v en -s 150 -w "$tmp/en.wav" "Hello everyone. Today we are going to talk about the differential download feature."
espeak-ng -v it -s 150 -w "$tmp/it.wav" "Ciao a tutti. Oggi parliamo della nuova versione del programma."
ff -i "$tmp/en.wav" -ar 16000 -ac 1 "$tmp/en16.wav"
ff -i "$tmp/it.wav" -ar 16000 -ac 1 "$tmp/it16.wav"
ff -f lavfi -t 6 -i anullsrc=r=16000:cl=mono "$tmp/sil6.wav"
ff -f lavfi -t 4 -i anullsrc=r=16000:cl=mono "$tmp/sil4.wav"
ff -i "$tmp/en16.wav" -i "$tmp/sil4.wav" -filter_complex '[0][1]concat=n=2:v=0:a=1' -ar 16000 -ac 1 -c:a pcm_s16le me.wav
ff -i "$tmp/sil6.wav" -i "$tmp/it16.wav" -filter_complex '[0][1]concat=n=2:v=0:a=1' -ar 16000 -ac 1 -c:a pcm_s16le others.wav
