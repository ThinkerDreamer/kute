# Troubleshooting

Start with these two:

```bash
kute status                         # is the daemon running, what is it doing, what's queued
journalctl --user -u kute -n 100    # its recent log, if it runs as a systemd service
```

## A call wasn't recorded

- **Is the daemon running?** `kute status` says so. If you started it in a terminal, it stops
  when the terminal closes; see [running it at login](install.md#running-the-daemon-at-login).
- **Is it paused?** `kute status` shows `paused: yes` after `kute pause`; `kute resume` turns
  detection back on.
- **Was the call shorter than `min_duration`** (20 s by default)? Shorter mic use counts as a
  mic check and is dropped.
- **Is the app in `ignore_apps`, or missing from `allow_apps`?** Check the
  [config](configuration.md#detect).
- **Does the log say it lost the PipeWire graph?** kute needs `pw-dump` and `pw-record` on the
  daemon's `PATH`. Under systemd, that's the `Environment=PATH=...` line in the unit.

To record by hand regardless of detection: `kute record "Title"`, then `kute stop`.

## A call was recorded that shouldn't have been

Something kept the microphone open for longer than `min_duration`, for example a browser tab, a
game's voice chat or a dictation tool. `kute discard` deletes the current session. Add the app to
`ignore_apps` so it doesn't happen again.

## A session keeps going after the call ended

The session ends `stop_grace` (15 s) after no app has the microphone open any more. Some apps keep
it open after a call (a browser tab left on the call's page, a video room). Close the tab or app,
or `kute stop`.

## My voice is in the transcript while I was muted

The call app muted you for the other participants but kept the microphone open, so kute still
heard you. Mute at the system level (your desktop's microphone icon) instead, or as well.

## Everyone else's voice is also on my track

You're using speakers, and the microphone picks up the other side. kute assumes headphones; with
speakers, the "Me" track contains both and the transcript will repeat lines.

## Transcription is slow

- Check that it's using the GPU: a CPU-only build (`--no-default-features`), or
  `transcribe.gpu = false`, is several times slower. A GPU build also falls back to the CPU
  when it finds no usable GPU at run time, for example without Vulkan drivers (see
  [Installing](install.md#requirements)).
- Try a smaller model, e.g. `large-v3-turbo-q5_0` or `small`.

## Transcription failed

`kute status` shows how many sessions are waiting to retry (`retrying:`), and the daemon's log
says why each one failed. The daemon retries with a backoff. Once the problem is fixed (a missing
model, a full disk), find the session in `kute list`, where it still shows as `processing`, and
run `kute process <id>` to finish it right away.

## No summaries

- Summaries are off until `summary.backend` is set. See [Configuration](configuration.md#summary).
- Conversations shorter than `summary.min_words` (80 words) are skipped on purpose.
- The log says why a summary failed: a command not found (it isn't on the daemon's `PATH`), an
  API key variable that isn't set (systemd doesn't see your shell's environment), a timeout.
- Run `kute summarize last` to try again once it's fixed.

## The "Summary ready" notification doesn't open the file

Set `notify.open` to an editor that's installed, e.g. `["kate"]`, and restart the daemon.

## `kute` says it can't reach the daemon or `XDG_RUNTIME_DIR` is not set

The CLI and the daemon talk over `$XDG_RUNTIME_DIR/kute.sock`, which exists only in a desktop
login session. Over SSH or in a container, commands that need the daemon won't work. Without a
daemon running, the commands that change sessions (`import`, `summarize`, ...) do the work
themselves.

## Reporting a bug

Open an issue with what you did, what you expected, what happened, the output of `kute status`,
and the relevant part of the log. Remove names and anything else private from transcripts and
logs before you post them.
