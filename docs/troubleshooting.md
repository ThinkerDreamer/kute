# Troubleshooting

Start with these three:

```bash
kute doctor                         # is everything kute needs installed and set up
kute status                         # is the daemon running, what is it doing, what's queued
journalctl --user -u kute -n 100    # its recent log, if it runs as a systemd service
```

`kute doctor` prints one line per check: `ok`, `WARN` (kute runs, but something won't work) or
`FAIL` (kute can't record or transcribe), each with what to do. It checks programs on your
shell's `PATH` and, when the service is installed, on the service's `PATH` too, since that's the
one the daemon uses.

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

To record by hand regardless of detection: `kute record "Title"`, then `kute stop`. A manual
recording that a call joins ends with the call unless `detect.end_manual = false`.

If none of that explains it, record what PipeWire showed during a call and replay it through
the detector; see [Reporting a bug](#reporting-a-bug).

## A call was recorded that shouldn't have been

Something kept the microphone open for longer than `min_duration`, for example a browser tab, a
game's voice chat or a dictation tool. `kute discard` deletes the current session. Add the app to
`ignore_apps` so it doesn't happen again.

## A session keeps going after the call ended

The session ends `stop_grace` (15 s) after no app has the microphone open any more. Some apps keep
it open after a call (a browser tab left on the call's page, a video room). Close the tab or app,
or `kute stop`.

Recorders that hold the microphone for their own sake, such as OBS, don't count (see
`ignore_recorders` in the [config](configuration.md#detect)). If another app keeps the
microphone open without being part of the call, add it to `ignore_apps`.

## My voice is in the transcript while I was muted

The call app muted you for the other participants but kept the microphone open, so kute still
heard you. Mute at the system level (your desktop's microphone icon) instead, or as well.

## Everyone else's voice is also on my track

You're using speakers, and the microphone picks up the other side. kute assumes headphones; with
speakers, the "Me" track contains both and the transcript will repeat lines.

## Transcription is slow

- Check that it's using the GPU: a CPU-only build (`--no-default-features`), or
  `transcribe.gpu = false`, is several times slower on most machines. With `transcribe.gpu`
  unset, kute uses the CPU when the only GPU is integrated (`kute doctor` says so). A GPU
  build also falls back to the CPU when it finds no usable GPU at run time, for example without Vulkan drivers (see
  [Installing](install.md#requirements)).
- Measure it: `kute doctor --bench` times the model on the GPU and on the CPU with a 30 s clip
  and offers to set `transcribe.gpu` to the faster one (`--save` sets it without asking,
  `--sample FILE` times your own recording). An integrated GPU can be slower than a fast CPU.
  Restart the daemon afterwards.
- Try a smaller model, e.g. `large-v3-turbo-q5_0` or `small`.

## Transcription failed

`kute status` shows how many sessions are waiting to retry (`retrying:`), and the daemon's log
says why each one failed. The daemon retries with a backoff. Once the problem is fixed (a missing
model, a full disk), find the session in `kute list`, where it still shows as `processing`, and
run `kute process <id>` to finish it right away.

## No summaries

- Summaries are off until `summary.backend` is set. See [Configuration](configuration.md#summary).
- Conversations shorter than `summary.min_words` (80 words) are skipped on purpose. `kute list`
  names them under their session, with their word count, and `meta.toml` records the reason as
  `skip_reason`.
- The log says why a summary failed: a command not found (it isn't on the daemon's `PATH`), an
  API key variable that isn't set (systemd doesn't see your shell's environment), a timeout.
- Run `kute summarize last` to try again once it's fixed.

## The "Summary ready" notification doesn't open the file

The log (`journalctl --user -u kute`) says why after each click, e.g. `xdg-open ... failed`.

- **`no DISPLAY environment variable specified` or `no method available`:** the daemon started
  before your desktop session and has no display. kute takes the display from
  `systemctl --user show-environment` on each click, which your desktop fills in when it starts;
  if that's empty too, your desktop doesn't tell systemd (some window managers need
  `systemctl --user import-environment DISPLAY WAYLAND_DISPLAY` in their startup).
- **The editor isn't installed:** set `notify.open` to one that is, e.g. `["kate"]`, and restart
  the daemon.
- **The button is missing:** some desktops (KDE Plasma among them) drop a notification's buttons
  once it has moved from the popup to the history. Open the summary with
  `kute show last --summary` or `kute open last`.

## There's no tray icon

- **Is the daemon running, with the tray on?** The icon belongs to the daemon (`[tray] enabled`,
  on by default). The log says "no tray host to show the icon" while there's no panel to show it
  in; it appears once there is one.
- **Look in the hidden icons.** While no call is being recorded, the icon is marked passive, and
  Plasma puts it in the panel's overflow (the arrow). Right-click the arrow, *Configure System
  Tray*, and set kute to *Always shown* if you want it in view.
- **GNOME has no tray of its own.** Install the AppIndicator and KStatusNotifierItem Support
  extension. `kute doctor` says whether a tray host is running.
- **A blank or generic icon?** kute writes its icons to `~/.local/share/kute/icons` when the
  daemon starts and tells the tray host to look there. It also adds them to
  `~/.local/share/icons/hicolor/scalable/apps`, where hosts that ignore kute's folder (Cinnamon
  before 6.8) find them by name. A panel that was already running may need a restart to notice
  them: on Cinnamon, press Alt+F2, type `r` and Enter, or log out and back in. If kute's folder
  can't be written, kute uses the theme's mic, red dot and pause icons instead (the log says why).

## `kute` says it can't reach the daemon or `XDG_RUNTIME_DIR` is not set

The CLI and the daemon talk over `$XDG_RUNTIME_DIR/kute.sock`, which exists only in a desktop
login session. Over SSH or in a container, commands that need the daemon won't work. Without a
daemon running, the commands that change sessions (`import`, `summarize`, ...) do the work
themselves.

## Reporting a bug

Open an issue with what you did, what you expected, what happened, the output of `kute doctor`
and `kute status`, and the relevant part of the log. Remove names and anything else private
from transcripts and logs before you post them.

For a call that was detected wrongly (not recorded, recorded when it shouldn't have been, split
in two, never stopping), a trace of the PipeWire graph shows what kute saw:

```bash
kute debug trace                    # records until Ctrl-C, to kute-trace-<time>.jsonl
kute debug trace --duration 2m --out call.jsonl
kute debug replay call.jsonl        # what the detector does with it, with your [detect] settings
```

Start the trace, reproduce the problem (join and leave the call, open the app that misbehaves),
then stop it. It records which audio streams apps open and close, not any audio. Device serial
numbers, Bluetooth addresses, your user and host names and your home folder are replaced before
anything is written; app names (Firefox, Zoom) and device names (your headset's model) are kept.
Read it before you attach it: it's plain text.
