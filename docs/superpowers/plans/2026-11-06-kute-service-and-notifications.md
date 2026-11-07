# kute plan 4a: service install and notifications

**Spec:** `docs/superpowers/specs/2026-10-08-kute-design.md`, sections 5 (Recording state),
6 (Summarization, Errors) and 8 (`service install`). Plan 4b covers the tray icon, `kute doctor`
and `kute debug trace`.

**Goal:** the daemon installs itself as a systemd user service, and tells the user what it is
doing through desktop notifications: a call is being recorded (Stop / Discard), summaries are
ready (open them), and something failed.

## `kute service install` / `kute service uninstall` (`src/service.rs`)

- Runs before the config and the store are loaded, so a broken config doesn't block it.
- **Which binary:** the running one. `current_exe()` resolves links, which would pin a Nix
  store path that the next update garbage-collects, so when `argv[0]` (looked up on `PATH` if it
  is a bare name) leads to the same file, that path is used instead: `~/.nix-profile/bin/kute`,
  `~/.cargo/bin/kute`. Otherwise the resolved binary.
- **`PATH`:** `~/.local/bin`, `~/.cargo/bin`, the binary's folder, then the installing shell's
  `PATH`, deduplicated, relative entries dropped, `/usr/bin` last if missing. A fixed list would
  miss `/run/current-system/sw/bin` on NixOS, where `pw-record` and `ffmpeg` live.
- **The unit:** `ExecStart=<binary> daemon`, `Restart=on-failure`, `TimeoutStopSec=10`,
  `EnvironmentFile=-%h/.config/kute/env` (optional, for API keys), `WantedBy=default.target`.
  Values are quoted and `%` doubled for systemd. The first line is a marker comment.
- **An existing unit:** identical → nothing written; has the marker → rewritten; no marker
  (hand-written) → refused unless `--force`, which moves it to `kute.service.bak`.
- **Then:** `daemon-reload`, `enable`, and:
  - service inactive and nothing on the socket → `start`;
  - service inactive but a daemon answers (running in a terminal) → say so, don't start;
  - service active, unit unchanged → nothing;
  - service active, unit changed, daemon idle with no running job → `restart`;
  - otherwise → don't restart (a restart mid-call splits the session), print the command.
- `--print` prints the unit and does nothing else.
- `uninstall`: `disable --now`, and removes the file only if it has the marker.

## Notifications (`src/notify.rs`, `src/daemon.rs`)

`Note` has a title (plain text), a body (markup, escaped), an optional click on the body, buttons,
and an optional key. A `Click` names the session by ID and is resolved at click time
(`Store::resolve`, then the ID in each `meta.toml`, so a folder renamed by hand after the
note went up still opens).
`Notifier::withdraw(key)` takes a keyed note back.

| When | Note | Click / buttons |
|---|---|---|
| `Action::Commit` (Tentative → Recording, or `kute record`) | "Recording: <app>" ("Recording" when manual), key `recording` | Stop, Discard |
| `Action::Finish` / `Action::Discard` | withdraws `recording` | |
| A job wrote summaries | "Summary ready: <title or app>" (unchanged) | body and "Open summary": `summary.md` |
| A job left conversations failed or compression failed | "Processing failed: <title or app>", one line per problem, then `kute process <id>` | body: session folder |
| A job failed outright (e.g. missing model) | "kute: <job> failed" with the error | |
| A recording ended on a failure (disk full, capture died) | "Recording stopped: <app>" | body: session folder, if kept |

- A failed transcription is announced only once it has no retries left (`jobs::retries_left`);
  summaries and compression at once. Only the conversations the job itself left failed count,
  so earlier failures aren't announced again. `jobs::Problems` carries them: in `Outcome`
  (compression) and in `ConversationsFailed` (conversations and compression); the worker copies
  them into `WorkerDone::problems`.
- Stop and Discard reach the daemon as `Event::Click`; the daemon acts only if that session is
  still the live one.
- **Opening a file** (`notify.open`, default `xdg-open`) gets `DISPLAY`, `WAYLAND_DISPLAY`,
  `XDG_CURRENT_DESKTOP` and the like from `systemctl --user show-environment` at click time. A
  service started at boot (`default.target`) starts before the desktop session and never had
  them, so `xdg-open` failed with "no DISPLAY environment variable specified" and the click did
  nothing. The opener's failure is now logged.
- Withdrawing: notify-rust only closes a notification through the handle the waiting thread
  holds, so `withdraw` shows a replacement with the same ID and closes that. A keyed note is registered
  before its thread shows it, so a withdraw that comes first marks it, and it is closed as soon
  as it is up.

## Tests

- `service`: unit text, quoting, `PATH` order and deduplication, link vs resolved binary,
  existing-unit decisions, restart decisions (idle, recording, tentative, running job, foreign).
- `notify`: failure and recording notes, `cut`, parsing `show-environment`.
- `tests/daemon.rs`: the recording note and its withdrawal; Stop and Discard from the note, and a
  stale button ignored; a transcription failure announced only after its last retry; summary and
  compression failures; a job that fails outright.

## Not done here

- KDE Plasma drops a notification's buttons once it moves to the history; making the summary note
  resident would keep them but leaves it on screen. Left as is; troubleshooting says so.
- No XDG activation token is passed to the opener, so on Wayland the editor may open without
  focus.
