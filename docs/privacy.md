# Privacy

kute records conversations, so it's worth knowing exactly what it keeps and what, if anything,
leaves your machine.

> [!IMPORTANT]
> **kute records everything your computer plays while a call is recorded, not only the call.**
> The "others" track is the system's whole audio output: a video, music, or a notification sound
> during a call ends up in the recording and the transcript too.

## What leaves your machine

With the default config, nothing but the model downloads:

- **`kute model download`** fetches the Whisper and Silero VAD models from Hugging Face
  (`huggingface.co`). That's a plain file download; no audio or text is sent.
- **Transcription** runs locally with whisper.cpp. Audio is never uploaded.
- **Summaries** are off by default. When you turn them on, the transcript of each conversation
  (text, not audio) goes to the backend you chose:
  - `command`: whatever that program does with it. A local model (e.g. `ollama run`) keeps it on
    your machine; a CLI for a hosted model sends it to that provider.
  - `openai`: the server at `base_url`. The default is a local Ollama.
  - `anthropic`: Anthropic's API.

  The prompt also includes the date, the call app, the session title, the session's notes
  (`kute notes`), the others' name (`kute others`) and your configured name.

kute has no telemetry, update checks or crash reporting.

## What kute stores

Everything lives under `~/.local/share/kute` (`$XDG_DATA_HOME/kute` if you set that, or
`storage.dir`), which kute makes readable only by you:

```
sessions/<start time>/
  meta.toml          what kute knows about the session: times, app, conversations, status
  me.opus            your microphone
  others.opus        everyone else (the system audio)
  me.json            raw transcription results, per track
  others.json
  transcript.md      the transcript
  summary.md         the summaries, if any
  summaries/NN.md    one summary per conversation, combined into summary.md
index.sqlite         the search index, rebuilt from the session folders
models/              downloaded models
```

While a call is being recorded and processed, the audio is kept as WAV and compressed to the
`opus` format afterwards.

## Deleting recordings

- During a call, `kute discard` stops recording and deletes the session.
- Afterwards, delete the session's folder under `sessions/` and run `kute reindex` so search
  forgets it.

## Keeping kute out of a call

- `kute pause` stops it from starting new recordings until `kute resume`.
- `ignore_apps` in the [config](configuration.md#detect) excludes an app for good, and
  `allow_apps` limits recording to the apps you list.
- Mute at the system level (e.g. your desktop's microphone icon), not only in the call app.
  Many call apps keep the microphone open while muted, so kute still hears you; a system mute
  silences kute too.

## Other people on the call

> [!IMPORTANT]
> **kute does not announce itself to the other participants.** Laws about recording
> conversations differ between countries and states, and some require everyone's consent. Making
> sure you're allowed to record is your responsibility, and telling people you're transcribing the
> call is good manners everywhere.
