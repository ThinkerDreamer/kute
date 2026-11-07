//! `kute debug trace`: PipeWire graph updates recorded to a file, scrubbed of what identifies
//! the machine or its user, for bug reports and as detector test fixtures.
//!
//! The file is JSON lines: a header (`{"kute_trace": 1, ...}`), then one line per `pw-dump`
//! update, `{"t_ms": <ms since the trace started>, "objects": [...]}`, where `objects` is the
//! array `pw-dump` printed. Replaying the objects through a `Graph` gives the snapshots the
//! daemon saw.

use super::{Graph, Snapshot, Stream, StreamKind};
use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::mpsc::RecvTimeoutError;

/// The trace format's version, in the header line.
pub const FORMAT: u64 = 1;

/// One `pw-dump` update and when it came.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub t_ms: u64,
    pub objects: Vec<Value>,
}

/// The header line.
pub fn header() -> Value {
    json!({ "kute_trace": FORMAT, "kute_version": env!("CARGO_PKG_VERSION") })
}

/// Writes one entry as a line, flushed so a trace stopped with Ctrl-C is complete.
pub fn write_entry(out: &mut impl Write, entry: &Entry) -> Result<()> {
    let line = json!({ "t_ms": entry.t_ms, "objects": entry.objects });
    writeln!(out, "{line}")?;
    out.flush()?;
    Ok(())
}

/// Reads a trace: its entries, in order. Blank lines and the header are skipped.
pub fn read(r: impl BufRead) -> Result<Vec<Entry>> {
    let mut entries = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value =
            serde_json::from_str(&line).with_context(|| format!("trace line {}", i + 1))?;
        if let Some(format) = value.get("kute_trace") {
            if format.as_u64() != Some(FORMAT) {
                bail!("trace format {format} is not supported (this kute reads {FORMAT})");
            }
            continue;
        }
        let t_ms = value["t_ms"]
            .as_u64()
            .with_context(|| format!("trace line {} has no t_ms", i + 1))?;
        let Value::Array(objects) = value["objects"].clone() else {
            bail!("trace line {} has no objects array", i + 1);
        };
        entries.push(Entry { t_ms, objects });
    }
    Ok(entries)
}

/// Reads a trace file.
pub fn read_file(path: &Path) -> Result<Vec<Entry>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read(std::io::BufReader::new(file)).with_context(|| format!("reading {}", path.display()))
}

/// The snapshots a trace gives, each with its time: only the ones that changed, as the
/// daemon's monitor sends them.
pub fn snapshots(entries: &[Entry]) -> Vec<(u64, Snapshot)> {
    let mut graph = Graph::default();
    let mut out: Vec<(u64, Snapshot)> = Vec::new();
    for entry in entries {
        graph.apply(&entry.objects);
        let snap = graph.snapshot();
        if out.last().is_none_or(|(_, last)| *last != snap) {
            out.push((entry.t_ms, snap));
        }
    }
    out
}

/// Props whose whole value identifies a device, the machine or its user. Their values are
/// replaced, and wherever else they appear (a USB serial is part of the node names) too.
const SECRET_PROPS: &[(&str, &str)] = &[
    ("device.serial", "SERIAL"),
    ("api.bluez5.address", "MAC"),
    ("api.v4l2.cap.bus_info", "BUS"),
    ("device.bus-path", "BUS"),
    ("application.process.host", "HOST"),
    ("application.process.user", "USER"),
    ("application.process.machine-id", "MACHINE-ID"),
    ("application.process.session-id", "SESSION"),
    ("pipewire.sec.label", "LABEL"),
];

/// Values shorter than this are not replaced inside other strings: a one-letter user name
/// would mangle everything.
const MIN_SECRET_LEN: usize = 3;

/// Replaces device serials, MAC addresses, the host and user names, and the home folder in
/// `pw-dump` objects. It learns secrets as it goes, and gives each the same placeholder every
/// time, so a scrubbed trace still tells the same devices apart.
pub struct Scrubber {
    home: Option<String>,
    /// Secret text and its placeholder.
    secrets: BTreeMap<String, String>,
    /// How many placeholders of each kind were handed out.
    counts: BTreeMap<&'static str, usize>,
}

impl Scrubber {
    /// A scrubber that knows this machine's home folder, user and host names from the start.
    pub fn for_this_machine() -> Scrubber {
        let home = dirs::home_dir().map(|h| h.display().to_string());
        let user = std::env::var("USER")
            .or_else(|_| std::env::var("LOGNAME"))
            .ok();
        Scrubber::new(home, user, hostname())
    }

    pub fn new(home: Option<String>, user: Option<String>, host: Option<String>) -> Scrubber {
        let mut s = Scrubber {
            home: home
                .filter(|h| h.len() > 1)
                .map(|h| h.trim_end_matches('/').to_owned()),
            secrets: BTreeMap::new(),
            counts: BTreeMap::new(),
        };
        if let Some(user) = user {
            s.learn(&user, "user");
        }
        if let Some(host) = host {
            s.learn(&host, "host");
        }
        s
    }

    /// Scrubs one update in place: first learns the secrets of every object in it, then
    /// replaces them everywhere, so a node listed before its device is covered too.
    pub fn scrub(&mut self, objects: &mut [Value]) {
        for obj in objects.iter() {
            self.learn_from(obj);
        }
        for obj in objects.iter_mut() {
            self.replace_in(obj);
        }
    }

    fn learn(&mut self, secret: &str, kind: &'static str) {
        if secret.is_empty() || self.secrets.contains_key(secret) {
            return;
        }
        let placeholder = match kind {
            // One of each: numbering them would only add noise.
            "user" | "host" => kind.to_owned(),
            _ => {
                let n = self.counts.entry(kind).or_default();
                *n += 1;
                format!("{kind}-{n}")
            }
        };
        self.secrets.insert(secret.to_owned(), placeholder);
    }

    fn learn_from(&mut self, value: &Value) {
        match value {
            Value::Object(map) => {
                for (key, v) in map {
                    if let Some((_, kind)) = SECRET_PROPS.iter().find(|(k, _)| k == key) {
                        match v {
                            Value::String(s) => self.learn(s, kind),
                            Value::Number(n) => self.learn(&n.to_string(), kind),
                            _ => {}
                        }
                    }
                    self.learn_from(v);
                }
            }
            Value::Array(items) => items.iter().for_each(|v| self.learn_from(v)),
            _ => {}
        }
    }

    fn replace_in(&self, value: &mut Value) {
        match value {
            Value::String(s) => *s = self.replace(s),
            Value::Object(map) => self.replace_in_map(map),
            Value::Array(items) => items.iter_mut().for_each(|v| self.replace_in(v)),
            _ => {}
        }
    }

    fn replace_in_map(&self, map: &mut Map<String, Value>) {
        for (key, v) in map.iter_mut() {
            // A secret prop is replaced whole, however short: text search skips short ones.
            if SECRET_PROPS.iter().any(|(k, _)| k == key) {
                let text = match v {
                    Value::String(s) => Some(s.clone()),
                    Value::Number(n) => Some(n.to_string()),
                    _ => None,
                };
                if let Some(text) = text {
                    *v = Value::String(
                        self.secrets
                            .get(&text)
                            .cloned()
                            .unwrap_or_else(|| "scrubbed".into()),
                    );
                    continue;
                }
            }
            self.replace_in(v);
        }
    }

    /// `text` with the home folder, every known secret and any MAC address replaced.
    pub fn replace(&self, text: &str) -> String {
        let mut out = text.to_owned();
        if let Some(home) = &self.home {
            out = out.replace(home.as_str(), "~");
        }
        // Longest first, so a secret that contains another is replaced whole.
        let mut secrets: Vec<_> = self.secrets.iter().collect();
        secrets.retain(|(s, _)| s.len() >= MIN_SECRET_LEN);
        secrets.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        for (secret, placeholder) in secrets {
            out = out.replace(secret.as_str(), placeholder);
        }
        scrub_macs(&out)
    }
}

/// Six hex pairs joined by `:`, `_` or `-` (Bluetooth node names use `_`) become zeros, so
/// the structure of the name is kept.
fn scrub_macs(text: &str) -> String {
    let bytes = text.as_bytes();
    const LEN: usize = 17;
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + LEN <= bytes.len() && is_mac(&bytes[i..i + LEN]) {
            let sep = bytes[i + 2] as char;
            out.push_str(&["00"; 6].join(&sep.to_string()));
            i += LEN;
            continue;
        }
        // Not a MAC here: copy one character (which may be several bytes).
        let ch = text[i..].chars().next().expect("i is on a char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_mac(b: &[u8]) -> bool {
    let sep = b[2];
    if !matches!(sep, b':' | b'_' | b'-') {
        return false;
    }
    (0..6).all(|k| b[3 * k].is_ascii_hexdigit() && b[3 * k + 1].is_ascii_hexdigit())
        && (0..5).all(|k| b[3 * k + 2] == sep)
}

fn hostname() -> Option<String> {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is valid for its length; gethostname NUL-terminates when it fits.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    if rc != 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let name = String::from_utf8_lossy(&buf[..end]).into_owned();
    (!name.is_empty()).then_some(name)
}

/// One line for the terminal per stream that appeared, went away or changed state, so the
/// person recording a trace can see it pick up their call.
pub fn describe_changes(before: &Snapshot, after: &Snapshot) -> Vec<String> {
    let label = |s: &Stream| {
        let kind = match s.kind {
            StreamKind::Mic => "mic",
            StreamKind::Output => "output",
        };
        let app = s
            .app
            .name
            .as_deref()
            .or(s.app.binary.as_deref())
            .unwrap_or("?");
        let monitor = if s.monitor { ", peak meter" } else { "" };
        format!("{kind} stream #{} {app}{monitor}", s.serial)
    };
    let state = |s: &Stream| if s.running { "running" } else { "idle" };
    let mut lines = Vec::new();
    for s in &after.streams {
        match before.streams.iter().find(|b| b.serial == s.serial) {
            None => lines.push(format!("+ {} ({})", label(s), state(s))),
            Some(b) if b.running != s.running => {
                lines.push(format!("~ {} is {}", label(s), state(s)))
            }
            Some(_) => {}
        }
    }
    for b in &before.streams {
        if !after.streams.iter().any(|s| s.serial == b.serial) {
            lines.push(format!("- {}", label(b)));
        }
    }
    if before.default_source != after.default_source
        && let Some(source) = &after.default_source
    {
        lines.push(format!("default source: {source}"));
    }
    if before.default_sink != after.default_sink
        && let Some(sink) = &after.default_sink
    {
        lines.push(format!("default sink: {sink}"));
    }
    lines
}

/// Records `pw-dump --monitor` to `out` until `pw-dump` exits, `duration` passes, or the
/// process is interrupted. What changed is described on stderr as it happens.
pub fn record(out: &mut impl Write, duration: Option<std::time::Duration>) -> Result<()> {
    let mut child = super::spawn_pw_dump().context("starting pw-dump (is PipeWire installed?)")?;
    let stdout = child.stdout.take().expect("stdout is piped");
    // Stops pw-dump after `duration`, unless cancelled first. It is joined before the child
    // is reaped, so the PID it kills is still pw-dump's.
    let (cancel, cancelled) = std::sync::mpsc::channel::<()>();
    let timer = duration.map(|duration| {
        let id = child.id();
        std::thread::spawn(move || {
            let fire = cancelled.recv_timeout(duration) == Err(RecvTimeoutError::Timeout);
            if fire {
                // SAFETY: plain kill(2) on our own unreaped child.
                unsafe { libc::kill(id as libc::pid_t, libc::SIGTERM) };
            }
            fire
        })
    });
    let header = writeln!(out, "{}", header()).and_then(|()| out.flush());
    if let Err(e) = header {
        drop(cancel);
        if let Some(timer) = timer {
            let _ = timer.join();
        }
        let _ = child.kill();
        let _ = child.wait();
        return Err(anyhow::Error::from(e).context("writing the trace"));
    }
    let started = std::time::Instant::now();
    let mut scrubber = Scrubber::for_this_machine();
    let mut graph = Graph::default();
    let mut last = Snapshot::default();
    let mut written = 0usize;
    let mut failed = None;
    let read = super::read_arrays(stdout, |mut objects| {
        if failed.is_some() {
            return;
        }
        let t_ms = started.elapsed().as_millis() as u64;
        graph.apply(&objects);
        let snap = graph.snapshot();
        for line in describe_changes(&last, &snap) {
            eprintln!("{:>8.1} s  {line}", t_ms as f64 / 1000.0);
        }
        last = snap;
        scrubber.scrub(&mut objects);
        match write_entry(out, &Entry { t_ms, objects }) {
            Ok(()) => written += 1,
            Err(e) => {
                failed = Some(e);
                // Closes the pipe, so the read ends instead of waiting for more updates.
                let _ = child.kill();
            }
        }
    });
    drop(cancel);
    let timed_out = timer.is_some_and(|t| t.join().unwrap_or(false));
    let _ = child.kill();
    let _ = child.wait();
    if let Some(e) = failed {
        return Err(e.context("writing the trace"));
    }
    // pw-dump stopped at the end of `duration` may end its output mid-array; anything
    // else that cut the read short is an error.
    if let Err(e) = read
        && !timed_out
    {
        return Err(e);
    }
    eprintln!("kute: {written} updates recorded");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scrubber() -> Scrubber {
        Scrubber::new(
            Some("/home/alex".into()),
            Some("alex".into()),
            Some("alex-laptop".into()),
        )
    }

    #[test]
    fn serials_are_replaced_everywhere_they_appear() {
        let mut objects = vec![
            json!({"id": 50, "type": "PipeWire:Interface:Node", "info": {"props": {
                "node.name": "alsa_input.usb-Acme_Headset_A1B2C3D4-00.mono-fallback",
                "object.serial": 500
            }}}),
            json!({"id": 40, "type": "PipeWire:Interface:Device", "info": {"props": {
                "device.serial": "Acme_Headset_A1B2C3D4",
                "device.name": "alsa_card.usb-Acme_Headset_A1B2C3D4-00"
            }}}),
        ];
        let mut s = scrubber();
        s.scrub(&mut objects);
        let text = serde_json::to_string(&objects).unwrap();
        assert!(!text.contains("A1B2C3D4"), "{text}");
        assert_eq!(
            objects[0]["info"]["props"]["node.name"],
            "alsa_input.usb-SERIAL-1-00.mono-fallback"
        );
        // Numbers that aren't secrets stay numbers.
        assert_eq!(objects[0]["info"]["props"]["object.serial"], 500);
    }

    #[test]
    fn the_same_secret_gets_the_same_placeholder_later() {
        let mut s = scrubber();
        let mut first = vec![json!({"info": {"props": {"device.serial": "XYZ123"}}})];
        let mut second = vec![json!({"info": {"props": {"node.name": "dev-XYZ123"}}})];
        s.scrub(&mut first);
        s.scrub(&mut second);
        assert_eq!(second[0]["info"]["props"]["node.name"], "dev-SERIAL-1");
    }

    #[test]
    fn user_host_home_and_macs_are_scrubbed() {
        let mut objects = vec![json!({"info": {"props": {
            "application.process.user": "alex",
            "application.process.host": "alex-laptop",
            "application.process.machine-id": "0123456789abcdef",
            "media.filename": "/home/alex/Music/song.flac",
            "node.name": "bluez_output.AA_BB_CC_11_22_33.1",
            "api.bluez5.address": "AA:BB:CC:11:22:33",
            "application.name": "Firefox"
        }}})];
        scrubber().scrub(&mut objects);
        let props = &objects[0]["info"]["props"];
        assert_eq!(props["application.process.user"], "user");
        assert_eq!(props["application.process.host"], "host");
        assert_eq!(props["application.process.machine-id"], "MACHINE-ID-1");
        assert_eq!(props["media.filename"], "~/Music/song.flac");
        assert_eq!(props["node.name"], "bluez_output.00_00_00_00_00_00.1");
        assert_eq!(props["api.bluez5.address"], "MAC-1");
        assert_eq!(props["application.name"], "Firefox");
    }

    #[test]
    fn short_device_secrets_are_still_replaced_where_they_are_named() {
        let mut objects = vec![json!({"info": {"props": {
            "device.serial": "7",
            "device.bus-path": 12,
            "node.name": "alsa_input.7"
        }}})];
        scrubber().scrub(&mut objects);
        let props = &objects[0]["info"]["props"];
        assert_eq!(props["device.serial"], "SERIAL-1");
        assert_eq!(props["device.bus-path"], "BUS-1");
        // Too short to search for inside other text.
        assert_eq!(props["node.name"], "alsa_input.7");
    }

    #[test]
    fn short_user_names_are_not_replaced_inside_words() {
        let s = Scrubber::new(None, Some("al".into()), None);
        assert_eq!(s.replace("alsa_input"), "alsa_input");
    }

    #[test]
    fn mac_scrubbing_leaves_other_text_alone() {
        assert_eq!(scrub_macs("node 12:34 é 0a-1b"), "node 12:34 é 0a-1b");
        assert_eq!(scrub_macs("x 0a-1b-2c-3d-4e-5f y"), "x 00-00-00-00-00-00 y");
    }

    #[test]
    fn a_written_trace_reads_back() {
        let entries = vec![
            Entry {
                t_ms: 0,
                objects: vec![json!({"id": 1})],
            },
            Entry {
                t_ms: 250,
                objects: vec![json!({"id": 1, "type": null, "info": null})],
            },
        ];
        let mut buf = Vec::new();
        writeln!(buf, "{}", header()).unwrap();
        for e in &entries {
            write_entry(&mut buf, e).unwrap();
        }
        assert_eq!(read(buf.as_slice()).unwrap(), entries);
    }

    #[test]
    fn an_unknown_format_is_refused() {
        let text = "{\"kute_trace\": 99}\n";
        assert!(read(text.as_bytes()).is_err());
    }

    #[test]
    fn changes_are_described_for_the_terminal() {
        let mic = |running| Stream {
            serial: 7,
            kind: StreamKind::Mic,
            app: crate::pw::App {
                name: Some("Firefox".into()),
                binary: Some("firefox".into()),
            },
            node_name: None,
            pid: None,
            monitor: false,
            running,
        };
        let empty = Snapshot::default();
        let one = Snapshot {
            streams: vec![mic(true)],
            ..Snapshot::default()
        };
        let idle = Snapshot {
            streams: vec![mic(false)],
            ..Snapshot::default()
        };
        assert_eq!(
            describe_changes(&empty, &one),
            ["+ mic stream #7 Firefox (running)"]
        );
        assert_eq!(
            describe_changes(&one, &idle),
            ["~ mic stream #7 Firefox is idle"]
        );
        assert_eq!(describe_changes(&idle, &empty), ["- mic stream #7 Firefox"]);
    }
}
