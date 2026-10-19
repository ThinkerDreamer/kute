//! The audio side of the PipeWire graph, built from `pw-dump`'s JSON objects.

use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct App {
    pub name: Option<String>,
    pub binary: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    Mic,
    Output,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    pub serial: u64,
    pub kind: StreamKind,
    pub app: App,
    pub node_name: Option<String>,
    pub pid: Option<u32>,
    /// `stream.monitor = true`: a peak meter, not a real consumer of the audio.
    pub monitor: bool,
    /// `info.state == "running"`.
    pub running: bool,
}

/// An `Audio/Source*` node: a device `capture.mic` can pin.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    /// `node.name`
    pub name: String,
    /// `node.description`
    pub description: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    /// Sorted by serial.
    pub streams: Vec<Stream>,
    pub default_source: Option<String>,
    pub default_sink: Option<String>,
    /// So the daemon can resolve `capture.mic` from a snapshot.
    pub sources: Vec<Source>,
}

impl Snapshot {
    /// The `node.name` of the first source whose `node.name` or `node.description`
    /// contains `substring`, ignoring case.
    pub fn find_source(&self, substring: &str) -> Option<String> {
        let needle = substring.to_lowercase();
        self.sources
            .iter()
            .find(|s| {
                std::iter::once(&s.name)
                    .chain(&s.description)
                    .any(|v| v.to_lowercase().contains(&needle))
            })
            .map(|s| s.name.clone())
    }
}

#[derive(Default)]
struct Node {
    props: Map<String, Value>,
    state: Option<String>,
}

impl Node {
    fn str(&self, key: &str) -> Option<String> {
        match self.props.get(key)? {
            Value::String(s) if !s.is_empty() => Some(s.clone()),
            _ => None,
        }
    }

    /// Numbers come as JSON numbers, but accept a numeric string too.
    fn u64(&self, key: &str) -> Option<u64> {
        match self.props.get(key)? {
            Value::Number(n) => n.as_u64(),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    fn class(&self) -> Option<&str> {
        self.props.get("media.class")?.as_str()
    }

    /// `Audio/Source` or a variant such as `Audio/Source/Virtual` (noise suppression, echo
    /// cancel). Sink monitors are `Audio/Sink` nodes, so they don't count.
    fn is_source(&self) -> bool {
        self.class().is_some_and(|c| c.starts_with("Audio/Source"))
    }
}

#[derive(Default)]
struct Link {
    output: Option<u64>,
    input: Option<u64>,
}

/// The `default` metadata object: which source and sink are the defaults.
#[derive(Default)]
struct Defaults {
    is_default: bool,
    source: Option<String>,
    sink: Option<String>,
}

/// Everything `pw-dump` has told us so far, keyed by object id.
#[derive(Default)]
pub struct Graph {
    nodes: BTreeMap<u64, Node>,
    links: BTreeMap<u64, Link>,
    metadata: BTreeMap<u64, Defaults>,
}

impl Graph {
    /// Applies one `pw-dump` update: an array of new, changed or removed objects.
    pub fn apply(&mut self, objects: &[Value]) {
        for obj in objects {
            let Some(id) = obj.get("id").and_then(Value::as_u64) else {
                continue;
            };
            // Only a key that is present and null means removal: metadata objects have no
            // `info` at all.
            let null = |key| obj.get(key).is_some_and(Value::is_null);
            if null("info") || null("type") {
                // A link to a removed node is stale even before its own removal arrives,
                // and the node's id may be reused.
                if self.nodes.remove(&id).is_some() {
                    self.links
                        .retain(|_, l| l.input != Some(id) && l.output != Some(id));
                }
                self.links.remove(&id);
                self.metadata.remove(&id);
                continue;
            }
            match obj.get("type").and_then(Value::as_str) {
                Some("PipeWire:Interface:Node") => self.apply_node(id, obj),
                Some("PipeWire:Interface:Link") => self.apply_link(id, obj),
                Some("PipeWire:Interface:Metadata") => self.apply_metadata(id, obj),
                _ => {}
            }
        }
    }

    fn apply_node(&mut self, id: u64, obj: &Value) {
        let node = self.nodes.entry(id).or_default();
        let info = &obj["info"];
        // An update without props only changes the state.
        if let Some(props) = info.get("props").and_then(Value::as_object) {
            node.props = props.clone();
        }
        if let Some(state) = info.get("state").and_then(Value::as_str) {
            node.state = Some(state.to_owned());
        }
    }

    fn apply_link(&mut self, id: u64, obj: &Value) {
        let link = self.links.entry(id).or_default();
        let info = &obj["info"];
        if let Some(out) = info.get("output-node-id").and_then(Value::as_u64) {
            link.output = Some(out);
        }
        if let Some(inn) = info.get("input-node-id").and_then(Value::as_u64) {
            link.input = Some(inn);
        }
    }

    fn apply_metadata(&mut self, id: u64, obj: &Value) {
        let meta = self.metadata.entry(id).or_default();
        if let Some(name) = obj.pointer("/props/metadata.name").and_then(Value::as_str) {
            meta.is_default = name == "default";
        }
        // `pw-dump` prints every entry of the object each time, so this replaces them all.
        let Some(entries) = obj.get("metadata").and_then(Value::as_array) else {
            return;
        };
        let entry = |key: &str| {
            let value = entries.iter().find(|e| e["key"] == key)?.get("value")?;
            default_name(value)
        };
        meta.source = entry("default.audio.source");
        meta.sink = entry("default.audio.sink");
    }

    /// The audio streams that matter for detection, and the default devices.
    pub fn snapshot(&self) -> Snapshot {
        let mut streams: Vec<Stream> = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| {
                let kind = match node.class()? {
                    "Stream/Output/Audio" => StreamKind::Output,
                    "Stream/Input/Audio" if self.fed_by_source(id) => StreamKind::Mic,
                    _ => return None,
                };
                Some(Stream {
                    serial: node.u64("object.serial")?,
                    kind,
                    app: App {
                        name: node.str("application.name"),
                        binary: node.str("application.process.binary"),
                    },
                    node_name: node.str("node.name"),
                    pid: node
                        .u64("application.process.id")
                        .and_then(|p| p.try_into().ok()),
                    monitor: match node.props.get("stream.monitor") {
                        Some(Value::Bool(b)) => *b,
                        Some(Value::String(s)) => s == "true",
                        _ => false,
                    },
                    running: node.state.as_deref() == Some("running"),
                })
            })
            .collect();
        streams.sort_by_key(|s| s.serial);

        let defaults = self.metadata.values().find(|m| m.is_default);
        let sources = self
            .nodes
            .values()
            .filter(|n| n.is_source())
            .filter_map(|n| {
                Some(Source {
                    name: n.str("node.name")?,
                    description: n.str("node.description"),
                })
            })
            .collect();
        Snapshot {
            streams,
            default_source: defaults.and_then(|d| d.source.clone()),
            default_sink: defaults.and_then(|d| d.sink.clone()),
            sources,
        }
    }

    /// Whether some link goes from an `Audio/Source*` node into node `id`.
    fn fed_by_source(&self, id: u64) -> bool {
        self.links.values().any(|link| {
            link.input == Some(id)
                && link
                    .output
                    .and_then(|out| self.nodes.get(&out))
                    .is_some_and(Node::is_source)
        })
    }

    /// See `Snapshot::find_source`.
    pub fn find_source(&self, substring: &str) -> Option<String> {
        self.snapshot().find_source(substring)
    }
}

/// A default device entry's value: `{"name": "..."}`, or the same JSON encoded in a string.
fn default_name(value: &Value) -> Option<String> {
    let parsed;
    let value = match value {
        Value::String(s) => {
            parsed = serde_json::from_str::<Value>(s).ok()?;
            &parsed
        }
        v => v,
    };
    Some(value.get("name")?.as_str()?.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const SOURCE: &str = "Audio/Source";
    const MIC_IN: &str = "Stream/Input/Audio";
    const OUT: &str = "Stream/Output/Audio";

    fn node(id: u32, serial: u64, class: &str, props: &[(&str, &str)], state: &str) -> Value {
        let mut p = json!({ "object.serial": serial, "media.class": class });
        for (k, v) in props {
            p[*k] = json!(v);
        }
        json!({ "id": id, "type": "PipeWire:Interface:Node", "info": { "state": state, "props": p } })
    }

    fn link(id: u32, out: u32, inn: u32) -> Value {
        json!({
            "id": id,
            "type": "PipeWire:Interface:Link",
            "info": { "output-node-id": out, "input-node-id": inn, "state": "active" }
        })
    }

    fn removed(id: u32) -> Value {
        json!({ "id": id, "type": null, "info": null })
    }

    fn defaults(id: u32, source: &str, sink: &str) -> Value {
        json!({
            "id": id,
            "type": "PipeWire:Interface:Metadata",
            "props": { "metadata.name": "default" },
            "metadata": [
                { "subject": 0, "key": "default.audio.source", "type": "Spa:String:JSON", "value": { "name": source } },
                { "subject": 0, "key": "default.audio.sink", "type": "Spa:String:JSON", "value": { "name": sink } }
            ]
        })
    }

    fn graph(objects: &[Value]) -> Graph {
        let mut g = Graph::default();
        g.apply(objects);
        g
    }

    #[test]
    fn mic_stream_needs_a_link_from_a_real_source() {
        let g = graph(&[
            node(
                30,
                300,
                SOURCE,
                &[("node.name", "alsa_input.usb")],
                "running",
            ),
            node(
                40,
                400,
                MIC_IN,
                &[("application.name", "Jitsi Meet")],
                "running",
            ),
            node(
                50,
                500,
                OUT,
                &[("application.name", "Jitsi Meet")],
                "running",
            ),
            node(
                60,
                600,
                MIC_IN,
                &[("media.name", "PlasmaPA-VolumeMeter")],
                "running",
            ),
            link(70, 30, 40),
            link(71, 50, 60),
        ]);
        let snap = g.snapshot();
        let kinds: Vec<(u64, StreamKind)> =
            snap.streams.iter().map(|s| (s.serial, s.kind)).collect();
        assert_eq!(
            kinds,
            vec![(400, StreamKind::Mic), (500, StreamKind::Output)]
        );
        assert_eq!(snap.streams[0].app.name.as_deref(), Some("Jitsi Meet"));
        assert!(snap.streams[0].running);
    }

    #[test]
    fn removal_drops_node_and_its_links() {
        let mut g = graph(&[
            node(30, 300, SOURCE, &[], "running"),
            node(40, 400, MIC_IN, &[], "running"),
            node(41, 410, MIC_IN, &[], "running"),
            node(50, 500, OUT, &[], "running"),
            link(70, 30, 40),
            link(71, 30, 41),
        ]);
        assert_eq!(g.snapshot().streams.len(), 3);

        g.apply(&[removed(70)]);
        let serials: Vec<u64> = g.snapshot().streams.iter().map(|s| s.serial).collect();
        assert_eq!(serials, vec![410, 500]);

        g.apply(&[removed(50)]);
        let serials: Vec<u64> = g.snapshot().streams.iter().map(|s| s.serial).collect();
        assert_eq!(serials, vec![410]);

        // Removing the source takes its links along, so a new source that reuses its id
        // doesn't inherit them.
        g.apply(&[removed(30)]);
        assert!(g.snapshot().streams.is_empty());
        g.apply(&[node(30, 900, SOURCE, &[], "running")]);
        assert!(g.snapshot().streams.is_empty());
    }

    #[test]
    fn any_audio_source_class_feeds_a_mic() {
        let g = graph(&[
            node(30, 300, "Audio/Source/Virtual", &[], "running"),
            node(31, 310, "Audio/Sink", &[], "running"),
            node(40, 400, MIC_IN, &[], "running"),
            node(41, 410, MIC_IN, &[], "running"),
            link(70, 30, 40),
            link(71, 31, 41),
        ]);
        let kinds: Vec<(u64, StreamKind)> = g
            .snapshot()
            .streams
            .iter()
            .map(|s| (s.serial, s.kind))
            .collect();
        assert_eq!(kinds, vec![(400, StreamKind::Mic)]);
    }

    #[test]
    fn state_only_update_keeps_props() {
        let mut g = graph(&[node(
            50,
            500,
            OUT,
            &[("application.name", "Jitsi Meet")],
            "running",
        )]);
        g.apply(&[
            json!({ "id": 50, "type": "PipeWire:Interface:Node", "info": { "state": "idle" } }),
        ]);
        let snap = g.snapshot();
        assert_eq!(snap.streams.len(), 1);
        assert_eq!(snap.streams[0].app.name.as_deref(), Some("Jitsi Meet"));
        assert!(!snap.streams[0].running);
    }

    #[test]
    fn peak_meter_flag_and_app_fields() {
        let mut out = node(
            50,
            3527,
            OUT,
            &[
                ("application.name", "Jitsi Meet"),
                ("application.process.binary", "electron"),
                ("node.name", "Jitsi Meet"),
                ("stream.monitor", "true"),
            ],
            "running",
        );
        out["info"]["props"]["application.process.id"] = json!(4242);
        let mut meter = node(
            51,
            3600,
            OUT,
            &[("application.process.binary", "plasmashell")],
            "idle",
        );
        meter["info"]["props"]["stream.monitor"] = json!(true);
        let g = graph(&[out, meter, node(52, 3700, OUT, &[], "running")]);

        let snap = g.snapshot();
        let s = &snap.streams[0];
        assert_eq!(s.serial, 3527);
        assert_eq!(
            s.app,
            App {
                name: Some("Jitsi Meet".into()),
                binary: Some("electron".into())
            }
        );
        assert_eq!(s.node_name.as_deref(), Some("Jitsi Meet"));
        assert_eq!(s.pid, Some(4242));
        assert!(s.monitor);
        assert!(snap.streams[1].monitor);
        assert_eq!(
            snap.streams[1].app,
            App {
                name: None,
                binary: Some("plasmashell".into())
            }
        );
        assert!(!snap.streams[2].monitor);
    }

    #[test]
    fn defaults_come_from_metadata() {
        let mut g = graph(&[defaults(5, "alsa_input.usb", "alsa_output.hdmi")]);
        let snap = g.snapshot();
        assert_eq!(snap.default_source.as_deref(), Some("alsa_input.usb"));
        assert_eq!(snap.default_sink.as_deref(), Some("alsa_output.hdmi"));

        let mut encoded = defaults(5, "", "");
        encoded["metadata"][0]["value"] = json!("{\"name\":\"x\"}");
        g.apply(&[encoded]);
        assert_eq!(g.snapshot().default_source.as_deref(), Some("x"));

        g.apply(&[removed(5)]);
        assert_eq!(g.snapshot(), Snapshot::default());
    }

    #[test]
    fn other_metadata_is_ignored() {
        let mut settings = defaults(6, "a", "b");
        settings["props"]["metadata.name"] = json!("settings");
        assert_eq!(graph(&[settings]).snapshot(), Snapshot::default());
    }

    #[test]
    fn find_source_matches_a_substring() {
        let g = graph(&[
            node(
                30,
                300,
                SOURCE,
                &[
                    ("node.name", "alsa_input.pci-analog"),
                    ("node.description", "Built-in Audio"),
                ],
                "suspended",
            ),
            node(
                31,
                301,
                SOURCE,
                &[
                    ("node.name", "alsa_input.usb-headset"),
                    ("node.description", "Headset Mono"),
                ],
                "suspended",
            ),
            node(
                32,
                302,
                "Audio/Sink",
                &[("node.name", "alsa_output.usb-headset")],
                "idle",
            ),
        ]);
        assert_eq!(
            g.find_source("usb").as_deref(),
            Some("alsa_input.usb-headset")
        );
        assert_eq!(
            g.find_source("BUILT-IN").as_deref(),
            Some("alsa_input.pci-analog")
        );
        assert_eq!(g.find_source("bluetooth"), None);
        // The daemon resolves `capture.mic` from snapshots, which carry the sources.
        let names: Vec<_> = g.snapshot().sources.into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["alsa_input.pci-analog", "alsa_input.usb-headset"]);

        let g = graph(&[
            node(
                32,
                302,
                "Audio/Sink",
                &[("node.name", "noise-cancel-sink")],
                "idle",
            ),
            node(
                33,
                303,
                "Audio/Source/Virtual",
                &[("node.name", "noise-cancel-source")],
                "idle",
            ),
        ]);
        assert_eq!(
            g.find_source("noise").as_deref(),
            Some("noise-cancel-source")
        );
    }
}
