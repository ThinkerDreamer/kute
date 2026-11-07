//! User configuration (`~/.config/kute/config.toml`). Every key has a default.

use anyhow::{Context, Result, bail};
use jiff::SignedDuration;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub user: UserConfig,
    pub labels: Labels,
    pub storage: StorageConfig,
    pub transcribe: TranscribeConfig,
    pub summary: SummaryConfig,
    pub detect: DetectConfig,
    pub capture: CaptureConfig,
    pub notify: NotifyConfig,
    pub tray: TrayConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct UserConfig {
    /// Used in summary prompts as the person behind the "me" track.
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Labels {
    pub me: String,
    pub others: String,
}

impl Default for Labels {
    fn default() -> Self {
        Self {
            me: "Me".into(),
            others: "Others".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TranscribeConfig {
    /// A model name from `kute model list`, or a path to a ggml `.bin` file.
    pub model: String,
    /// `auto`, a Whisper language code such as `it` to force it, or several such as `ru,en`
    /// to detect among them. A session's own languages (`kute language`) win.
    pub language: String,
    pub gpu: bool,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            model: "large-v3-turbo".into(),
            language: "auto".into(),
            gpu: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SummaryBackend {
    #[default]
    None,
    Openai,
    Anthropic,
    Command,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SummaryConfig {
    pub backend: SummaryBackend,
    /// `auto` (dominant language of the transcript) or a language code.
    pub language: String,
    pub min_words: usize,
    pub prompt: Option<PathBuf>,
    pub command: Vec<String>,
    pub timeout: SignedDuration,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub api_key_env: Option<String>,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            backend: SummaryBackend::None,
            language: "auto".into(),
            min_words: 80,
            prompt: None,
            command: Vec::new(),
            timeout: SignedDuration::from_secs(600),
            base_url: None,
            model: None,
            api_key_env: None,
        }
    }
}

/// When the daemon starts and stops recording, and how it splits conversations.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DetectConfig {
    /// A call must be live this long before a session starts.
    pub min_duration: SignedDuration,
    /// How long the call may be gone before the session stops.
    pub stop_grace: SignedDuration,
    /// Silence on both tracks that ends a conversation.
    pub conversation_gap: SignedDuration,
    pub ignore_apps: Vec<String>,
    /// When set, only these apps trigger a recording.
    pub allow_apps: Option<Vec<String>>,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            min_duration: SignedDuration::from_secs(20),
            stop_grace: SignedDuration::from_secs(15),
            conversation_gap: SignedDuration::from_mins(5),
            ignore_apps: Vec::new(),
            allow_apps: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct CaptureConfig {
    /// A substring of the microphone's name; the default source when unset.
    pub mic: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct NotifyConfig {
    /// The command that opens a summary when its notification is clicked; the file's path is
    /// added as the last argument.
    pub open: Vec<String>,
}

impl Default for NotifyConfig {
    fn default() -> Self {
        Self {
            open: vec!["xdg-open".into()],
        }
    }
}

/// The tray icon the daemon shows.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TrayConfig {
    pub enabled: bool,
}

impl Default for TrayConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl Config {
    pub fn default_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("kute").join("config.toml"))
    }

    /// Loads `path`, or the default path. A missing default file means "all defaults";
    /// a missing explicit file is an error.
    pub fn load(path: Option<&Path>) -> Result<Config> {
        let path = match path {
            Some(p) => {
                if !p.is_file() {
                    bail!("config file {} does not exist", p.display());
                }
                p.to_path_buf()
            }
            None => match Self::default_path() {
                Some(p) if p.is_file() => p,
                _ => return Ok(Config::default()),
            },
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Config> {
        let config: Config = toml::from_str(text)?;
        crate::transcribe::parse_languages(&config.transcribe.language)
            .context("transcribe.language")?;
        Ok(config)
    }

    pub fn data_dir(&self) -> Result<PathBuf> {
        match &self.storage.dir {
            Some(d) => Ok(expand_tilde(d)),
            None => dirs::data_dir()
                .map(|d| d.join("kute"))
                .context("no XDG data directory found; set storage.dir in the config"),
        }
    }
}

/// Expands a leading `~` to the home directory.
pub fn expand_tilde(p: &Path) -> PathBuf {
    if let Ok(rest) = p.strip_prefix("~")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    p.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_gives_defaults() {
        let c = Config::parse("").unwrap();
        assert_eq!(c.labels.me, "Me");
        assert_eq!(c.labels.others, "Others");
        assert_eq!(c.transcribe.model, "large-v3-turbo");
        assert_eq!(c.transcribe.language, "auto");
        assert!(c.transcribe.gpu);
        assert_eq!(c.summary.backend, SummaryBackend::None);
        assert_eq!(c.summary.language, "auto");
        assert_eq!(c.summary.min_words, 80);
        assert_eq!(c.summary.timeout, jiff::SignedDuration::from_secs(600));
        assert!(c.storage.dir.is_none());
    }

    #[test]
    fn full_example_parses() {
        let c = Config::parse(
            r#"
            [user]
            name = "Alex"
            [labels]
            me = "Alex"
            others = "Team"
            [storage]
            dir = "~/Calls"
            [transcribe]
            model = "small"
            language = "it"
            gpu = false
            [summary]
            backend = "command"
            command = ["claude", "-p"]
            language = "en"
            min_words = 10
            timeout = "2m"
            "#,
        )
        .unwrap();
        assert_eq!(c.user.name, "Alex");
        assert_eq!(c.labels.others, "Team");
        assert_eq!(c.transcribe.model, "small");
        assert!(!c.transcribe.gpu);
        assert_eq!(c.summary.backend, SummaryBackend::Command);
        assert_eq!(c.summary.command, vec!["claude", "-p"]);
        assert_eq!(c.summary.timeout, jiff::SignedDuration::from_secs(120));
        let home = dirs::home_dir().unwrap();
        assert_eq!(c.data_dir().unwrap(), home.join("Calls"));
    }

    #[test]
    fn transcribe_language_takes_several_codes() {
        let c = Config::parse("[transcribe]\nlanguage = \"ru,en\"").unwrap();
        assert_eq!(c.transcribe.language, "ru,en");
        let err = Config::parse("[transcribe]\nlanguage = \"xx\"").unwrap_err();
        assert!(
            format!("{err:#}").contains("transcribe.language"),
            "{err:#}"
        );
    }

    #[test]
    fn unknown_backend_is_an_error() {
        let err = Config::parse("[summary]\nbackend = \"gpt\"").unwrap_err();
        assert!(format!("{err:#}").contains("gpt"), "{err:#}");
    }

    #[test]
    fn missing_default_file_is_fine_but_missing_explicit_file_is_not() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope.toml");
        assert!(Config::load(Some(&missing)).is_err());
    }

    #[test]
    fn expand_tilde_only_touches_leading_tilde() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_tilde(Path::new("~/a/b")), home.join("a/b"));
        assert_eq!(expand_tilde(Path::new("/x/~/y")), PathBuf::from("/x/~/y"));
    }

    #[test]
    fn the_tray_is_on_unless_turned_off() {
        assert!(Config::parse("").unwrap().tray.enabled);
        assert!(
            !Config::parse("[tray]\nenabled = false\n")
                .unwrap()
                .tray
                .enabled
        );
    }

    #[test]
    fn notify_open_defaults_to_xdg_open() {
        assert_eq!(Config::parse("").unwrap().notify.open, ["xdg-open"]);
        let c = Config::parse("[notify]\nopen = [\"kate\"]\n").unwrap();
        assert_eq!(c.notify.open, ["kate"]);
    }

    #[test]
    fn detect_and_capture_defaults() {
        let c = Config::parse("").unwrap();
        assert_eq!(c.detect.min_duration, SignedDuration::from_secs(20));
        assert_eq!(c.detect.stop_grace, SignedDuration::from_secs(15));
        assert_eq!(c.detect.conversation_gap, SignedDuration::from_mins(5));
        assert!(c.detect.ignore_apps.is_empty() && c.detect.allow_apps.is_none());
        assert!(c.capture.mic.is_none());
    }

    #[test]
    fn detect_values_parse() {
        let c = Config::parse(
            "[detect]\nmin_duration = \"30s\"\nconversation_gap = \"2m\"\nallow_apps = [\"Jitsi Meet\"]\n[capture]\nmic = \"USB\"\n",
        )
        .unwrap();
        assert_eq!(c.detect.min_duration, SignedDuration::from_secs(30));
        assert_eq!(c.detect.conversation_gap, SignedDuration::from_mins(2));
        assert_eq!(
            c.detect.allow_apps.as_deref(),
            Some(&["Jitsi Meet".to_string()][..])
        );
        assert_eq!(c.capture.mic.as_deref(), Some("USB"));
    }
}
