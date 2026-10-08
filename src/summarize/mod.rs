//! Summarization backends and the prompt they get.

pub mod command;

use crate::config::{SummaryBackend, SummaryConfig, expand_tilde};
use anyhow::{Context, Result, bail};

pub trait Summarizer {
    /// Recorded in `meta.toml`, e.g. `command: claude -p`.
    fn name(&self) -> String;
    fn summarize(&self, prompt: &str) -> Result<String>;
}

pub const DEFAULT_PROMPT: &str = include_str!("prompt.md");

pub struct PromptContext<'a> {
    pub date: String,
    pub time_range: String,
    pub app: String,
    pub title: Option<&'a str>,
    pub user_name: &'a str,
    pub me_label: &'a str,
    pub others_label: &'a str,
    pub language: &'a str,
}

/// Template, then a context block kute writes itself, then the transcript.
pub fn build_prompt(template: &str, ctx: &PromptContext, transcript: &str) -> String {
    let mut p = template.trim_end().to_string();
    p.push_str("\n\n## Context\n\n");
    p.push_str(&format!(
        "- Date: {}\n- Time: {}\n- App: {}\n",
        ctx.date, ctx.time_range, ctx.app
    ));
    if let Some(title) = ctx.title {
        p.push_str(&format!("- Title: {title}\n"));
    }
    let who = match ctx.user_name.trim() {
        "" => "the person who recorded the call".to_string(),
        name => name.to_string(),
    };
    p.push_str(&format!(
        "- \"{}\" is {who}. \"{}\" is everyone else on the call.\n",
        ctx.me_label, ctx.others_label
    ));
    p.push_str(&format!("- {}\n", language_instruction(ctx.language)));
    p.push_str("\n## Transcript\n\n");
    p.push_str(transcript.trim_end());
    p.push('\n');
    p
}

pub fn language_instruction(code: &str) -> String {
    if code == "auto" {
        return "Write the notes in the dominant language of the transcript.".into();
    }
    let name = match code {
        "en" => "English",
        "it" => "Italian",
        "de" => "German",
        "fr" => "French",
        "es" => "Spanish",
        "pt" => "Portuguese",
        "nl" => "Dutch",
        other => return format!("Write the notes in the language with code \"{other}\"."),
    };
    format!("Write the notes in {name}.")
}

pub fn load_template(cfg: &SummaryConfig) -> Result<String> {
    match &cfg.prompt {
        Some(path) => {
            let path = expand_tilde(path);
            std::fs::read_to_string(&path)
                .with_context(|| format!("reading prompt template {}", path.display()))
        }
        None => Ok(DEFAULT_PROMPT.to_string()),
    }
}

pub fn from_config(cfg: &SummaryConfig) -> Result<Option<Box<dyn Summarizer>>> {
    let timeout = cfg.timeout.unsigned_abs();
    Ok(match cfg.backend {
        SummaryBackend::None => None,
        SummaryBackend::Command => {
            if cfg.command.is_empty() {
                bail!(
                    "summary.backend = \"command\" needs summary.command, e.g. [\"claude\", \"-p\"]"
                );
            }
            Some(Box::new(command::CommandSummarizer::new(
                cfg.command.clone(),
                timeout,
            )))
        }
        SummaryBackend::Openai | SummaryBackend::Anthropic => {
            bail!("this summary backend is added in the next task")
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(user: &'a str, language: &'a str) -> PromptContext<'a> {
        PromptContext {
            date: "2026-10-08".into(),
            time_range: "09:40 to 09:54".into(),
            app: "firefox".into(),
            title: Some("Sync"),
            user_name: user,
            me_label: "Me",
            others_label: "Others",
            language,
        }
    }

    #[test]
    fn prompt_has_template_context_and_transcript() {
        let p = build_prompt("Do notes.\n", &ctx("Alex", "it"), "[09:40:11] Me: Ciao\n");
        assert!(p.starts_with("Do notes.\n\n## Context\n"));
        assert!(p.contains(
            "- Date: 2026-10-08\n- Time: 09:40 to 09:54\n- App: firefox\n- Title: Sync\n"
        ));
        assert!(p.contains("\"Me\" is Alex. \"Others\" is everyone else on the call."));
        assert!(p.contains("Write the notes in Italian."));
        assert!(p.ends_with("## Transcript\n\n[09:40:11] Me: Ciao\n"));
    }

    #[test]
    fn prompt_without_user_name_or_fixed_language() {
        let p = build_prompt(DEFAULT_PROMPT, &ctx("  ", "auto"), "x");
        assert!(p.contains("\"Me\" is the person who recorded the call."));
        assert!(p.contains("dominant language of the transcript"));
        assert_eq!(
            language_instruction("xx"),
            "Write the notes in the language with code \"xx\"."
        );
    }

    #[test]
    fn from_config_none_and_bad_command() {
        let mut cfg = SummaryConfig::default();
        assert!(from_config(&cfg).unwrap().is_none());
        cfg.backend = SummaryBackend::Command;
        assert!(format!("{:#}", from_config(&cfg).err().unwrap()).contains("summary.command"));
        cfg.command = vec!["cat".into()];
        assert_eq!(from_config(&cfg).unwrap().unwrap().name(), "command: cat");
    }

    #[test]
    fn custom_template_is_read_from_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("prompt.md");
        std::fs::write(&p, "Custom.").unwrap();
        let cfg = SummaryConfig {
            prompt: Some(p),
            ..Default::default()
        };
        assert_eq!(load_template(&cfg).unwrap(), "Custom.");
        assert_eq!(
            load_template(&SummaryConfig::default()).unwrap(),
            DEFAULT_PROMPT
        );
    }
}
