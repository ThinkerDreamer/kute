//! Anthropic Messages API over raw HTTP (there is no official Rust SDK).

use super::{Summarizer, http_agent};
use anyhow::{Context, Result, bail};
use std::time::Duration;

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";
const MAX_TOKENS: u32 = 16_000;

pub struct AnthropicSummarizer {
    base_url: String,
    model: String,
    api_key: String,
    agent: ureq::Agent,
}

impl AnthropicSummarizer {
    pub fn new(
        base_url: String,
        model: String,
        api_key: String,
        timeout: Duration,
    ) -> AnthropicSummarizer {
        AnthropicSummarizer {
            base_url,
            model,
            api_key,
            agent: http_agent(timeout),
        }
    }
}

impl Summarizer for AnthropicSummarizer {
    fn name(&self) -> String {
        format!("anthropic: {}", self.model)
    }

    fn summarize(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        // `fallbacks: "default"` lets the API retry a declined request on the model
        // Anthropic recommends for that refusal category.
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "fallbacks": "default",
            "messages": [{"role": "user", "content": prompt}],
        });
        let mut response = self
            .agent
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "server-side-fallback-2026-07-01")
            .send_json(&body)
            .with_context(|| format!("POST {url}"))?;
        let status = response.status().as_u16();
        let text = response.body_mut().read_to_string()?;
        if !(200..300).contains(&status) {
            bail!(
                "Anthropic API returned HTTP {status}: {}",
                text.chars().take(500).collect::<String>()
            );
        }
        let value: serde_json::Value =
            serde_json::from_str(&text).context("the Anthropic API returned invalid JSON")?;
        match value["stop_reason"].as_str() {
            Some("refusal") => bail!(
                "the model declined to summarize this conversation (category: {})",
                value["stop_details"]["category"]
                    .as_str()
                    .unwrap_or("unknown")
            ),
            Some("max_tokens") => bail!("the summary was cut off at max_tokens"),
            _ => {}
        }
        let summary: String = value["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|block| block["type"] == "text")
            .filter_map(|block| block["text"].as_str())
            .collect();
        let summary = summary.trim();
        if summary.is_empty() {
            bail!("the Anthropic API returned no text");
        }
        Ok(summary.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Matcher;

    fn summarizer(server: &mockito::Server) -> AnthropicSummarizer {
        AnthropicSummarizer::new(
            server.url(),
            DEFAULT_MODEL.into(),
            "k".into(),
            Duration::from_secs(5),
        )
    }

    #[test]
    fn sends_required_headers_and_reads_text_blocks_only() {
        let mut server = mockito::Server::new();
        let m = server
            .mock("POST", "/v1/messages")
            .match_header("x-api-key", "k")
            .match_header("anthropic-version", "2023-06-01")
            .match_header("anthropic-beta", "server-side-fallback-2026-07-01")
            .match_body(Matcher::PartialJson(serde_json::json!({
                "model": "claude-opus-5-5",
                "max_tokens": 16000,
                "fallbacks": "default",
                "messages": [{"role": "user", "content": "the prompt"}]
            })))
            .with_body(
                r###"{"stop_reason":"end_turn","content":[
                    {"type":"thinking","thinking":""},
                    {"type":"text","text":"## Summary\n\n"},
                    {"type":"text","text":"Done."}]}"###,
            )
            .create();
        assert_eq!(
            summarizer(&server).summarize("the prompt").unwrap(),
            "## Summary\n\nDone."
        );
        assert_eq!(summarizer(&server).name(), "anthropic: claude-opus-5-5");
        m.assert();
    }

    #[test]
    fn refusal_is_an_error() {
        let mut server = mockito::Server::new();
        server
            .mock("POST", "/v1/messages")
            .with_body(r#"{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber"},"content":[]}"#)
            .create();
        let msg = format!("{:#}", summarizer(&server).summarize("p").unwrap_err());
        assert!(msg.contains("declined") && msg.contains("cyber"), "{msg}");
    }

    #[test]
    fn truncated_output_is_an_error() {
        let mut server = mockito::Server::new();
        server
            .mock("POST", "/v1/messages")
            .with_body(r#"{"stop_reason":"max_tokens","content":[{"type":"text","text":"half"}]}"#)
            .create();
        assert!(
            format!("{:#}", summarizer(&server).summarize("p").unwrap_err()).contains("max_tokens")
        );
    }

    #[test]
    fn http_errors_include_status_and_body() {
        let mut server = mockito::Server::new();
        server
            .mock("POST", "/v1/messages")
            .with_status(401)
            .with_body(r#"{"error":{"message":"invalid x-api-key"}}"#)
            .create();
        let msg = format!("{:#}", summarizer(&server).summarize("p").unwrap_err());
        assert!(
            msg.contains("401") && msg.contains("invalid x-api-key"),
            "{msg}"
        );
    }
}
