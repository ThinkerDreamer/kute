//! Any OpenAI-compatible Chat Completions endpoint (Ollama, llama.cpp, vLLM, OpenRouter...).

use super::{Summarizer, http_agent};
use anyhow::{Context, Result, bail};
use std::time::Duration;

pub const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

pub struct OpenAiSummarizer {
    base_url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl OpenAiSummarizer {
    pub fn new(
        base_url: String,
        model: String,
        api_key: Option<String>,
        timeout: Duration,
    ) -> OpenAiSummarizer {
        OpenAiSummarizer {
            base_url,
            model,
            api_key,
            agent: http_agent(timeout),
        }
    }
}

impl Summarizer for OpenAiSummarizer {
    fn name(&self) -> String {
        format!("openai: {}", self.model)
    }

    fn summarize(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        });
        let mut request = self.agent.post(&url);
        if let Some(key) = &self.api_key {
            request = request.header("authorization", format!("Bearer {key}"));
        }
        let mut response = request
            .send_json(&body)
            .with_context(|| format!("POST {url}"))?;
        let status = response.status().as_u16();
        let text = response.body_mut().read_to_string()?;
        if !(200..300).contains(&status) {
            bail!(
                "{url} returned HTTP {status}: {}",
                text.chars().take(500).collect::<String>()
            );
        }
        let value: serde_json::Value =
            serde_json::from_str(&text).context("the summarizer returned invalid JSON")?;
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .context("the response has no choices[0].message.content")?
            .trim();
        if content.is_empty() {
            bail!("the summarizer returned an empty answer");
        }
        Ok(content.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Matcher;

    #[test]
    fn posts_chat_completion_and_reads_content() {
        let mut server = mockito::Server::new();
        let m = server
            .mock("POST", "/v1/chat/completions")
            .match_header("authorization", "Bearer k")
            .match_body(Matcher::PartialJson(serde_json::json!({
                "model": "qwen3:8b",
                "messages": [{"role": "user", "content": "the prompt"}]
            })))
            .with_body(r#"{"choices":[{"message":{"role":"assistant","content":"  ## Summary\n\nok  "}}]}"#)
            .create();
        let s = OpenAiSummarizer::new(
            format!("{}/v1/", server.url()),
            "qwen3:8b".into(),
            Some("k".into()),
            Duration::from_secs(5),
        );
        assert_eq!(s.summarize("the prompt").unwrap(), "## Summary\n\nok");
        assert_eq!(s.name(), "openai: qwen3:8b");
        m.assert();
    }

    #[test]
    fn no_key_means_no_auth_header() {
        let mut server = mockito::Server::new();
        let m = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", Matcher::Missing)
            .with_body(r#"{"choices":[{"message":{"content":"x"}}]}"#)
            .create();
        let s = OpenAiSummarizer::new(server.url(), "m".into(), None, Duration::from_secs(5));
        s.summarize("p").unwrap();
        m.assert();
    }

    #[test]
    fn http_errors_include_status_and_body() {
        let mut server = mockito::Server::new();
        server
            .mock("POST", "/chat/completions")
            .with_status(500)
            .with_body("model not loaded")
            .create();
        let s = OpenAiSummarizer::new(server.url(), "m".into(), None, Duration::from_secs(5));
        let msg = format!("{:#}", s.summarize("p").unwrap_err());
        assert!(
            msg.contains("500") && msg.contains("model not loaded"),
            "{msg}"
        );
    }
}
