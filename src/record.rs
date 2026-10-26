use anyhow::{Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

macro_rules! string_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$attr:meta])* $variant:ident => $value:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
        pub enum $name {
            $(
                $(#[$attr])*
                #[serde(rename = $value)]
                #[value(name = $value)]
                $variant,
            )+
        }
        impl $name {
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)+
                }
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
        impl std::str::FromStr for $name {
            type Err = anyhow::Error;
            fn from_str(value: &str) -> Result<Self> {
                <Self as clap::ValueEnum>::from_str(value, false)
                    .map_err(anyhow::Error::msg)
            }
        }
    };
}

string_enum!(Provider {
    Opencode => "opencode",
    Codex => "codex",
    Pi => "pi",
    #[serde(alias = "claude-code")]
    #[value(alias = "claude-code")]
    Claude => "claude",
    #[serde(alias = "copilot-cli")]
    #[value(alias = "copilot-cli")]
    CopilotCli => "copilot_cli",
});
string_enum!(EventType {
    Message => "message",
    ToolCall => "tool_call",
    ToolOutput => "tool_output",
    Metadata => "metadata",
});
string_enum!(#[derive(Default)] ContentScope {
    #[default]
    All => "all",
    Transcript => "transcript",
    Calls => "calls",
    Outputs => "outputs",
});
string_enum!(IndexMode {
    Auto => "auto",
    Manual => "manual",
});
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "system")]
    System,
    #[serde(rename = "developer")]
    Developer,
    #[serde(rename = "toolResult")]
    ToolResult,
    #[serde(rename = "tool")]
    Tool,
    #[serde(untagged)]
    Other(String),
}
impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Developer => "developer",
            Self::ToolResult => "toolResult",
            Self::Tool => "tool",
            Self::Other(value) => value,
        }
    }
}
impl From<&str> for Role {
    fn from(value: &str) -> Self {
        match value {
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "system" => Self::System,
            "developer" => Self::Developer,
            "toolResult" => Self::ToolResult,
            "tool" => Self::Tool,
            other => Self::Other(other.into()),
        }
    }
}

// Search results bound snippets; stored records keep the full text.
pub const SEARCH_SNIPPET_CHARS: usize = 400;

pub fn is_text_part(part: &Value) -> bool {
    matches!(
        part["type"].as_str(),
        Some("text" | "input_text" | "output_text")
    )
}

/// Hash strings with length prefixes so field boundaries affect the digest.
pub fn digest(parts: &[&str]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part.as_bytes());
    }
    hex::encode(hash.finalize())
}

pub fn timestamp(value: &str) -> Result<String> {
    Ok(DateTime::parse_from_rfc3339(value)
        .with_context(|| format!("invalid timestamp {value:?}"))?
        .with_timezone(&Utc)
        .to_rfc3339_opts(SecondsFormat::Millis, true))
}

pub fn millis(value: i64) -> Result<String> {
    Ok(DateTime::from_timestamp_millis(value)
        .context("timestamp outside supported range")?
        .to_rfc3339_opts(SecondsFormat::Millis, true))
}

pub fn session_id(provider: Provider, native: &str) -> String {
    digest(&["session", provider.as_str(), native])
}

/// Best score first, then newest timestamp and finally ascending event ID.
pub fn relevance_order(a: (f32, &str, &str), b: (f32, &str, &str)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0)
        .then_with(|| b.1.cmp(a.1))
        .then_with(|| a.2.cmp(b.2))
}

/// Oldest timestamp first; source sequence and event ID break ties.
pub fn chronological_order(a: (&str, u64, &str), b: (&str, u64, &str)) -> std::cmp::Ordering {
    a.cmp(&b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeEventKind {
    CopilotAssistantMessage,
    CopilotAssistantDetail,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    #[serde(rename = "trwl_event_id")]
    pub event_id: String,
    #[serde(rename = "trwl_session_id")]
    pub session_id: String,
    pub provider: Provider,
    pub provider_session_id: String,
    pub provider_event_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_message_id: Option<String>,
    #[serde(rename = "parent_trwl_session_id")]
    pub parent_session_id: Option<String>,
    pub event_type: EventType,
    pub role: Option<Role>,
    pub occurred_at: String,
    pub text: String,
    pub structured_content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_spans: Option<crate::texture::SourceSpans>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_kind: Option<NativeEventKind>,
    pub workspace: String,
    pub workspace_root: Option<crate::workspace::Root>,
    pub source: SourceLocation,
    pub files: Vec<String>,
    pub sequence: u64,
    pub is_subagent: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceLocation {
    #[serde(rename = "source_format")]
    pub format: String,
    pub path: PathBuf,
}

/// The input to conversation pairing. Only user messages supply result metadata;
/// assistant payloads contribute visible text, grouping and ordering.
#[derive(Serialize)]
pub struct ConversationMessage {
    pub session_id: String,
    pub occurred_at: String,
    pub sequence: u64,
    pub event_id: String,
    pub provider_message_id: Option<String>,
    pub source: PathBuf,
    pub text: String,
    pub request: Option<Record>,
}

impl ConversationMessage {
    pub fn new(record: &Record) -> Option<Self> {
        let text = record.message_text()?.into_owned();
        let request = matches!(record.role, Some(Role::User)).then(|| {
            let mut request = record.clone();
            request.text = text.chars().take(SEARCH_SNIPPET_CHARS).collect();
            request.structured_content = Value::Null;
            request.source_spans = None;
            request
        });
        Some(Self {
            session_id: record.session_id.clone(),
            occurred_at: record.occurred_at.clone(),
            sequence: record.sequence,
            event_id: record.event_id.clone(),
            provider_message_id: record.provider_message_id.clone(),
            source: record.source.path.clone(),
            text,
            request,
        })
    }
}

impl Record {
    pub fn is_conversation_message(&self) -> bool {
        self.event_type == EventType::Message
            && matches!(self.role, Some(Role::User | Role::Assistant))
            && self
                .source_spans
                .as_ref()
                .is_none_or(|spans| spans.ordinary)
    }

    pub fn message_text(&self) -> Option<Cow<'_, str>> {
        if !self.is_conversation_message() {
            return None;
        }
        // A message's display text can include reasoning. Extract visible text
        // from its parts before pairing user messages with assistant replies.
        if let Some(text) = self.structured_content["content"].as_str() {
            Some(Cow::Borrowed(text))
        } else if self.native_kind == Some(NativeEventKind::CopilotAssistantMessage) {
            Some(Cow::Borrowed(""))
        } else if let Some(parts) = self.structured_content["content"].as_array() {
            Some(Cow::Owned(
                parts
                    .iter()
                    .filter(|part| is_text_part(part))
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        } else {
            Some(Cow::Borrowed(&self.text))
        }
    }

    pub fn new(
        provider: Provider,
        session: &str,
        native: &str,
        kind: EventType,
        path: &Path,
    ) -> Self {
        let sid = session_id(provider, session);
        Self {
            event_id: digest(&["event", &sid, native, kind.as_str()]),
            session_id: sid,
            provider,
            provider_session_id: session.into(),
            provider_event_id: native.into(),
            provider_message_id: None,
            parent_session_id: None,
            event_type: kind,
            role: None,
            occurred_at: String::new(),
            text: String::new(),
            structured_content: Value::Null,
            source_spans: None,
            native_kind: None,
            workspace: String::new(),
            workspace_root: None,
            source: SourceLocation {
                format: match provider {
                    Provider::Opencode => "opencode_sqlite",
                    Provider::Codex => "codex_rollout_jsonl",
                    Provider::Pi => "pi_jsonl",
                    Provider::Claude => "claude_code_jsonl",
                    Provider::CopilotCli => "copilot_cli_jsonl",
                }
                .into(),
                path: path.into(),
            },
            files: Vec::new(),
            sequence: 0,
            is_subagent: false,
        }
    }
}
