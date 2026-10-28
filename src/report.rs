use crate::{
    cli::Backend,
    config::Source,
    record::{EventType, IndexMode, Provider, Record, Role, SourceLocation},
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Ready,
    Disabled,
    Partial,
    Pending,
    CatchingUp,
    Uninitialized,
    Blocked,
    Unavailable,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Completed,
    CompletedWithSourceFailures,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub path: PathBuf,
    pub detail: String,
}
impl Diagnostic {
    pub fn new(path: &std::path::Path, detail: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            detail: detail.into(),
        }
    }
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum ImportSource {
    Parsed {
        path: PathBuf,
        events: usize,
        rejected_records: usize,
        rejection_diagnostics: Vec<Diagnostic>,
    },
    Unchanged {
        path: PathBuf,
        unchanged: bool,
    },
    Failed(Diagnostic),
}
#[derive(Serialize)]
pub struct ImportReport {
    pub outcome: Outcome,
    pub sources: Vec<ImportSource>,
}
impl ImportReport {
    pub fn completed(&self) -> bool {
        matches!(self.outcome, Outcome::Completed)
    }
}

#[derive(Serialize)]
pub struct LexicalStatus {
    pub state: State,
    pub active: bool,
    pub indexed_events: u64,
    pub indexed_inputs: usize,
    pub discovered_inputs: usize,
    pub pending_inputs: usize,
    pub removed_inputs: usize,
    pub errors: Vec<Diagnostic>,
}
#[derive(Serialize)]
pub struct Coverage {
    pub total: usize,
    pub ready: usize,
    pub pending: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub searchable: Option<usize>,
}
#[derive(Serialize)]
pub struct SemanticStatus {
    pub enabled: bool,
    pub active: bool,
    pub executor: String,
    pub model: String,
    pub state: State,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documents: Option<Coverage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<Coverage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}
#[derive(Serialize)]
#[serde(untagged)]
pub enum Status<T> {
    Available(T),
    Unavailable { state: State, reason: String },
}
impl<T> From<anyhow::Result<T>> for Status<T> {
    fn from(result: anyhow::Result<T>) -> Self {
        match result {
            Ok(value) => Self::Available(value),
            Err(error) => Self::Unavailable {
                state: State::Unavailable,
                reason: format!("{error:#}"),
            },
        }
    }
}

#[derive(Serialize)]
pub struct SearchHit {
    pub trwl_event_id: String,
    pub trwl_session_id: String,
    pub provider: Provider,
    pub provider_session_id: String,
    pub timestamp: String,
    pub workspace_root: Option<crate::workspace::Root>,
    pub event_type: EventType,
    pub snippet: String,
    pub rank: usize,
    pub score: f32,
}
#[derive(Serialize)]
pub struct Retrieval {
    pub effective_mode: Backend,
    pub semantic_status: State,
    pub reason: Option<String>,
}
#[derive(Serialize)]
pub struct SearchReport {
    pub query: String,
    pub results: Vec<SearchHit>,
    pub retrieval: Retrieval,
}
#[derive(Serialize)]
pub struct Location {
    pub trwl_event_id: String,
    pub trwl_session_id: String,
    pub provider: Provider,
    pub provider_session_id: String,
    pub provider_event_id: String,
    pub parent_trwl_session_id: Option<String>,
    pub source: SourceLocation,
    pub workspace: String,
    pub workspace_root: Option<crate::workspace::Root>,
}
#[derive(Clone, Serialize)]
pub struct OutputRecord {
    #[serde(rename = "trwl_event_id")]
    pub event_id: String,
    #[serde(rename = "trwl_session_id")]
    pub session_id: String,
    pub provider: Provider,
    pub provider_session_id: String,
    pub provider_event_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_message_id: Option<String>,
    #[serde(rename = "parent_trwl_session_id")]
    pub parent_session_id: Option<String>,
    pub event_type: EventType,
    pub role: Option<Role>,
    pub occurred_at: String,
    pub text: String,
    pub structured_content: serde_json::Value,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub details: Vec<serde_json::Value>,
    pub workspace: String,
    pub workspace_root: Option<crate::workspace::Root>,
    pub source: SourceLocation,
    pub files: Vec<String>,
    pub sequence: u64,
    pub is_subagent: bool,
}

impl Record {
    pub fn project(self) -> anyhow::Result<OutputRecord> {
        anyhow::ensure!(self.has_ordinary_content(), "record reference not found");
        let text = self.source_spans.as_ref().map_or_else(
            || Ok(self.text.clone()),
            |spans| spans.project_text(&self.text),
        )?;
        let projection = match &self.source_spans {
            Some(spans) => spans.project_native(&self)?,
            None => crate::redaction::NativeProjection {
                value: self.structured_content,
                details: Vec::new(),
            },
        };
        let mut structured_content = projection.value;
        // Tool payload keys and values are owned by the tool, not the envelope.
        if !matches!(self.event_type, EventType::ToolCall | EventType::ToolOutput) {
            crate::redaction::neutral_metadata(
                self.provider,
                self.native_kind,
                &mut structured_content,
            );
        }
        Ok(OutputRecord {
            event_id: self.event_id,
            session_id: self.session_id,
            provider: self.provider,
            provider_session_id: self.provider_session_id,
            provider_event_id: self.provider_event_id,
            provider_message_id: self.provider_message_id,
            parent_session_id: self.parent_session_id,
            event_type: self.event_type,
            role: self.role,
            occurred_at: self.occurred_at,
            text,
            structured_content,
            details: projection.details,
            workspace: self.workspace,
            workspace_root: self.workspace_root,
            source: self.source,
            files: self.files,
            sequence: self.sequence,
            is_subagent: self.is_subagent,
        })
    }
}

#[derive(Serialize)]
pub struct SelectedEvents {
    pub event: OutputRecord,
    pub events: Vec<OutputRecord>,
}
#[derive(Serialize)]
pub struct SessionEvents {
    pub session: Location,
    pub events: Vec<OutputRecord>,
}
#[derive(Serialize)]
pub struct EventsPage {
    pub events: Vec<OutputRecord>,
    pub next_cursor: Option<String>,
}
#[derive(Serialize)]
pub struct SessionsPage {
    pub sessions: Vec<Location>,
    pub next_cursor: Option<String>,
}
#[derive(Serialize)]
pub struct Running {
    pub running: bool,
}
#[derive(Serialize)]
pub struct OverallStatus {
    pub directories: crate::directories::Directories,
    pub indexing_mode: IndexMode,
    pub daemon: Running,
    pub ready: bool,
    pub lexical: Status<LexicalStatus>,
    pub semantic: Status<SemanticStatus>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum Report {
    Search(SearchReport),
    ShowEvent(SelectedEvents),
    ShowSession(SessionEvents),
    Events(EventsPage),
    Sessions(SessionsPage),
    Location(Location),
    Import(ImportReport),
    Sources {
        sources: Vec<Source>,
    },
    SourceAdded(Source),
    SourceRemoved {
        removed: String,
    },
    SemanticEnabled {
        semantic: bool,
        executor: String,
        model: String,
    },
    SemanticDisabled {
        semantic: bool,
    },
    Lexical(Status<LexicalStatus>),
    Semantic(Status<SemanticStatus>),
    Mode {
        mode: IndexMode,
    },
    Stopped {
        stopped: bool,
    },
    Daemon(Running),
    Setup {
        setup: bool,
    },
    Status(OverallStatus),
}
