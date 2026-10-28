use crate::{
    cli::{Backend, Format},
    report::*,
    workspace::Root,
};
use anyhow::Result;
use serde::Serialize;
use serde_json::{Value, json};

pub fn render(report: &Report, format: Format) -> Result<String> {
    match format {
        Format::Json => json_line(report),
        Format::Jsonl => jsonl(report),
        Format::Text => Ok(text(report, "")),
        Format::Markdown => Ok(text(report, "## ")),
    }
}

fn json_line(value: &impl Serialize) -> Result<String> {
    let mut output = serde_json::to_string(value)?;
    output.push('\n');
    Ok(output)
}
fn collection<T: Serialize>(
    rows: &[T],
    item: &str,
    row_type: &str,
    completion: Value,
) -> Result<String> {
    let mut output = String::new();
    for row in rows {
        output.push_str(&json_line(&json!({"record_type":row_type,item:row}))?);
    }
    output.push_str(&json_line(&completion)?);
    Ok(output)
}
fn event_completion(cursor: &Option<String>) -> Value {
    json!({"record_type":"event_range_completion", "next_cursor":cursor,"terminal":cursor.is_none(),"truncated":cursor.is_some()})
}
fn jsonl(report: &Report) -> Result<String> {
    match report {
        Report::Search(r) => collection(
            &r.results,
            "result",
            "search_result",
            json!({"record_type":"search_completion","query":r.query,"retrieval":r.retrieval}),
        ),
        Report::ShowEvent(r) => {
            // `show event` repeats the selected event outside its event range.
            // Keep selection identity without repeating a full transcript record.
            let mut completion = event_completion(&None);
            completion["trwl_event_id"] = r.event.event_id.clone().into();
            collection(&r.events, "event", "event_range_event", completion)
        }
        Report::ShowSession(r) => {
            let mut completion = event_completion(&None);
            completion["session"] = serde_json::to_value(&r.session)?;
            collection(&r.events, "event", "event_range_event", completion)
        }
        Report::Events(r) => collection(
            &r.events,
            "event",
            "event_range_event",
            event_completion(&r.next_cursor),
        ),
        Report::Sessions(r) => collection(
            &r.sessions,
            "session",
            "session_range_session",
            json!({
                "record_type": "session_range_completion",
                "next_cursor": r.next_cursor,
                "terminal": r.next_cursor.is_none(),
                "truncated": r.next_cursor.is_some(),
            }),
        ),
        Report::Sources { sources } => collection(
            sources,
            "source",
            "source",
            json!({"record_type":"sources_completion"}),
        ),
        Report::Import(r) => collection(
            &r.sources,
            "source",
            "source",
            json!({"record_type":"sources_completion","outcome":r.outcome}),
        ),
        _ => json_line(report),
    }
}
fn state(state: State) -> &'static str {
    match state {
        State::Ready => "ready",
        State::Disabled => "disabled",
        State::Partial => "partial",
        State::Pending => "pending",
        State::CatchingUp => "catching up",
        State::Uninitialized => "uninitialized",
        State::Blocked => "blocked",
        State::Unavailable => "unavailable",
    }
}
fn backend(mode: Backend) -> &'static str {
    match mode {
        Backend::Lexical => "lexical",
        Backend::Semantic => "semantic",
        Backend::Hybrid => "hybrid",
    }
}
fn running(active: bool) -> &'static str {
    if active { "running" } else { "stopped" }
}
fn optional_line(output: &mut String, label: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|s| !s.is_empty()) {
        output.push_str(&format!("{label}: {value}\n"));
    }
}
fn text(report: &Report, heading: &str) -> String {
    match report {
        Report::Search(r) => search(r, heading),
        Report::ShowEvent(r) => events(&r.events, None, heading),
        Report::ShowSession(r) => events(&r.events, None, heading),
        Report::Events(r) => events(&r.events, r.next_cursor.as_deref(), heading),
        Report::Sessions(r) => sessions(r, heading),
        Report::Location(r) => location(r),
        Report::Import(r) => import(r),
        Report::Sources { sources } => {
            if sources.is_empty() {
                "No sources.\n".into()
            } else {
                sources
                    .iter()
                    .map(|s| format!("{} ({}): {}\n", s.name, s.provider, s.root.display()))
                    .collect()
            }
        }
        Report::SourceAdded(s) => format!(
            "Added source {} ({}): {}\n",
            s.name,
            s.provider,
            s.root.display()
        ),
        Report::SourceRemoved { removed } => format!("Removed source {removed}.\n"),
        Report::SemanticEnabled {
            executor, model, ..
        } => format!("Semantic search enabled: {model} ({executor})\n"),
        Report::SemanticDisabled { .. } => "Semantic search disabled.\n".into(),
        Report::Lexical(r) => lexical_status(r, heading),
        Report::Semantic(r) => semantic_status(r, heading),
        Report::Mode { mode } => format!("Indexing mode: {mode}\n"),
        Report::Stopped { .. } => "Updater stopped.\n".into(),
        Report::Daemon(r) => format!("Updater: {}\n", running(r.running)),
        Report::Setup { .. } => "Setup complete.\n".into(),
        Report::Status(r) => format!(
            "Config directory: {}\nData directory: {}\nCache directory: {}\nState directory: {}\nRuntime directory: {}\nIndexing: {}; updater {}\nReady: {}\n\n{}\n{}",
            r.directories.config.display(),
            r.directories.data.display(),
            r.directories.cache.display(),
            r.directories.state.display(),
            r.directories.runtime.display(),
            r.indexing_mode,
            running(r.daemon.running),
            if r.ready { "yes" } else { "no" },
            lexical_status(&r.lexical, heading),
            semantic_status(&r.semantic, heading)
        ),
    }
}
fn workspace_root_text(root: Option<&Root>) -> &str {
    root.map(Root::text).unwrap_or("unresolved")
}

fn search(r: &SearchReport, heading: &str) -> String {
    let mut output = format!(
        "Search: {}\nRetrieval: {} (semantic: {})\n",
        r.query,
        backend(r.retrieval.effective_mode),
        state(r.retrieval.semantic_status)
    );
    optional_line(&mut output, "Reason", r.retrieval.reason.as_deref());
    output.push('\n');
    if r.results.is_empty() {
        output.push_str("No results.\n");
    }
    for row in &r.results {
        output.push_str(&format!(
            "{heading}{}. {}\n{} · {} · {}\nSession: {}\nWorkspace root: {}\n\n{}\n\n",
            row.rank,
            row.trwl_event_id,
            row.provider,
            row.timestamp,
            row.event_type,
            row.trwl_session_id,
            workspace_root_text(row.workspace_root.as_ref()),
            row.snippet
        ));
    }
    output
}
fn events(rows: &[OutputRecord], cursor: Option<&str>, heading: &str) -> String {
    let mut output = String::new();
    if rows.is_empty() {
        output.push_str("No events.\n");
    }
    for r in rows {
        output.push_str(&format!(
            "{heading}{} {}\n{} · {} · {}\nSession: {}\n\n{}\n\n",
            r.event_id,
            r.role
                .as_ref()
                .map(crate::record::Role::as_str)
                .unwrap_or("-"),
            r.provider,
            r.occurred_at,
            r.event_type,
            r.session_id,
            r.text
        ));
    }
    optional_line(&mut output, "Next cursor", cursor);
    output
}
fn sessions(r: &SessionsPage, heading: &str) -> String {
    let mut output = String::new();
    if r.sessions.is_empty() {
        output.push_str("No sessions.\n");
    }
    for s in &r.sessions {
        output.push_str(&format!(
            "{heading}{}\n{} · {}\nWorkspace: {}\nWorkspace root: {}\n\n",
            s.trwl_session_id,
            s.provider,
            s.provider_session_id,
            s.workspace,
            workspace_root_text(s.workspace_root.as_ref())
        ));
    }
    optional_line(&mut output, "Next cursor", r.next_cursor.as_deref());
    output
}
fn location(r: &Location) -> String {
    let mut output = format!(
        "Event: {}\nSession: {}\nProvider: {}\nProvider session: {}\nSource: {} ({})\nWorkspace: {}\n",
        r.trwl_event_id,
        r.trwl_session_id,
        r.provider,
        r.provider_session_id,
        r.source.path.display(),
        r.source.format,
        r.workspace
    );
    optional_line(
        &mut output,
        "Workspace root",
        Some(workspace_root_text(r.workspace_root.as_ref())),
    );
    optional_line(&mut output, "Provider event", Some(&r.provider_event_id));
    optional_line(
        &mut output,
        "Parent session",
        r.parent_trwl_session_id.as_deref(),
    );
    output
}
fn import(r: &ImportReport) -> String {
    let mut output = format!(
        "Import: {}\n",
        if r.completed() {
            "completed"
        } else {
            "completed with source failures"
        }
    );
    for s in &r.sources {
        match s {
            ImportSource::Unchanged { path, .. } => {
                output.push_str(&format!("{}: unchanged\n", path.display()))
            }
            ImportSource::Failed(error) => {
                output.push_str(&format!("{}: {}\n", error.path.display(), error.detail))
            }
            ImportSource::Parsed {
                path,
                events,
                rejected_records,
                rejection_diagnostics,
            } => {
                output.push_str(&format!("{}: {events} events\n", path.display()));
                for diagnostic in rejection_diagnostics {
                    output.push_str(&format!("  {}\n", diagnostic.detail));
                }
                if *rejected_records > rejection_diagnostics.len() {
                    output.push_str(&format!("  {rejected_records} rejected records in total\n"));
                }
            }
        }
    }
    output
}
fn worker(output: &mut String, active: bool, reason: Option<&str>) {
    output.push_str(if active {
        "  Worker: active\n"
    } else {
        "  Worker: idle\n"
    });
    optional_line(output, "  Reason", reason);
}
fn lexical_status(r: &Status<LexicalStatus>, heading: &str) -> String {
    match r {
        Status::Unavailable { reason, .. } => {
            format!("{heading}Lexical: unavailable\n  Reason: {reason}\n")
        }
        Status::Available(r) => {
            let mut output = format!(
                "{heading}Lexical: {}\n  Indexed events: {}; inputs: {}\n  Source files: {}; new/changed: {}, removed: {}\n",
                state(r.state),
                r.indexed_events,
                r.indexed_inputs,
                r.discovered_inputs,
                r.pending_inputs,
                r.removed_inputs
            );
            worker(&mut output, r.active, None);
            for error in &r.errors {
                output.push_str(&format!("  {}: {}\n", error.path.display(), error.detail));
            }
            output
        }
    }
}
fn semantic_status(r: &Status<SemanticStatus>, heading: &str) -> String {
    match r {
        Status::Unavailable { reason, .. } => {
            format!("{heading}Semantic: unavailable\n  Reason: {reason}\n")
        }
        Status::Available(r) => {
            let mut output = format!("{heading}Semantic: {}\n", state(r.state));
            if let (Some(d), Some(c)) = (&r.documents, &r.chunks) {
                output.push_str(&format!(
                "  Documents: {} / {} complete ({} searchable)\n  Chunks: {} / {} embedded; {} pending\n",
                d.ready,d.total,d.searchable.unwrap_or(0),c.ready,c.total,c.pending));
            }
            optional_line(&mut output, "  Model", Some(&r.model));
            optional_line(&mut output, "  Executor", Some(&r.executor));
            worker(&mut output, r.active, r.reason.as_deref());
            output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query,
        record::{EventType, IndexMode, Provider},
    };
    use std::path::Path;

    fn record() -> crate::record::Record {
        let mut r = crate::record::Record::new(
            Provider::Codex,
            "native",
            "event",
            EventType::Message,
            Path::new("input.jsonl"),
        );
        r.text = "Complete transcript 日本語 🦀".into();
        r.role = Some("user".into());
        r
    }

    #[test]
    fn selected_event_jsonl_keeps_identity_without_repeating_selected_payload() {
        let r = record();
        let expected = serde_json::to_value(&r).unwrap();
        let id = r.event_id.clone();
        let report = Report::ShowEvent(SelectedEvents {
            event: r.clone().project().unwrap(),
            events: vec![r.project().unwrap()],
        });
        let lines: Vec<Value> = render(&report, Format::Jsonl)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            json!({"record_type":"event_range_event","event":expected})
        );
        assert_eq!(
            lines[1],
            json!({"record_type":"event_range_completion","trwl_event_id":id,"next_cursor":null,"terminal":true,"truncated":false})
        );
        let plain = render(&report, Format::Text).unwrap();
        let markdown = render(&report, Format::Markdown).unwrap();
        assert!(plain.contains("Complete transcript 日本語 🦀"));
        assert!(markdown.starts_with("## "));
        assert!(markdown.contains("Complete transcript 日本語 🦀"));
    }

    #[test]
    fn pages_and_empty_collections_have_completion_records() {
        let r = record();
        let report = Report::Events(EventsPage {
            events: vec![r.project().unwrap()],
            next_cursor: Some("next-event".into()),
        });
        let lines: Vec<Value> = render(&report, Format::Jsonl)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[1],
            json!({"record_type":"event_range_completion","next_cursor":"next-event","terminal":false,"truncated":true})
        );
        for (report, completion) in [
            (
                Report::Events(EventsPage {
                    events: vec![],
                    next_cursor: None,
                }),
                "event_range_completion",
            ),
            (
                Report::Sessions(SessionsPage {
                    sessions: vec![],
                    next_cursor: None,
                }),
                "session_range_completion",
            ),
            (Report::Sources { sources: vec![] }, "sources_completion"),
            (
                Report::Search(SearchReport {
                    query: "missing".into(),
                    results: vec![],
                    retrieval: Retrieval {
                        effective_mode: Backend::Lexical,
                        semantic_status: State::Disabled,
                        reason: None,
                    },
                }),
                "search_completion",
            ),
        ] {
            let rendered = render(&report, Format::Jsonl).unwrap();
            assert_eq!(rendered.lines().count(), 1);
            assert_eq!(
                serde_json::from_str::<Value>(&rendered).unwrap()["record_type"],
                completion
            );
        }
    }

    #[test]
    fn location_uses_plain_provider_identity_in_every_format() {
        let report = Report::Location(query::locate(&record()));
        for format in [Format::Json, Format::Jsonl] {
            let value: Value = serde_json::from_str(&render(&report, format).unwrap()).unwrap();
            assert_eq!(value["provider_event_id"], "event");
        }
        for format in [Format::Text, Format::Markdown] {
            assert!(
                render(&report, format)
                    .unwrap()
                    .contains("Provider event: event\n")
            );
        }
        assert_eq!(
            render(
                &Report::Mode {
                    mode: IndexMode::Manual
                },
                Format::Text
            )
            .unwrap(),
            "Indexing mode: manual\n"
        );
    }
}
