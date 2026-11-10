//! Provider readers that normalize native history into [`Record`]s.
//!
//! A parse returns a [`Checkpoint`] passed to the input's next parse, and
//! which previously indexed records its output replaces.
//!
//! OpenCode: one read transaction per parse. Saved row prefixes verify each
//! session and let the parser keep
//! an unchanged beginning and replace only the suffix, including sequence
//! numbers after an insertion or deletion. Deleted sessions are removed.
//!
//! JSONL: a parse resumes after the last complete line once the saved prefix
//! hash matches, so appends index only new records; edits and truncations
//! reparse the whole file. An incomplete trailing line is read again after the
//! next change. Checkpoints keep session metadata, record and line counts,
//! duplicate-identity counters and rejections (a count and the first few
//! diagnostics).

use crate::record::{
    EventType, NativeEventKind, Provider, Record, Role, digest, millis, session_id, timestamp,
};
use crate::redaction::{BlockArray, BlockCollection, Body, Opaque, OpaqueKind, Selection, Target};
use crate::report::Diagnostic;
use anyhow::{Context, Result, bail, ensure};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{BufRead, BufReader, Read, Seek},
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

/// Harness notices do not anchor a semantic conversation turn.
pub fn is_control_message(text: &str) -> bool {
    [
        "<environment_context>",
        "<turn_aborted>",
        "<subagent_notification>",
        "Warning: The maximum number of unified exec processes",
    ]
    .iter()
    .any(|prefix| text.starts_with(prefix))
}

pub struct Parsed {
    pub records: usize,
    pub diagnostics: Diagnostics,
    pub replace: Replace,
    pub checkpoint: Checkpoint,
}

/// Rejected records: a total count and the first few diagnostics, so parser
/// state does not grow with the number of rejections.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Diagnostics {
    pub count: usize,
    pub first: Vec<Diagnostic>,
}

impl Diagnostics {
    pub const LIMIT: usize = 10;

    fn push(&mut self, diagnostic: Diagnostic) {
        if self.first.len() < Self::LIMIT {
            self.first.push(diagnostic);
        }
        self.count += 1;
    }

    /// Equivalent to pushing every diagnostic `other` received.
    fn extend(&mut self, other: &Self) {
        let room = Self::LIMIT.saturating_sub(self.first.len());
        self.first.extend(other.first.iter().take(room).cloned());
        self.count += other.count;
    }
}

/// Which previously indexed records of an input the parsed records replace.
pub enum Replace {
    /// Every record of the input.
    All,
    /// None: the parsed records follow the previous ones.
    Append,
    /// Suffixes of native sessions; output for other sessions appends.
    Sessions(Vec<SessionSuffix>),
}

pub struct SessionSuffix {
    pub native_session: String,
    pub from_sequence: u64,
}

/// Parser state saved with an input's records and passed to its next parse.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Checkpoint {
    Jsonl(JsonlCheckpoint),
    /// OpenCode sessions by native session ID.
    Sqlite(BTreeMap<String, SqliteSession>),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct SqliteSession {
    records: usize,
    diagnostics: Diagnostics,
    prefixes: Vec<SqlitePrefix>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct SqlitePrefix {
    rows: usize,
    hash: String,
    records: usize,
    diagnostics: Diagnostics,
}

#[derive(Default)]
struct SqliteOutput {
    records: EmittedRecords,
    diagnostics: Diagnostics,
}

#[derive(Clone, Default)]
struct SqlitePosition {
    rows: usize,
    hasher: Sha256,
}

/// One verified boundary, including output counts that can differ from row counts.
#[derive(Default)]
struct VerifiedSqlitePrefix {
    position: SqlitePosition,
    output: SqliteOutput,
    prefixes: Vec<SqlitePrefix>,
}

struct SqliteResume<'stmt> {
    rows: rusqlite::Rows<'stmt>,
    verified: VerifiedSqlitePrefix,
    replacement: Option<SessionSuffix>,
}

impl VerifiedSqlitePrefix {
    fn advance(mut self, prefix: &SqlitePrefix, position: SqlitePosition) -> Self {
        self.prefixes.push(prefix.clone());
        Self {
            position,
            output: SqliteOutput {
                records: EmittedRecords(prefix.records),
                diagnostics: prefix.diagnostics.clone(),
            },
            prefixes: self.prefixes,
        }
    }

    fn restart<'stmt>(
        self,
        ordered: &'stmt mut rusqlite::Statement<'_>,
        id: &str,
    ) -> Result<SqliteResume<'stmt>> {
        let mut rows = ordered.query([id])?;
        for _ in 0..self.position.rows {
            rows.next()?
                .context("OpenCode prefix missing from snapshot")?;
        }
        Ok(self.replace(rows, id))
    }

    fn replace<'stmt>(self, rows: rusqlite::Rows<'stmt>, id: &str) -> SqliteResume<'stmt> {
        SqliteResume {
            rows,
            replacement: Some(SessionSuffix {
                native_session: id.to_owned(),
                from_sequence: self.output.records.0 as u64,
            }),
            verified: self,
        }
    }

    fn save_prefix(&mut self) {
        self.prefixes.push(SqlitePrefix {
            rows: self.position.rows,
            hash: hex::encode(self.position.hasher.clone().finalize()),
            records: self.output.records.0,
            diagnostics: self.output.diagnostics.clone(),
        });
    }

    fn capture(self) -> SqliteSession {
        SqliteSession {
            records: self.output.records.0,
            diagnostics: self.output.diagnostics,
            prefixes: self.prefixes,
        }
    }
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JsonlCheckpoint {
    offset: u64,
    prefix_hash: String,
    line: usize,
    session: String,
    workspace: String,
    parent: Option<String>,
    fallback_counts: HashMap<String, usize>,
    records: usize,
    diagnostics: Diagnostics,
}

pub fn files(provider: Provider, root: &Path) -> Result<Vec<PathBuf>> {
    ensure!(root.exists(), "input {} does not exist", root.display());
    if root.is_file() {
        return Ok(vec![root.canonicalize()?]);
    }
    let mut paths = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry.with_context(|| format!("reading {}", root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let wanted = match provider {
            Provider::Opencode => path.extension().is_some_and(|e| e == "db" || e == "sqlite"),
            Provider::CopilotCli => path.file_name().is_some_and(|n| n == "events.jsonl"),
            _ => path.extension().is_some_and(|e| e == "jsonl"),
        };
        if wanted {
            paths.push(path.canonicalize()?);
        }
    }
    paths.sort();
    Ok(paths)
}

pub fn parse(
    provider: Provider,
    path: &Path,
    previous: Option<&Checkpoint>,
    mut emit: impl FnMut(&Record) -> Result<()>,
) -> Result<Parsed> {
    let parsed = match (provider, previous) {
        (Provider::Opencode, Some(Checkpoint::Sqlite(sessions))) => {
            opencode(path, Some(sessions), &mut emit)
        }
        (Provider::Opencode, _) => opencode(path, None, &mut emit),
        (_, Some(Checkpoint::Jsonl(checkpoint))) => {
            jsonl(provider, path, Some(checkpoint), &mut emit)
        }
        _ => jsonl(provider, path, None, &mut emit),
    }
    .with_context(|| format!("parsing {} ({provider})", path.display()))?;
    ensure!(
        parsed.records != 0 || parsed.diagnostics.count == 0,
        "no accepted content in {}: {}",
        path.display(),
        parsed.diagnostics.first[0].detail
    );
    Ok(parsed)
}

/// Invalidate cached parses when normalized content or supported formats change.
pub fn parser_revision(provider: Provider) -> u32 {
    match provider {
        Provider::Opencode => 6,
        Provider::Claude => 5,
        Provider::Pi => 4,
        Provider::CopilotCli => 3,
        Provider::Codex => 2,
    }
}

fn text(value: &Value) -> String {
    text_parts(value).join("\n")
}

fn selected_part(provider: Provider, value: &Value) -> bool {
    selected_form(provider, value).is_some()
}

#[derive(Clone, Copy)]
enum SelectedForm {
    Opencode,
    Codex,
    Pi,
    Thinking,
    Redacted,
}

fn selected_form(provider: Provider, value: &Value) -> Option<SelectedForm> {
    match (provider, value["type"].as_str()?) {
        (Provider::Opencode, "reasoning") => Some(SelectedForm::Opencode),
        (Provider::Codex, "reasoning") => Some(SelectedForm::Codex),
        (Provider::Pi, "thinking") => Some(SelectedForm::Pi),
        (Provider::Claude, "thinking") => Some(SelectedForm::Thinking),
        (Provider::Claude, "redacted_thinking") => Some(SelectedForm::Redacted),
        _ => None,
    }
}

fn block_form(value: &Value) -> Option<SelectedForm> {
    match value["type"].as_str()? {
        "thinking" => Some(SelectedForm::Thinking),
        "redacted_thinking" => Some(SelectedForm::Redacted),
        "reasoning" => Some(SelectedForm::Codex),
        _ => None,
    }
}

fn select_part(body: &mut Body, pointer: String, value: &Value, form: SelectedForm) -> usize {
    let fields: &[(&str, OpaqueKind)] = match form {
        SelectedForm::Opencode => &[
            ("data", OpaqueKind::Data),
            ("encrypted_content", OpaqueKind::Data),
            ("signature", OpaqueKind::Signature),
            ("thinkingSignature", OpaqueKind::Signature),
        ],
        SelectedForm::Codex => &[("encrypted_content", OpaqueKind::Data)],
        SelectedForm::Pi => &[("thinkingSignature", OpaqueKind::Signature)],
        SelectedForm::Thinking => &[("signature", OpaqueKind::Signature)],
        SelectedForm::Redacted => &[("data", OpaqueKind::Data)],
    };
    let mut opaque: Vec<_> = fields
        .iter()
        .filter(|(key, _)| value.get(*key).is_some())
        .map(|(key, kind)| Opaque::Value {
            pointer: format!("{pointer}/{key}"),
            kind: *kind,
        })
        .collect();
    if matches!(form, SelectedForm::Opencode) {
        for key in ["metadata", "native"] {
            if value.get(key).is_some() {
                opaque.push(Opaque::NativeMap(format!("{pointer}/{key}")));
            }
        }
    }
    body.select(Selection {
        target: Target::Node(pointer),
        opaque,
    })
}

fn push_selected(body: &mut Body, node: usize, value: &Value, form: SelectedForm) {
    match form {
        SelectedForm::Codex => {
            for key in ["summary", "content"] {
                let parts = text_parts(&value[key]);
                if !parts.join("\n").is_empty() {
                    for text in parts {
                        body.push(text, Some(node));
                    }
                }
            }
        }
        SelectedForm::Opencode => body.push(&text(&value["text"]), Some(node)),
        SelectedForm::Pi | SelectedForm::Thinking => {
            if let Some(text) = value["thinking"].as_str() {
                body.push(text, Some(node));
            }
        }
        SelectedForm::Redacted => {}
    }
}

fn compose_copilot(record: &mut Record, kind: NativeEventKind) -> Result<()> {
    let value = &record.structured_content;
    let mut body = Body::default();
    if kind == NativeEventKind::CopilotAssistantDetail {
        let node = body.select(Selection {
            target: Target::Node(String::new()),
            opaque: Vec::new(),
        });
        body.push(
            value["content"]
                .as_str()
                .context("message content missing")?,
            Some(node),
        );
    } else {
        let members: Vec<_> = ["reasoningText", "reasoningOpaque", "encryptedContent"]
            .into_iter()
            .filter(|key| value.get(*key).is_some())
            .map(str::to_owned)
            .collect();
        if members.is_empty() && value.get("reasoningBlocks").is_none() {
            return Ok(());
        }
        if let Some(content) = value.get("content") {
            body.ordinary();
            for text in text_parts(content) {
                body.push(text, None);
            }
        }
        if !members.is_empty() {
            let opaque = [
                ("reasoningOpaque", OpaqueKind::Signature),
                ("encryptedContent", OpaqueKind::Data),
            ]
            .into_iter()
            .filter(|(key, _)| value.get(*key).is_some())
            .map(|(key, kind)| Opaque::Value {
                pointer: format!("/{key}"),
                kind,
            })
            .collect();
            let node = body.select(Selection {
                target: Target::Fields {
                    object: String::new(),
                    members,
                },
                opaque,
            });
            if let Some(text) = value["reasoningText"].as_str() {
                body.push(text, Some(node));
            }
        }
        if let Some(envelope) = value.get("reasoningBlocks") {
            let envelope = envelope.as_object().context("invalid content container")?;
            let mut arrays = Vec::new();
            for array in [BlockArray::Blocks, BlockArray::OrderedBlocks] {
                if let Some(parts) = envelope.get(array.name()) {
                    let parts = parts.as_array().context("invalid content array")?;
                    arrays.push(array);
                    for (position, part) in parts.iter().enumerate() {
                        if let Some(form) = block_form(part) {
                            let node = select_part(
                                &mut body,
                                format!("/reasoningBlocks/{}/{position}", array.name()),
                                part,
                                form,
                            );
                            push_selected(&mut body, node, part, form);
                        } else {
                            body.ordinary();
                        }
                    }
                }
            }
            if envelope
                .keys()
                .any(|key| !matches!(key.as_str(), "provider" | "blocks" | "orderedBlocks"))
            {
                body.ordinary();
            }
            body.collection(BlockCollection {
                pointer: "/reasoningBlocks".into(),
                arrays,
            });
        }
    }
    body.finish(record)
}

fn text_parts(value: &Value) -> Vec<&str> {
    if let Some(text) = value.as_str() {
        vec![text]
    } else {
        value
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|part| {
                part.get("text")
                    .or_else(|| part.get("thinking"))
                    .and_then(Value::as_str)
            })
            .collect()
    }
}

/// Retain provenance while composing native message parts. Tool records and
/// arbitrary objects inside them never enter this classifier.
fn compose_message(record: &mut Record) -> Result<()> {
    if record.event_type != EventType::Message {
        return Ok(());
    }
    if let Some(kind) = record.native_kind {
        return compose_copilot(record, kind);
    }
    let value = &record.structured_content;
    let mut body = Body::default();
    if matches!(record.provider, Provider::Opencode | Provider::Codex)
        && selected_part(record.provider, value)
    {
        let form = selected_form(record.provider, value).context("invalid content owner")?;
        let node = select_part(&mut body, String::new(), value, form);
        push_selected(&mut body, node, value, form);
    } else if let Some(parts) = value["content"].as_array() {
        if !(matches!(record.provider, Provider::Pi | Provider::Claude)
            || record.provider == Provider::Opencode && value["type"] == "assistant")
        {
            return Ok(());
        }
        if !parts
            .iter()
            .any(|part| selected_part(record.provider, part))
        {
            return Ok(());
        }
        // OpenCode v2 appends errors after its content, excluding an empty
        // composed content body. Other readers preserve empty text parts.
        let empty_content =
            record.provider == Provider::Opencode && text(&value["content"]).is_empty();
        for (position, part) in parts.iter().enumerate() {
            if let Some(form) = selected_form(record.provider, part) {
                let node = select_part(&mut body, format!("/content/{position}"), part, form);
                if !empty_content {
                    push_selected(&mut body, node, part, form);
                }
            } else {
                body.ordinary();
                if !empty_content
                    && let Some(text) = part
                        .get("text")
                        .or_else(|| part.get("thinking"))
                        .and_then(Value::as_str)
                {
                    body.push(text, None);
                }
            }
        }
        if record.provider == Provider::Opencode {
            let mut errors = Vec::new();
            strings(&value["error"], &mut errors);
            for error in errors.into_iter().filter(|text| !text.is_empty()) {
                body.ordinary();
                body.push(error, None);
            }
        }
    } else {
        return Ok(());
    }
    body.finish(record)
}

/// Collect the string values of a native part, except identifiers and inline
/// `data:` URLs, whose base64 content is not searchable text.
fn strings<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
    match value {
        Value::String(s) if !s.starts_with("data:") => out.push(s),
        Value::Array(items) => items.iter().for_each(|item| strings(item, out)),
        Value::Object(map) => {
            for (key, item) in map {
                if !matches!(key.as_str(), "id" | "sessionID" | "messageID" | "callID") {
                    strings(item, out);
                }
            }
        }
        _ => {}
    }
}

fn arguments(value: &Value) -> Cow<'_, str> {
    value
        .as_str()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(value.to_string()))
}

fn file_paths(tool: &str, args: &Value) -> Vec<String> {
    let mut paths = Vec::new();
    for key in ["file_path", "filePath", "path", "filename"] {
        if let Some(path) = args.get(key).and_then(Value::as_str) {
            paths.push(path.into());
        }
    }
    if tool.eq_ignore_ascii_case("apply_patch") {
        let patch = args
            .as_str()
            .or_else(|| args.get("patch").and_then(Value::as_str))
            .or_else(|| args.get("patchText").and_then(Value::as_str))
            .unwrap_or("");
        for line in patch.lines() {
            for prefix in [
                "*** Add File: ",
                "*** Update File: ",
                "*** Delete File: ",
                "*** Move to: ",
            ] {
                if let Some(path) = line.strip_prefix(prefix) {
                    paths.push(path.to_owned());
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpencodeSchema {
    V1,
    V2 { has_parts: bool },
}

impl OpencodeSchema {
    fn ordered_sql(self) -> &'static str {
        match self {
            Self::V1 => {
                "SELECT p.rowid FROM part p JOIN message m ON p.message_id = m.id
                 WHERE p.session_id = ?1 ORDER BY p.time_created, p.id"
            }
            Self::V2 { .. } => {
                "SELECT rowid FROM session_message WHERE session_id = ?1 ORDER BY seq"
            }
        }
    }

    fn payload_sql(self) -> String {
        match self {
            Self::V1 => {
                "SELECT p.id, p.data, p.time_created, m.data, s.directory, s.parent_id, m.id
                 FROM part p JOIN message m ON p.message_id = m.id
                 JOIN session s ON p.session_id = s.id WHERE p.rowid = ?1"
                    .into()
            }
            Self::V2 { has_parts } => {
                // Keep native JSON as strings so invalid parts produce parser
                // diagnostics rather than failing the SQL query.
                let parts = if has_parts {
                    "(SELECT json_group_array(data) FROM
                      (SELECT data FROM session_part WHERE message_id = m.id ORDER BY ordinal))"
                } else {
                    "'[]'"
                };
                format!(
                    "SELECT m.id, m.data, m.time_created, m.type, s.directory, s.parent_id, m.id, m.seq,
                        {parts}
                     FROM session_message m JOIN session_v2 s ON m.session_id = s.id
                     WHERE m.rowid = ?1"
                )
            }
        }
    }
}

/// Verify session prefixes before emitting changed suffixes.
fn opencode(
    path: &Path,
    previous: Option<&BTreeMap<String, SqliteSession>>,
    emit: &mut impl FnMut(&Record) -> Result<()>,
) -> Result<Parsed> {
    // A read transaction includes committed WAL content without copying or
    // modifying input; verification and records come from the same snapshot.
    let mut db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    db.busy_timeout(std::time::Duration::from_secs(5))?;
    let tx = db.transaction()?;
    let has_table = |name: &str| -> Result<bool> {
        Ok(tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?1)",
            [name],
            |row| row.get(0),
        )?)
    };
    let mut schemas = Vec::new();
    if has_table("session")? && has_table("message")? && has_table("part")? {
        schemas.push(OpencodeSchema::V1);
    }
    if has_table("session_v2")? && has_table("session_message")? {
        schemas.push(OpencodeSchema::V2 {
            has_parts: has_table("session_part")?,
        });
    }
    ensure!(!schemas.is_empty(), "no supported OpenCode session tables");
    let mut session_schemas = BTreeMap::new();
    for &schema in &schemas {
        let sql = match schema {
            OpencodeSchema::V1 => "SELECT id FROM session",
            OpencodeSchema::V2 { .. } => "SELECT id FROM session_v2",
        };
        let mut statement = tx.prepare(sql)?;
        let ids = statement.query_map([], |row| row.get::<_, String>(0))?;
        for id in ids {
            // OpenCode's v1 migration copies session IDs and keeps legacy rows.
            // Prefer the v2 copy so migrated transcripts are not indexed twice.
            session_schemas.insert(id?, schema);
        }
    }
    // Sort row IDs rather than full JSON payloads, which can spill to temporary
    // files. Read each payload afterward, within the same snapshot.
    let mut sessions = BTreeMap::new();
    let mut changed = Vec::new();
    for schema in schemas {
        let mut ordered = tx.prepare(schema.ordered_sql())?;
        let mut statement = tx.prepare(&schema.payload_sql())?;
        for (id, _) in session_schemas.iter().filter(|(_, s)| **s == schema) {
            let old = previous.and_then(|sessions| sessions.get(id));
            let session = {
                let mut rows = ordered.query([&id])?;
                let mut verified = VerifiedSqlitePrefix::default();
                if let Some(old) = old {
                    for prefix in &old.prefixes {
                        let mut position = verified.position.clone();
                        while position.rows < prefix.rows {
                            let Some(row) = rows.next()? else { break };
                            let mut payload = statement.query([row.get::<_, i64>(0)?])?;
                            let row = payload
                                .next()?
                                .context("OpenCode row missing from snapshot")?;
                            hash_opencode_row(&mut position.hasher, row)?;
                            position.rows += 1;
                        }
                        if position.rows != prefix.rows
                            || hex::encode(position.hasher.clone().finalize()) != prefix.hash
                        {
                            // A growing session can also contain edits, removals
                            // or parts inserted before the old end. Restart at
                            // the last verified prefix, before staging records.
                            break;
                        }
                        verified = verified.advance(prefix, position);
                    }
                }
                let SqliteResume {
                    mut rows,
                    mut verified,
                    replacement,
                } = if old.is_some_and(|old| verified.prefixes.len() != old.prefixes.len()) {
                    drop(rows);
                    verified.restart(&mut ordered, id)?
                } else {
                    SqliteResume {
                        rows,
                        verified,
                        replacement: None,
                    }
                };
                changed.extend(replacement);
                // Keep checkpoints every 64 rows and at the end, rather than
                // accumulating one checkpoint for every small append.
                const PREFIX_ROWS: usize = 64;
                if verified
                    .prefixes
                    .last()
                    .is_some_and(|p| p.rows % PREFIX_ROWS != 0)
                {
                    verified.prefixes.pop();
                }
                while let Some(row) = rows.next()? {
                    let mut payload = statement.query([row.get::<_, i64>(0)?])?;
                    let row = payload
                        .next()?
                        .context("OpenCode row missing from snapshot")?;
                    hash_opencode_row(&mut verified.position.hasher, row)?;
                    verified.position.rows += 1;
                    match schema {
                        OpencodeSchema::V1 => {
                            opencode_part(path, id, row, &mut verified.output, emit)?
                        }
                        OpencodeSchema::V2 { .. } => {
                            opencode_v2_message(path, id, row, &mut verified.output, emit)?
                        }
                    }
                    if verified.position.rows % PREFIX_ROWS == 0 {
                        verified.save_prefix();
                    }
                }
                if verified
                    .prefixes
                    .last()
                    .is_none_or(|p| p.rows != verified.position.rows)
                {
                    verified.save_prefix();
                }
                verified.capture()
            };
            sessions.insert(id.clone(), session);
        }
    }
    let replace = match previous {
        None => Replace::All,
        Some(previous) => {
            changed.extend(
                previous
                    .keys()
                    .filter(|id| !sessions.contains_key(*id))
                    .map(|id| SessionSuffix {
                        native_session: id.clone(),
                        from_sequence: 0,
                    }),
            );
            Replace::Sessions(changed)
        }
    };
    let mut diagnostics = Diagnostics::default();
    for session in sessions.values() {
        diagnostics.extend(&session.diagnostics);
    }
    Ok(Parsed {
        records: sessions.values().map(|s| s.records).sum(),
        diagnostics,
        replace,
        checkpoint: Checkpoint::Sqlite(sessions),
    })
}

fn hash_opencode_row(hasher: &mut Sha256, row: &rusqlite::Row) -> Result<()> {
    use rusqlite::types::ValueRef;
    // Include message and session metadata as well as parts: changing a role,
    // workspace or parent changes records even when part JSON is untouched.
    for column in 0..row.as_ref().column_count() {
        let value = row.get_ref(column)?;
        hasher.update([value.data_type() as u8]);
        match value {
            ValueRef::Null => {}
            ValueRef::Integer(value) => hasher.update(value.to_le_bytes()),
            ValueRef::Real(value) => hasher.update(value.to_le_bytes()),
            ValueRef::Text(value) | ValueRef::Blob(value) => {
                hasher.update((value.len() as u64).to_le_bytes());
                hasher.update(value);
            }
        }
    }
    Ok(())
}

fn opencode_part(
    path: &Path,
    native_session: &str,
    row: &rusqlite::Row,
    session: &mut SqliteOutput,
    emit: &mut impl FnMut(&Record) -> Result<()>,
) -> Result<()> {
    let id: String = row.get(0)?;
    let data = row.get_ref(1)?.as_str()?;
    let message = row.get_ref(3)?.as_str()?;
    let value: Value = match serde_json::from_str(data) {
        Ok(value) => value,
        Err(error) => {
            session
                .diagnostics
                .push(Diagnostic::new(path, format!("part {id}: {error}")));
            return Ok(());
        }
    };
    let message: Value = serde_json::from_str(message)?;
    let kind = match value["type"].as_str().unwrap_or("") {
        "text" | "reasoning" => EventType::Message,
        "tool" => EventType::ToolCall,
        "tool_result" => EventType::ToolOutput,
        "compaction" => EventType::Compaction,
        _ => EventType::Metadata,
    };
    let mut record = Record::new(Provider::Opencode, native_session, &id, kind, path);
    record.provider_message_id = Some(row.get(6)?);
    record.occurred_at = millis(row.get(2)?)?;
    record.role = message["role"].as_str().map(Into::into);
    record.workspace = row.get::<_, Option<String>>(4)?.unwrap_or_default();
    record.parent_session_id = row
        .get::<_, Option<String>>(5)?
        .map(|p| session_id(Provider::Opencode, &p));
    record.is_subagent = record.parent_session_id.is_some();
    if kind == EventType::ToolCall {
        let args = &value["state"]["input"];
        let tool = value["tool"].as_str().unwrap_or("tool");
        record.text = format!("{tool} {}", arguments(args));
        record.files = file_paths(tool, args);
    } else {
        record.text = text(&value["text"]);
        if kind == EventType::Compaction && record.text.is_empty() {
            record.text = text(&value["summary"]);
        }
        if kind == EventType::Metadata {
            // Step bookkeeping has no transcript text; other native parts
            // can carry prompts, file changes, or provider-specific content.
            if !matches!(value["type"].as_str(), Some("step-start" | "step-finish")) {
                let mut parts = Vec::new();
                strings(&value, &mut parts);
                record.text = parts.join("\n");
            }
            if value["type"] == "patch" {
                record.files = value["files"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect();
            }
        }
    }
    // A tool part can contain both the call and its output or error.
    let output: Vec<_> = ["output", "error"]
        .into_iter()
        .filter_map(|field| value["state"][field].as_str())
        .collect();
    let output = (kind == EventType::ToolCall && !output.is_empty()).then(|| output.join("\n"));
    record.structured_content = value;
    compose_message(&mut record)?;
    session.records.emit(&mut record, emit)?;
    if let Some(output) = output {
        record.event_type = crate::record::EventType::ToolOutput;
        record.event_id = digest(&["event", &record.session_id, &id, "tool_output"]);
        record.text = output;
        // The call record, one sequence number earlier, holds the native part.
        record.structured_content = Value::Null;
        session.records.emit(&mut record, emit)?;
    }
    Ok(())
}

fn opencode_v2_message(
    path: &Path,
    native_session: &str,
    row: &rusqlite::Row,
    session: &mut SqliteOutput,
    sink: &mut impl FnMut(&Record) -> Result<()>,
) -> Result<()> {
    let id: String = row.get(0)?;
    let data = row.get_ref(1)?.as_str()?;
    let kind = row.get_ref(3)?.as_str()?;
    let parts = row.get_ref(8)?.as_str()?;
    // The discriminator lives in its own column, outside the JSON payload.
    let value =
        serde_json::from_str::<serde_json::Map<String, Value>>(data).and_then(|mut value| {
            if kind == "assistant" && value.contains_key("content") {
                let parts: Vec<String> = serde_json::from_str(parts)?;
                if !parts.is_empty() {
                    let content = parts
                        .iter()
                        .map(|part| serde_json::from_str(part))
                        .collect::<serde_json::Result<Vec<Value>>>()?;
                    value.insert("content".into(), Value::Array(content));
                }
            }
            Ok(Value::Object(value))
        });
    let mut value = match value {
        Ok(value) => value,
        Err(error) => {
            session
                .diagnostics
                .push(Diagnostic::new(path, format!("message {id}: {error}")));
            return Ok(());
        }
    };
    value["type"] = kind.into();
    let mut base = Record::new(
        Provider::Opencode,
        native_session,
        &id,
        EventType::Message,
        path,
    );
    base.provider_message_id = Some(id);
    base.occurred_at = millis(row.get(2)?)?;
    base.workspace = row.get::<_, Option<String>>(4)?.unwrap_or_default();
    base.parent_session_id = row
        .get::<_, Option<String>>(5)?
        .map(|p| session_id(Provider::Opencode, &p));
    base.is_subagent = base.parent_session_id.is_some();
    base.role = match kind {
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "system" | "synthetic" | "skill" => Some(Role::System),
        _ => None,
    };
    let mut emit = |selector: &str,
                    event_type: EventType,
                    body: String,
                    structured: Value,
                    files: Vec<String>|
     -> Result<()> {
        let mut record = base.clone();
        record.event_id = digest(&[
            "event",
            &record.session_id,
            base.provider_message_id.as_deref().unwrap(),
            event_type.as_str(),
            selector,
        ]);
        record.event_type = event_type;
        record.text = body;
        record.structured_content = structured;
        compose_message(&mut record)?;
        record.files = files;
        session.records.emit(&mut record, sink)
    };
    match kind {
        "compaction" if value["status"] == "completed" => {
            emit(
                "",
                EventType::Compaction,
                text(&value["summary"]),
                value,
                Vec::new(),
            )?;
        }
        "user" | "system" | "synthetic" | "skill" => {
            emit(
                "",
                EventType::Message,
                text(&value["text"]),
                value,
                Vec::new(),
            )?;
        }
        "assistant" => {
            let visible = text(&value["content"]);
            let mut body = vec![visible.as_str()];
            strings(&value["error"], &mut body);
            let body = body
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            let tools = value["content"]
                .as_array()
                .is_some_and(|parts| parts.iter().any(|part| part["type"] == "tool"));
            if !tools {
                return emit("", EventType::Message, body, value, Vec::new());
            }
            emit("", EventType::Message, body, value.clone(), Vec::new())?;
            let content = value.as_object_mut().unwrap().remove("content");
            let Some(Value::Array(parts)) = content else {
                unreachable!("assistant tool content is an array")
            };
            for (index, part) in parts.into_iter().enumerate() {
                if part["type"] != "tool" {
                    continue;
                }
                let tool = part["name"].as_str().unwrap_or("tool");
                let args = &part["state"]["input"];
                // Tool IDs survive streaming updates and content-array insertions.
                let selector = part["id"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("index:{index}"));
                let body = format!("{tool} {}", arguments(args));
                let files = file_paths(tool, args);
                let content = text(&part["state"]["content"]);
                let mut output = vec![content.as_str()];
                strings(&part["state"]["error"], &mut output);
                let output = output
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                let output_files = (!output.is_empty()).then(|| file_paths(tool, args));
                emit(&selector, EventType::ToolCall, body, part, files)?;
                if !output.is_empty() {
                    emit(
                        &selector,
                        EventType::ToolOutput,
                        output,
                        Value::Null,
                        output_files.unwrap(),
                    )?;
                }
            }
        }
        "shell" => {
            let output = value["output"]["output"].as_str().map(str::to_owned);
            emit(
                "",
                EventType::ToolCall,
                text(&value["command"]),
                value,
                Vec::new(),
            )?;
            if let Some(output) = output {
                emit("", EventType::ToolOutput, output, Value::Null, Vec::new())?;
            }
        }
        _ => {
            let mut body = Vec::new();
            strings(&value, &mut body);
            emit("", EventType::Metadata, body.join("\n"), value, Vec::new())?;
        }
    }
    Ok(())
}

#[derive(Default)]
enum JsonlSession {
    #[default]
    AwaitingMetadata,
    Ready(String),
}

impl From<String> for JsonlSession {
    fn from(session: String) -> Self {
        if session.is_empty() {
            Self::AwaitingMetadata
        } else {
            Self::Ready(session)
        }
    }
}

impl JsonlSession {
    fn set(&mut self, session: &str) {
        *self = session.to_owned().into();
    }

    fn ready(&self) -> Option<&str> {
        match self {
            Self::AwaitingMetadata => None,
            Self::Ready(session) => Some(session),
        }
    }
}

#[derive(Default)]
struct JsonlContext {
    session: JsonlSession,
    workspace: String,
    parent: Option<String>,
}

#[derive(Default)]
struct InputOccurrences(HashMap<String, usize>);

impl InputOccurrences {
    fn allocate(&mut self, value: &Value) -> String {
        // For records without native IDs, distinguish identical content by its
        // occurrence count. Unrelated insertions and deletions do not change IDs.
        let fallback = digest(&[&value.to_string()]);
        let count = self.0.entry(fallback.clone()).or_default();
        let fallback = format!("{fallback}:{count}");
        *count += 1;
        fallback
    }
}

#[derive(Default)]
struct EmittedRecords(usize);

impl EmittedRecords {
    fn emit(
        &mut self,
        record: &mut Record,
        sink: &mut impl FnMut(&Record) -> Result<()>,
    ) -> Result<()> {
        record.sequence = self.0 as u64;
        sink(record)?;
        self.0 += 1;
        Ok(())
    }
}

#[derive(Default)]
struct JsonlState {
    offset: u64,
    line: usize,
    hasher: Sha256,
    context: JsonlContext,
    input_occurrences: InputOccurrences,
    emitted_records: EmittedRecords,
    diagnostics: Diagnostics,
}

impl JsonlState {
    fn restore(checkpoint: &JsonlCheckpoint, hasher: Sha256) -> Self {
        let JsonlCheckpoint {
            offset,
            prefix_hash: _,
            line,
            session,
            workspace,
            parent,
            fallback_counts,
            records,
            diagnostics,
        } = checkpoint.clone();
        Self {
            offset,
            line,
            hasher,
            context: JsonlContext {
                session: session.into(),
                workspace,
                parent,
            },
            input_occurrences: InputOccurrences(fallback_counts),
            emitted_records: EmittedRecords(records),
            diagnostics,
        }
    }

    fn capture(self) -> JsonlCheckpoint {
        let Self {
            offset,
            line,
            hasher,
            context,
            input_occurrences: InputOccurrences(fallback_counts),
            emitted_records: EmittedRecords(records),
            diagnostics,
        } = self;
        let JsonlContext {
            session,
            workspace,
            parent,
        } = context;
        JsonlCheckpoint {
            offset,
            prefix_hash: hex::encode(hasher.finalize()),
            line,
            session: match session {
                JsonlSession::AwaitingMetadata => String::new(),
                JsonlSession::Ready(session) => session,
            },
            workspace,
            parent,
            fallback_counts,
            records,
            diagnostics,
        }
    }
}

enum JsonlStart {
    VerifiedResume(JsonlReader),
    Replay(JsonlReader),
}

impl JsonlStart {
    fn open(path: &Path, previous: Option<&JsonlCheckpoint>) -> Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        if let Some(previous) = previous {
            let mut hasher = Sha256::new();
            // Growth does not rule out edits to earlier records. Verify every byte
            // that contributed to the saved parser state before resuming it.
            let mut prefix = input.by_ref().take(previous.offset);
            let mut block = [0; 65_536];
            loop {
                let read = prefix.read(&mut block)?;
                if read == 0 {
                    break;
                }
                hasher.update(&block[..read]);
            }
            if prefix.limit() == 0 && hex::encode(hasher.clone().finalize()) == previous.prefix_hash
            {
                return Ok(Self::VerifiedResume(JsonlReader {
                    input,
                    state: JsonlState::restore(previous, hasher),
                }));
            }
        }
        input.rewind()?;
        Ok(Self::Replay(JsonlReader {
            input,
            state: JsonlState::default(),
        }))
    }
}

struct JsonlReader {
    input: BufReader<File>,
    state: JsonlState,
}

enum JsonlLine<'a> {
    Committed { number: usize, bytes: &'a [u8] },
    DeferredTail { number: usize },
    End,
}

impl JsonlReader {
    fn next_line<'a>(&mut self, buffer: &'a mut Vec<u8>) -> Result<JsonlLine<'a>> {
        buffer.clear();
        if self.input.read_until(b'\n', buffer)? == 0 {
            return Ok(JsonlLine::End);
        }
        let number = self.state.line + 1;
        // Providers append records. Do not interpret an unfinished trailing record.
        if !buffer.ends_with(b"\n") {
            return Ok(JsonlLine::DeferredTail { number });
        }
        self.state.line = number;
        self.state.offset += buffer.len() as u64;
        self.state.hasher.update(&*buffer);
        Ok(JsonlLine::Committed {
            number,
            bytes: buffer,
        })
    }
}

fn jsonl(
    provider: Provider,
    path: &Path,
    previous: Option<&JsonlCheckpoint>,
    sink: &mut impl FnMut(&Record) -> Result<()>,
) -> Result<Parsed> {
    let (mut reader, replace) = match JsonlStart::open(path, previous)? {
        JsonlStart::VerifiedResume(reader) => (reader, Replace::Append),
        JsonlStart::Replay(reader) => (reader, Replace::All),
    };
    let mut buffer = Vec::new();
    let incomplete = loop {
        let (line, bytes) = match reader.next_line(&mut buffer)? {
            JsonlLine::Committed { number, bytes } => (number, bytes),
            JsonlLine::DeferredTail { number } => {
                break Some(Diagnostic::new(
                    path,
                    format!("line {number}: incomplete trailing record"),
                ));
            }
            JsonlLine::End => break None,
        };
        let state = &mut reader.state;
        if bytes.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let mut value: Value = match serde_json::from_slice(bytes) {
            Ok(value) => value,
            Err(error) => {
                state
                    .diagnostics
                    .push(Diagnostic::new(path, format!("line {line}: {error}")));
                continue;
            }
        };
        let kind = value["type"].as_str().unwrap_or("");
        match (provider, kind) {
            (Provider::Codex, "session_meta") => {
                state.context.session.set(
                    value["payload"]["id"]
                        .as_str()
                        .context("session_meta has no id")?,
                );
                state.context.workspace = value["payload"]["cwd"].as_str().unwrap_or("").into();
                state.context.parent = value
                    .pointer("/payload/source/subagent/thread_spawn/parent_thread_id")
                    .and_then(Value::as_str)
                    .map(|p| session_id(provider, p));
                continue;
            }
            (Provider::Pi, "session") => {
                state
                    .context
                    .session
                    .set(value["id"].as_str().context("session header has no id")?);
                state.context.workspace = value["cwd"].as_str().unwrap_or("").into();
                continue;
            }
            (Provider::CopilotCli, "session.start") => {
                state.context.session.set(
                    value["data"]["sessionId"]
                        .as_str()
                        .context("session.start has no sessionId")?,
                );
                state.context.workspace = value["data"]["context"]["cwd"]
                    .as_str()
                    .unwrap_or("")
                    .into();
                continue;
            }
            (Provider::Claude, _) => {
                if let Some(id) = value["sessionId"].as_str() {
                    if state.context.session.ready() != Some(id) {
                        state.context.workspace.clear();
                    }
                    state.context.session.set(id);
                }
                if let Some(cwd) = value["cwd"].as_str() {
                    state.context.workspace = cwd.into();
                }
            }
            _ => {}
        }
        let compaction = match provider {
            Provider::Codex => kind == "compacted",
            Provider::Pi => kind == "compaction",
            Provider::Claude => kind == "system" && value["subtype"] == "compact_boundary",
            Provider::CopilotCli => kind == "session.compaction_complete",
            Provider::Opencode => false,
        };
        let message_field = if compaction {
            None
        } else {
            Some(match provider {
                Provider::Codex if kind == "response_item" => "payload",
                Provider::Pi if kind == "message" => "message",
                Provider::Claude if matches!(kind, "user" | "assistant") => "message",
                Provider::CopilotCli
                    if matches!(
                        kind,
                        "user.message"
                            | "assistant.message"
                            | "assistant.reasoning"
                            | "tool.execution_start"
                            | "tool.execution_complete"
                    ) =>
                {
                    "data"
                }
                _ => continue,
            })
        };
        let message = message_field.map_or(&value, |field| &value[field]);
        let Some(session) = state.context.session.ready() else {
            state.diagnostics.push(Diagnostic::new(
                path,
                format!("line {line}: no session metadata"),
            ));
            continue;
        };
        let native = value["uuid"]
            .as_str()
            .or(value["id"].as_str())
            .or(message["id"].as_str());
        // Native-ID rows and timestamp rejections still consume an occurrence.
        let fallback = state.input_occurrences.allocate(&value);
        let native = native.unwrap_or(&fallback);
        let time = match value["timestamp"].as_str().map(timestamp).transpose() {
            Ok(time) => time.unwrap_or_default(),
            Err(error) => {
                state
                    .diagnostics
                    .push(Diagnostic::new(path, format!("line {line}: {error}")));
                continue;
            }
        };
        let role = message["role"]
            .as_str()
            .unwrap_or(if kind == "user.message" {
                "user"
            } else {
                "assistant"
            });
        let tool_result = role == "toolResult";
        let mut base = Record::new(provider, session, native, EventType::Message, path);
        base.parent_session_id = state.context.parent.clone();
        base.is_subagent =
            state.context.parent.is_some() || value["isSidechain"].as_bool().unwrap_or(false);
        base.workspace = state.context.workspace.clone();
        base.role = if compaction {
            (provider == Provider::Claude).then_some(Role::System)
        } else {
            Some(role.into())
        };
        base.occurred_at = time;
        base.native_kind = if provider == Provider::CopilotCli {
            match kind {
                "assistant.message" => Some(NativeEventKind::CopilotAssistantMessage),
                "assistant.reasoning" => Some(NativeEventKind::CopilotAssistantDetail),
                _ => None,
            }
        } else {
            None
        };
        let mut emit = |mut record: Record,
                        selector: &str,
                        event_type: EventType,
                        body: String,
                        structured: Value,
                        files: Vec<String>|
         -> Result<()> {
            record.event_id = digest(&[
                "event",
                &record.session_id,
                &format!("{}:{selector}", record.provider_event_id),
                event_type.as_str(),
            ]);
            record.event_type = event_type;
            if event_type != EventType::Message {
                record.native_kind = None;
            }
            record.text = body;
            record.structured_content = structured;
            compose_message(&mut record)?;
            record.files = files;
            state.emitted_records.emit(&mut record, sink)
        };
        if compaction {
            let summary = match provider {
                Provider::Codex => &value["payload"]["message"],
                Provider::Pi => &value["summary"],
                Provider::Claude => &value["content"],
                Provider::CopilotCli => &value["data"]["summaryContent"],
                Provider::Opencode => unreachable!(),
            };
            let event_type = if provider == Provider::CopilotCli
                && value["data"]["success"].as_bool() != Some(true)
            {
                EventType::Metadata
            } else {
                EventType::Compaction
            };
            let body = if event_type == EventType::Metadata {
                let mut parts = Vec::new();
                strings(&value["data"], &mut parts);
                parts.join("\n")
            } else {
                text(summary)
            };
            emit(base, "compaction", event_type, body, value, Vec::new())?;
            continue;
        }
        let copilot_reasoning = provider == Provider::CopilotCli && kind == "assistant.reasoning";
        let copilot_call = provider == Provider::CopilotCli && kind == "tool.execution_start";
        let copilot_output = provider == Provider::CopilotCli && kind == "tool.execution_complete";
        let copilot_message = provider == Provider::CopilotCli && kind == "assistant.message";
        let mut message = value
            .as_object_mut()
            .and_then(|envelope| envelope.remove(message_field.unwrap()))
            .unwrap_or(Value::Null);
        let message_kind = message["type"].as_str().unwrap_or("");
        if copilot_reasoning {
            emit(
                base,
                "message",
                EventType::Message,
                text(&message["content"]),
                message,
                Vec::new(),
            )?;
        } else if provider == Provider::Codex && message_kind == "reasoning" {
            let body = [text(&message["summary"]), text(&message["content"])]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            emit(
                base,
                "message",
                EventType::Message,
                body,
                message,
                Vec::new(),
            )?;
        } else if provider == Provider::Codex
            && matches!(message_kind, "function_call" | "custom_tool_call")
        {
            let tool = message["name"].as_str().unwrap_or("tool");
            let args = message
                .get("arguments")
                .or(message.get("input"))
                .unwrap_or(&Value::Null);
            let decoded = args
                .as_str()
                .and_then(|s| serde_json::from_str::<Value>(s).ok());
            let body = format!("{tool} {}", arguments(args));
            let files = file_paths(tool, decoded.as_ref().unwrap_or(args));
            emit(base, "call", EventType::ToolCall, body, message, files)?;
        } else if provider == Provider::Codex
            && matches!(
                message_kind,
                "function_call_output" | "custom_tool_call_output"
            )
        {
            emit(
                base,
                "output",
                EventType::ToolOutput,
                text(&message["output"]),
                message,
                Vec::new(),
            )?;
        } else if tool_result {
            emit(
                base,
                "output",
                EventType::ToolOutput,
                text(&message["content"]),
                message,
                Vec::new(),
            )?;
        } else if copilot_call {
            let tool = message["toolName"].as_str().unwrap_or("tool");
            let body = format!("{tool} {}", arguments(&message["arguments"]));
            let files = file_paths(tool, &message["arguments"]);
            emit(base, "call", EventType::ToolCall, body, message, files)?;
        } else if copilot_output {
            emit(
                base,
                "output",
                EventType::ToolOutput,
                text(&message["result"]["content"]),
                message,
                Vec::new(),
            )?;
        } else {
            let content = &message["content"];
            let body = text(content);
            // Empty visible messages still mark turn boundaries. A tool-only
            // content array is not an empty user message (Claude uses the user
            // role for tool results), so it must not interrupt pairing.
            let visible = content.is_string()
                || content.as_array().is_some_and(|parts| {
                    parts.is_empty()
                        || parts.iter().any(|part| {
                            !matches!(
                                part["type"].as_str(),
                                Some("toolCall" | "tool_use" | "tool_result")
                            )
                        })
                });
            let marked = content
                .as_array()
                .is_some_and(|parts| parts.iter().any(|part| selected_part(provider, part)));
            let visible = !body.is_empty() || visible || marked || copilot_message;
            let tools = content.as_array().is_some_and(|parts| {
                parts.iter().any(|part| {
                    matches!(
                        part["type"].as_str(),
                        Some("toolCall" | "tool_use" | "tool_result")
                    )
                })
            });
            if !tools {
                if visible {
                    emit(
                        base,
                        "message",
                        EventType::Message,
                        body,
                        message,
                        Vec::new(),
                    )?;
                }
                continue;
            }
            if visible {
                emit(
                    base.clone(),
                    "message",
                    EventType::Message,
                    body,
                    message.clone(),
                    Vec::new(),
                )?;
            }
            let content = message
                .as_object_mut()
                .and_then(|message| message.remove("content"));
            if let Some(Value::Array(parts)) = content {
                for (i, part) in parts.into_iter().enumerate() {
                    match part["type"].as_str().unwrap_or("") {
                        "toolCall" | "tool_use" => {
                            let tool = part["name"].as_str().unwrap_or("tool");
                            let args = part
                                .get("arguments")
                                .or(part.get("input"))
                                .unwrap_or(&Value::Null);
                            let body = format!("{tool} {}", arguments(args));
                            let files = file_paths(tool, args);
                            emit(
                                base.clone(),
                                &format!("call:{i}"),
                                EventType::ToolCall,
                                body,
                                part,
                                files,
                            )?;
                        }
                        "tool_result" => emit(
                            base.clone(),
                            &format!("output:{i}"),
                            EventType::ToolOutput,
                            text(&part["content"]),
                            part,
                            Vec::new(),
                        )?,
                        _ => {}
                    }
                }
            }
        }
    };
    if reader.state.context.session.ready().is_none() {
        bail!("no valid session metadata in {}", path.display());
    }
    let checkpoint = reader.state.capture();
    let mut diagnostics = checkpoint.diagnostics.clone();
    // The unfinished line has not affected counters or metadata. Read it again
    // on the next change, and replace its transient diagnostic rather than save it.
    if let Some(incomplete) = incomplete {
        diagnostics.push(incomplete);
    }
    Ok(Parsed {
        records: checkpoint.records,
        diagnostics,
        replace,
        checkpoint: Checkpoint::Jsonl(checkpoint),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_resumption_matches_full_parse_at_every_byte_boundary() {
        let root = tempfile::tempdir().unwrap();
        let cases = [
            (
                Provider::Codex,
                vec![
                    json!({
                        "type": "session_meta",
                        "payload": {
                            "id": "one", "cwd": "/first",
                            "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent"}}},
                        },
                    }),
                    crate::test_support::codex_message("user", "日本語"),
                    json!({"type":"session_meta", "payload":{"id":"two", "cwd":"/second"}}),
                    json!({"type":"response_item", "payload":{"type":"reasoning", "summary":[{"text":"reasoning"}]}}),
                ],
            ),
            (
                Provider::Pi,
                vec![
                    json!({"type":"session", "version":3, "id":"one", "cwd":"/first"}),
                    json!({"type":"message", "message":{"role":"user", "content":[{"type":"text", "text":"日本語"}]}}),
                    json!({"type":"session", "version":3, "id":"two", "cwd":"/second"}),
                    json!({
                        "type": "message", "id": "native",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "answer"},
                                {"type": "toolCall", "name": "read", "arguments": {"path": "src/main.rs"}},
                            ],
                        },
                    }),
                ],
            ),
            (
                Provider::Claude,
                vec![
                    json!({"type":"system", "sessionId":"one", "cwd":"/first"}),
                    json!({"type":"user", "isSidechain":true, "message":{"role":"user", "content":"日本語"}}),
                    json!({"type":"system", "sessionId":"two", "cwd":"/second"}),
                    json!({
                        "type": "assistant", "uuid": "native",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "answer"},
                                {"type": "tool_use", "name": "read", "input": {"path": "src/main.rs"}},
                            ],
                        },
                    }),
                ],
            ),
            (
                Provider::CopilotCli,
                vec![
                    json!({"type":"session.start", "data":{"sessionId":"one", "context":{"cwd":"/first"}}}),
                    json!({"type":"user.message", "data":{"content":"日本語"}}),
                    json!({"type":"session.start", "data":{"sessionId":"two", "context":{"cwd":"/second"}}}),
                    json!({
                        "type": "tool.execution_start", "id": "native",
                        "data": {"toolName": "read", "arguments": {"path": "src/main.rs"}},
                    }),
                ],
            ),
        ];
        for (provider, rows) in cases {
            let path = root.path().join(format!("{provider}.jsonl"));
            let mut rejected = rows[1].clone();
            rejected["timestamp"] = json!("invalid timestamp");
            // Include missing metadata, rejections, metadata-only changes, repeated
            // canonical JSON with different whitespace, and rows emitting multiple records.
            let data = format!(
                "{}\n{}\n{}\nmalformed\n{rejected}\n \r\n{}\n{}\n{}\n{}\n",
                rows[1],
                rows[0],
                rows[1],
                rows[2],
                rows[3],
                rows[1],
                rows[1].to_string().replace("\":", "\" : ")
            );
            let mut checkpoint = None;
            let mut records = Vec::new();
            for end in 1..=data.len() {
                std::fs::write(&path, &data.as_bytes()[..end]).unwrap();
                let mut delta = Vec::new();
                let incremental = parse(provider, &path, checkpoint.as_ref(), |record| {
                    delta.push(record.clone());
                    Ok(())
                });
                let mut expected = Vec::new();
                let full = parse(provider, &path, None, |record| {
                    expected.push(record.clone());
                    Ok(())
                });
                match (incremental, full) {
                    (Ok(incremental), Ok(full)) => {
                        assert_eq!(incremental.records, full.records);
                        assert_eq!(incremental.diagnostics, full.diagnostics);
                        if matches!(incremental.replace, Replace::All) {
                            records.clear();
                        }
                        records.extend(delta);
                        assert_eq!(
                            serde_json::to_value(&records).unwrap(),
                            serde_json::to_value(expected).unwrap(),
                            "{provider} at byte {end}"
                        );
                        // Round-trip through storage format.
                        checkpoint = serde_json::from_slice(
                            &serde_json::to_vec(&incremental.checkpoint).unwrap(),
                        )
                        .unwrap();
                    }
                    (Err(_), Err(_)) => {}
                    _ => panic!("resumed and full parse disagree for {provider} at byte {end}"),
                }
            }
            assert!(records.iter().any(|record| record.workspace == "/first"));
            assert!(records.iter().any(|record| record.workspace == "/second"));
        }
    }
}
