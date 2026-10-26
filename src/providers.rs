//! Provider readers that normalize native history into [`Record`]s.
//!
//! A parse returns a [`Checkpoint`] passed to the input's next parse, and
//! which previously indexed records its output replaces.
//!
//! OpenCode: one read transaction per parse. Each session is fingerprinted
//! from its row and aggregates over its parts and messages; only sessions
//! whose fingerprint changed are read. Saved row prefixes let the parser keep
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
use crate::report::Diagnostic;
use crate::texture::{BlockArray, BlockCollection, Body, Opaque, OpaqueKind, Selection, Target};
use anyhow::{Context, Result, bail, ensure};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
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
    fingerprint: String,
    records: usize,
    diagnostics: Diagnostics,
    #[serde(default)]
    prefixes: Vec<SqlitePrefix>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct SqlitePrefix {
    rows: usize,
    hash: String,
    records: usize,
    diagnostics: Diagnostics,
}

impl SqliteSession {
    fn save_prefix(&mut self, rows: usize, hasher: &Sha256) {
        self.prefixes.push(SqlitePrefix {
            rows,
            hash: hex::encode(hasher.clone().finalize()),
            records: self.records,
            diagnostics: self.diagnostics.clone(),
        });
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

/// Invalidate cached parses when a reader learns a previously ignored format.
pub fn parser_revision(provider: Provider) -> u32 {
    match provider {
        Provider::Opencode | Provider::Pi | Provider::Claude => 2,
        Provider::CopilotCli => 1,
        _ => 0,
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
        if !matches!(record.provider, Provider::Pi | Provider::Claude)
            && !(record.provider == Provider::Opencode && value["type"] == "assistant")
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

fn arguments(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
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

/// Fingerprint each OpenCode session from its own row and aggregates over its
/// parts and messages. OpenCode sets `time_updated` on every insert and
/// update; row counts and data sizes also reflect deletions. Rowids change
/// when a writer replaces a row, except when it deletes and re-inserts the
/// table's last row. A same-size `UPDATE` that keeps `time_updated` is not
/// detected; a full import reparses it.
fn opencode_fingerprints(
    db: &Connection,
    schema: OpencodeSchema,
) -> Result<BTreeMap<String, String>> {
    const AGGREGATE: &str = "count(*) || ':' || ifnull(max(time_updated), '') || ':' \
        || total(time_updated) || ':' || total(octet_length(data)) || ':' || total(rowid)";
    let sql = match schema {
        OpencodeSchema::V1 => format!(
            "SELECT s.id, s.directory, s.parent_id,
           (SELECT {AGGREGATE} FROM part WHERE session_id = s.id),
           (SELECT {AGGREGATE} FROM message WHERE session_id = s.id)
         FROM session s"
        ),
        OpencodeSchema::V2 => format!(
            "SELECT s.id, s.directory, s.parent_id,
               (SELECT {AGGREGATE} || ':' || total(seq) FROM session_message WHERE session_id = s.id),
               'v2'
             FROM session_v2 s"
        ),
    };
    let mut statement = db.prepare(&sql)?;
    let mut rows = statement.query([])?;
    let mut sessions = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let columns = (1..5)
            .map(|i| Ok(row.get::<_, Option<String>>(i)?.unwrap_or_default()))
            .collect::<Result<Vec<_>>>()?;
        let columns: Vec<_> = columns.iter().map(String::as_str).collect();
        sessions.insert(row.get(0)?, digest(&columns));
    }
    Ok(sessions)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpencodeSchema {
    V1,
    V2,
}

impl OpencodeSchema {
    fn ordered_sql(self) -> &'static str {
        match self {
            Self::V1 => {
                "SELECT p.rowid FROM part p JOIN message m ON p.message_id = m.id
                 WHERE p.session_id = ?1 ORDER BY p.time_created, p.id"
            }
            Self::V2 => "SELECT rowid FROM session_message WHERE session_id = ?1 ORDER BY seq",
        }
    }

    fn payload_sql(self) -> &'static str {
        match self {
            Self::V1 => {
                "SELECT p.id, p.data, p.time_created, m.data, s.directory, s.parent_id, m.id
                 FROM part p JOIN message m ON p.message_id = m.id
                 JOIN session s ON p.session_id = s.id WHERE p.rowid = ?1"
            }
            Self::V2 => {
                "SELECT m.id, m.data, m.time_created, m.type, s.directory, s.parent_id, m.id, m.seq
                 FROM session_message m JOIN session_v2 s ON m.session_id = s.id
                 WHERE m.rowid = ?1"
            }
        }
    }
}

/// Read changed sessions, verifying their saved prefixes before emitting a suffix.
fn opencode(
    path: &Path,
    previous: Option<&BTreeMap<String, SqliteSession>>,
    emit: &mut impl FnMut(&Record) -> Result<()>,
) -> Result<Parsed> {
    // A read transaction includes committed WAL content without copying or
    // modifying input; fingerprints and records come from the same snapshot.
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
        schemas.push(OpencodeSchema::V2);
    }
    ensure!(!schemas.is_empty(), "no supported OpenCode session tables");
    let mut fingerprints = BTreeMap::new();
    for &schema in &schemas {
        for (id, fingerprint) in opencode_fingerprints(&tx, schema)? {
            // OpenCode's v1 migration copies session IDs and keeps legacy rows.
            // Prefer the v2 copy so migrated transcripts are not indexed twice.
            fingerprints.insert(id, (schema, fingerprint));
        }
    }
    // Sort row IDs rather than full JSON payloads, which can spill to temporary
    // files. Read each payload afterward, within the same snapshot.
    let mut sessions = BTreeMap::new();
    let mut changed = Vec::new();
    for schema in schemas {
        let mut ordered = tx.prepare(schema.ordered_sql())?;
        let mut statement = tx.prepare(schema.payload_sql())?;
        for (id, (_, fingerprint)) in fingerprints.iter().filter(|(_, (s, _))| *s == schema) {
            let old = previous.and_then(|sessions| sessions.get(id));
            let session = match old {
                Some(old) if &old.fingerprint == fingerprint => old.clone(),
                _ => {
                    let mut rows = ordered.query([&id])?;
                    let mut hasher = Sha256::new();
                    let mut count = 0;
                    let mut session = SqliteSession {
                        fingerprint: fingerprint.clone(),
                        records: 0,
                        diagnostics: Diagnostics::default(),
                        prefixes: Vec::new(),
                    };
                    if let Some(old) = old {
                        for prefix in &old.prefixes {
                            let verified_hasher = hasher.clone();
                            while count < prefix.rows {
                                let Some(row) = rows.next()? else { break };
                                let mut payload = statement.query([row.get::<_, i64>(0)?])?;
                                let row = payload
                                    .next()?
                                    .context("OpenCode row missing from snapshot")?;
                                hash_opencode_row(&mut hasher, row)?;
                                count += 1;
                            }
                            if count != prefix.rows
                                || hex::encode(hasher.clone().finalize()) != prefix.hash
                            {
                                // A growing session can also contain edits, removals
                                // or parts inserted before the old end. Restart at
                                // the last verified prefix, before staging records.
                                hasher = verified_hasher;
                                count = session.prefixes.last().map_or(0, |p| p.rows);
                                drop(rows);
                                rows = ordered.query([&id])?;
                                for _ in 0..count {
                                    rows.next()?
                                        .context("OpenCode prefix missing from snapshot")?;
                                }
                                break;
                            }
                            session.records = prefix.records;
                            session.diagnostics = prefix.diagnostics.clone();
                            session.prefixes.push(prefix.clone());
                        }
                        if old.prefixes.is_empty() || session.prefixes.len() != old.prefixes.len() {
                            changed.push(SessionSuffix {
                                native_session: id.clone(),
                                from_sequence: session.records as u64,
                            });
                        }
                    }
                    // Keep checkpoints every 64 rows and at the end, rather than
                    // accumulating one checkpoint for every small append.
                    const PREFIX_ROWS: usize = 64;
                    if session
                        .prefixes
                        .last()
                        .is_some_and(|p| p.rows % PREFIX_ROWS != 0)
                    {
                        session.prefixes.pop();
                    }
                    while let Some(row) = rows.next()? {
                        let mut payload = statement.query([row.get::<_, i64>(0)?])?;
                        let row = payload
                            .next()?
                            .context("OpenCode row missing from snapshot")?;
                        hash_opencode_row(&mut hasher, row)?;
                        count += 1;
                        match schema {
                            OpencodeSchema::V1 => opencode_part(path, id, row, &mut session, emit)?,
                            OpencodeSchema::V2 => {
                                opencode_v2_message(path, id, row, &mut session, emit)?
                            }
                        }
                        if count % PREFIX_ROWS == 0 {
                            session.save_prefix(count, &hasher);
                        }
                    }
                    if session.prefixes.last().is_none_or(|p| p.rows != count) {
                        session.save_prefix(count, &hasher);
                    }
                    session
                }
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
    session: &mut SqliteSession,
    emit: &mut impl FnMut(&Record) -> Result<()>,
) -> Result<()> {
    let id: String = row.get(0)?;
    let data: String = row.get(1)?;
    let message: String = row.get(3)?;
    let value: Value = match serde_json::from_str(&data) {
        Ok(value) => value,
        Err(error) => {
            session
                .diagnostics
                .push(Diagnostic::new(path, format!("part {id}: {error}")));
            return Ok(());
        }
    };
    let message: Value = serde_json::from_str(&message)?;
    let kind = match value["type"].as_str().unwrap_or("") {
        "text" | "reasoning" => EventType::Message,
        "tool" => EventType::ToolCall,
        "tool_result" => EventType::ToolOutput,
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
    record.sequence = session.records as u64;
    if kind == EventType::ToolCall {
        let args = &value["state"]["input"];
        let tool = value["tool"].as_str().unwrap_or("tool");
        record.text = format!("{tool} {}", arguments(args));
        record.files = file_paths(tool, args);
    } else {
        record.text = text(&value["text"]);
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
    emit(&record)?;
    session.records += 1;
    if let Some(output) = output {
        record.event_type = crate::record::EventType::ToolOutput;
        record.event_id = digest(&["event", &record.session_id, &id, "tool_output"]);
        record.text = output;
        // The call record, one sequence number earlier, holds the native part.
        record.structured_content = Value::Null;
        record.sequence += 1;
        emit(&record)?;
        session.records += 1;
    }
    Ok(())
}

fn opencode_v2_message(
    path: &Path,
    native_session: &str,
    row: &rusqlite::Row,
    session: &mut SqliteSession,
    sink: &mut impl FnMut(&Record) -> Result<()>,
) -> Result<()> {
    let id: String = row.get(0)?;
    let data: String = row.get(1)?;
    let kind: String = row.get(3)?;
    // The discriminator lives in its own column, outside the JSON payload.
    let mut value = match serde_json::from_str::<serde_json::Map<String, Value>>(&data) {
        Ok(value) => Value::Object(value),
        Err(error) => {
            session
                .diagnostics
                .push(Diagnostic::new(path, format!("message {id}: {error}")));
            return Ok(());
        }
    };
    value["type"] = kind.clone().into();
    let mut base = Record::new(
        Provider::Opencode,
        native_session,
        &id,
        EventType::Message,
        path,
    );
    base.provider_message_id = Some(id.clone());
    base.occurred_at = millis(row.get(2)?)?;
    base.workspace = row.get::<_, Option<String>>(4)?.unwrap_or_default();
    base.parent_session_id = row
        .get::<_, Option<String>>(5)?
        .map(|p| session_id(Provider::Opencode, &p));
    base.is_subagent = base.parent_session_id.is_some();
    base.role = match kind.as_str() {
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
            &id,
            event_type.as_str(),
            selector,
        ]);
        record.event_type = event_type;
        record.text = body;
        record.structured_content = structured;
        compose_message(&mut record)?;
        record.files = files;
        record.sequence = session.records as u64;
        sink(&record)?;
        session.records += 1;
        Ok(())
    };
    match kind.as_str() {
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
            emit(
                "",
                EventType::Message,
                body.into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n"),
                value.clone(),
                Vec::new(),
            )?;
            for (index, part) in value["content"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
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
                emit(
                    &selector,
                    EventType::ToolCall,
                    format!("{tool} {}", arguments(args)),
                    part.clone(),
                    file_paths(tool, args),
                )?;
                let content = text(&part["state"]["content"]);
                let mut output = vec![content.as_str()];
                strings(&part["state"]["error"], &mut output);
                let output = output
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                if !output.is_empty() {
                    emit(
                        &selector,
                        EventType::ToolOutput,
                        output,
                        Value::Null,
                        file_paths(tool, args),
                    )?;
                }
            }
        }
        "shell" => {
            emit(
                "",
                EventType::ToolCall,
                text(&value["command"]),
                value.clone(),
                Vec::new(),
            )?;
            if let Some(output) = value["output"]["output"].as_str() {
                emit(
                    "",
                    EventType::ToolOutput,
                    output.into(),
                    Value::Null,
                    Vec::new(),
                )?;
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

fn jsonl(
    provider: Provider,
    path: &Path,
    previous: Option<&JsonlCheckpoint>,
    sink: &mut impl FnMut(&Record) -> Result<()>,
) -> Result<Parsed> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut append = false;
    if let Some(previous) = previous {
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
        append =
            prefix.limit() == 0 && hex::encode(hasher.clone().finalize()) == previous.prefix_hash;
    }
    let mut state = if append {
        previous.unwrap().clone()
    } else {
        input.rewind()?;
        hasher = Sha256::new();
        JsonlCheckpoint::default()
    };
    let mut buffer = Vec::new();
    let mut parsed = Parsed {
        records: state.records,
        diagnostics: std::mem::take(&mut state.diagnostics),
        replace: if append {
            Replace::Append
        } else {
            Replace::All
        },
        checkpoint: Checkpoint::Jsonl(JsonlCheckpoint::default()),
    };
    let mut incomplete = None;
    loop {
        buffer.clear();
        if input.read_until(b'\n', &mut buffer)? == 0 {
            break;
        }
        let line = state.line + 1;
        // Providers append records. Do not interpret an unfinished trailing record.
        if !buffer.ends_with(b"\n") {
            incomplete = Some(Diagnostic::new(
                path,
                format!("line {line}: incomplete trailing record"),
            ));
            break;
        }
        state.line = line;
        state.offset += buffer.len() as u64;
        hasher.update(&buffer);
        if buffer.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let value: Value = match serde_json::from_slice(&buffer) {
            Ok(value) => value,
            Err(error) => {
                parsed
                    .diagnostics
                    .push(Diagnostic::new(path, format!("line {line}: {error}")));
                continue;
            }
        };
        let kind = value["type"].as_str().unwrap_or("");
        match (provider, kind) {
            (Provider::Codex, "session_meta") => {
                state.session = value["payload"]["id"]
                    .as_str()
                    .context("session_meta has no id")?
                    .into();
                state.workspace = value["payload"]["cwd"].as_str().unwrap_or("").into();
                state.parent = value
                    .pointer("/payload/source/subagent/thread_spawn/parent_thread_id")
                    .and_then(Value::as_str)
                    .map(|p| session_id(provider, p));
                continue;
            }
            (Provider::Pi, "session") => {
                state.session = value["id"]
                    .as_str()
                    .context("session header has no id")?
                    .into();
                state.workspace = value["cwd"].as_str().unwrap_or("").into();
                continue;
            }
            (Provider::CopilotCli, "session.start") => {
                state.session = value["data"]["sessionId"]
                    .as_str()
                    .context("session.start has no sessionId")?
                    .into();
                state.workspace = value["data"]["context"]["cwd"]
                    .as_str()
                    .unwrap_or("")
                    .into();
                continue;
            }
            (Provider::Claude, _) => {
                if let Some(id) = value["sessionId"].as_str() {
                    state.session = id.into();
                }
                if let Some(cwd) = value["cwd"].as_str() {
                    state.workspace = cwd.into();
                }
            }
            _ => {}
        }
        let message = match provider {
            Provider::Codex if kind == "response_item" => &value["payload"],
            Provider::Pi if kind == "message" => &value["message"],
            Provider::Claude if matches!(kind, "user" | "assistant") => &value["message"],
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
                &value["data"]
            }
            _ => continue,
        };
        if state.session.is_empty() {
            parsed.diagnostics.push(Diagnostic::new(
                path,
                format!("line {line}: no session metadata"),
            ));
            continue;
        }
        let native = value["uuid"]
            .as_str()
            .or(value["id"].as_str())
            .or(message["id"].as_str());
        // For records without native IDs, distinguish identical content by its
        // occurrence count. Unrelated insertions and deletions do not change IDs.
        let fallback = digest(&[&value.to_string()]);
        let count = state.fallback_counts.entry(fallback.clone()).or_default();
        let fallback = format!("{fallback}:{count}");
        *count += 1;
        let native = native.unwrap_or(&fallback);
        let time = match value["timestamp"].as_str().map(timestamp).transpose() {
            Ok(time) => time.unwrap_or_default(),
            Err(error) => {
                parsed
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
        let mut emit = |selector: &str,
                        event_type: EventType,
                        body: String,
                        structured: Value,
                        files: Vec<String>|
         -> Result<()> {
            let mut record = Record::new(
                provider,
                &state.session,
                &format!("{native}:{selector}"),
                event_type,
                path,
            );
            record.provider_event_id = native.into();
            record.parent_session_id = state.parent.clone();
            record.is_subagent =
                state.parent.is_some() || value["isSidechain"].as_bool().unwrap_or(false);
            record.workspace = state.workspace.clone();
            record.role = Some(role.into());
            record.occurred_at = time.clone();
            record.text = body;
            record.structured_content = structured;
            if provider == Provider::CopilotCli && event_type == EventType::Message {
                record.native_kind = match kind {
                    "assistant.message" => Some(NativeEventKind::CopilotAssistantMessage),
                    "assistant.reasoning" => Some(NativeEventKind::CopilotAssistantDetail),
                    _ => None,
                };
            }
            compose_message(&mut record)?;
            record.files = files;
            record.sequence = parsed.records as u64;
            sink(&record)?;
            parsed.records += 1;
            Ok(())
        };
        let message_kind = message["type"].as_str().unwrap_or("");
        if provider == Provider::CopilotCli && kind == "assistant.reasoning" {
            emit(
                "message",
                EventType::Message,
                text(&message["content"]),
                message.clone(),
                Vec::new(),
            )?;
        } else if provider == Provider::Codex && message_kind == "reasoning" {
            let body = [text(&message["summary"]), text(&message["content"])]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            emit(
                "message",
                EventType::Message,
                body,
                message.clone(),
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
            emit(
                "call",
                EventType::ToolCall,
                format!("{tool} {}", arguments(args)),
                message.clone(),
                file_paths(tool, decoded.as_ref().unwrap_or(args)),
            )?;
        } else if provider == Provider::Codex
            && matches!(
                message_kind,
                "function_call_output" | "custom_tool_call_output"
            )
        {
            emit(
                "output",
                EventType::ToolOutput,
                text(&message["output"]),
                message.clone(),
                Vec::new(),
            )?;
        } else if role == "toolResult" {
            emit(
                "output",
                EventType::ToolOutput,
                text(&message["content"]),
                message.clone(),
                Vec::new(),
            )?;
        } else if provider == Provider::CopilotCli && kind == "tool.execution_start" {
            let tool = message["toolName"].as_str().unwrap_or("tool");
            emit(
                "call",
                EventType::ToolCall,
                format!("{tool} {}", arguments(&message["arguments"])),
                message.clone(),
                file_paths(tool, &message["arguments"]),
            )?;
        } else if provider == Provider::CopilotCli && kind == "tool.execution_complete" {
            emit(
                "output",
                EventType::ToolOutput,
                text(&message["result"]["content"]),
                message.clone(),
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
            if !body.is_empty()
                || visible
                || marked
                || (provider == Provider::CopilotCli && kind == "assistant.message")
            {
                emit(
                    "message",
                    EventType::Message,
                    body,
                    message.clone(),
                    Vec::new(),
                )?;
            }
            if let Some(parts) = content.as_array() {
                for (i, part) in parts.iter().enumerate() {
                    match part["type"].as_str().unwrap_or("") {
                        "toolCall" | "tool_use" => {
                            let tool = part["name"].as_str().unwrap_or("tool");
                            let args = part
                                .get("arguments")
                                .or(part.get("input"))
                                .unwrap_or(&Value::Null);
                            emit(
                                &format!("call:{i}"),
                                EventType::ToolCall,
                                format!("{tool} {}", arguments(args)),
                                part.clone(),
                                file_paths(tool, args),
                            )?;
                        }
                        "tool_result" => emit(
                            &format!("output:{i}"),
                            EventType::ToolOutput,
                            text(&part["content"]),
                            part.clone(),
                            Vec::new(),
                        )?,
                        _ => {}
                    }
                }
            }
        }
    }
    if state.session.is_empty() {
        bail!("no valid session metadata in {}", path.display());
    }
    state.prefix_hash = hex::encode(hasher.finalize());
    state.records = parsed.records;
    state.diagnostics = parsed.diagnostics.clone();
    // The unfinished line has not affected counters or metadata. Read it again
    // on the next change, and replace its transient diagnostic rather than save it.
    if let Some(incomplete) = incomplete {
        parsed.diagnostics.push(incomplete);
    }
    parsed.checkpoint = Checkpoint::Jsonl(state);
    Ok(parsed)
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
