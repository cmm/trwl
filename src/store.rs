//! Tantivy index of complete records, and its import.
//!
//! Each commit's payload is the manifest: per-input source stamps and a semantic
//! revision that changes with conversation inputs and result metadata. Each input
//! also has a checkpoint document in the same commit, so a reader pinned to a
//! commit sees records, stamps and parser state of one import. Checkpoint
//! documents are excluded from record scans, counts and semantic input. An
//! import loads an input's checkpoint only when its stamp changed and rewrites
//! it only when it changed. Manifests without checkpoint documents cause one
//! full reparse; a full import reparses every input without its checkpoint.
//!
//! A file lock serializes writers. Changed inputs are parsed into one
//! temporary spool file; a source that fails to parse keeps its previous
//! records. Inputs missing from a listed root are removed; an inaccessible
//! root keeps its history. SQLite inputs record the stamp from before their
//! read transaction, so writes during the read cause another scan. JSONL
//! inputs must remain unchanged while they are parsed.
//!
//! Filters are index queries. A collector keeps bounded event winners or one
//! winner per session using fast fields; only winners load stored snippets.
//! Display text is stored once. Strings repeated in native content refer to
//! byte ranges of that text and are restored for full retrieval.

use crate::{
    cli::Filters,
    config::{Config, Source, is_locked, lock},
    providers,
    record::{ConversationMessage, Provider, Record, digest},
    report::{Diagnostic, ImportReport, ImportSource, LexicalStatus, Outcome, State},
    workspace,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
    fs::{self, File},
    hash::Hash,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};
use tantivy::{
    DocAddress, DocSet, Index, ReloadPolicy, Searcher, SegmentReader, TERMINATED, TantivyDocument,
    Term,
    collector::{Collector, Count, DocSetCollector, SegmentCollector, TopDocs},
    doc,
    query::{
        AllQuery, BooleanQuery, ConstScoreQuery, Occur, PhraseQuery, Query, RangeQuery, RegexQuery,
        TermQuery, TermSetQuery,
    },
    schema::{FAST, Field, IndexRecordOption, STORED, STRING, Schema, TEXT, Value as _},
    snippet::SnippetGenerator,
};

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Stamp(Vec<Option<FileStamp>>);

const WRITER_THREADS: usize = 2;
const WRITER_MEMORY_BYTES: usize = 64_000_000;

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FileStamp {
    device: u64,
    inode: u64,
    length: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

fn stamp(path: &Path, sqlite: bool) -> Result<Stamp> {
    let mut paths = vec![path.to_path_buf()];
    if sqlite {
        paths.push(PathBuf::from(format!("{}-wal", path.display())));
    }
    Ok(Stamp(
        paths
            .iter()
            .map(|p| match fs::metadata(p) {
                Ok(m) => Ok(Some(FileStamp {
                    device: m.dev(),
                    inode: m.ino(),
                    length: m.len(),
                    modified: (m.mtime(), m.mtime_nsec()),
                    changed: (m.ctime(), m.ctime_nsec()),
                })),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && p != path => Ok(None),
                Err(e) => Err(e).with_context(|| format!("reading {}", p.display())),
            })
            .collect::<Result<_>>()?,
    ))
}

#[derive(Clone, Serialize, Deserialize)]
struct InputState {
    provider: crate::record::Provider,
    path: PathBuf,
    stamp: Stamp,
    /// The index holds this input's checkpoint document. Manifests from
    /// earlier versions kept checkpoints inline; their inputs are reparsed.
    #[serde(default)]
    checkpointed: bool,
    #[serde(default)]
    parser_revision: u32,
}

impl InputState {
    fn unchanged(&self, stamp: &Stamp) -> bool {
        self.checkpointed
            && self.parser_revision == providers::parser_revision(self.provider)
            && self.stamp == *stamp
    }
}

struct Change {
    provider: Provider,
    replace: providers::Replace,
    range: Option<(u64, u64)>,
    /// A checkpoint to store, replacing the previous one.
    checkpoint: Option<providers::Checkpoint>,
}

/// The `id` of each input's checkpoint document. Event references are
/// hexadecimal, so they cannot match it.
const CHECKPOINT_ID: &str = "input-checkpoint";

const RECORD_VERSION: u32 = 5;
#[derive(Serialize, Deserialize)]
struct Manifest {
    #[serde(default)]
    record_version: u32,
    inputs: BTreeMap<String, InputState>,
    #[serde(default)]
    workspace_roots: workspace::Roots,
    #[serde(default)]
    semantic_version: Option<String>,
}
impl Default for Manifest {
    fn default() -> Self {
        Self {
            record_version: RECORD_VERSION,
            inputs: BTreeMap::new(),
            workspace_roots: workspace::Roots::default(),
            semantic_version: Some(digest(&["empty conversation"])),
        }
    }
}

fn decode_manifest(payload: Option<&str>) -> Result<Manifest> {
    let manifest: Manifest = payload
        .map(serde_json::from_str)
        .transpose()
        .context("invalid source manifest")?
        .unwrap_or_default();
    ensure!(
        manifest.record_version == RECORD_VERSION,
        "incompatible history records; move the index directory aside, then re-import your sources"
    );
    Ok(manifest)
}

/// Each indexed input has a document keyed by its `source` that stores its
/// parser checkpoint in `record`, with `id` set to [`CHECKPOINT_ID`]. It has
/// no text or session and is not a conversation message, so only scans over
/// every document need to exclude it.
#[derive(Clone, Copy)]
struct Fields {
    source: Field,
    id: Field,
    session: Field,
    text: Field,
    record: Field,
    summary: Field,
    message: Field,
    native_session: Field,
    provider: Field,
    workspace_root: Field,
    timestamp: Field,
    event_type: Field,
    subagent: Field,
    file: Field,
    sequence: Field,
}

fn schema() -> (Schema, Fields) {
    let mut schema = Schema::builder();
    let fields = Fields {
        source: schema.add_text_field("source", STRING),
        id: schema.add_text_field("id", STRING | FAST),
        session: schema.add_text_field("session", STRING | FAST),
        text: schema.add_text_field("text", TEXT | STORED),
        record: schema.add_bytes_field("record", STORED),
        summary: schema.add_bytes_field("summary", STORED),
        message: schema.add_bool_field("conversation_message", FAST),
        native_session: schema.add_text_field("provider_session_id", STRING | FAST),
        provider: schema.add_text_field("provider", STRING | FAST),
        workspace_root: schema.add_text_field("workspace_root", STRING | FAST),
        timestamp: schema.add_text_field("timestamp", STRING | FAST),
        event_type: schema.add_text_field("event_type", STRING | FAST),
        subagent: schema.add_bool_field("subagent", FAST | tantivy::schema::INDEXED),
        file: schema.add_text_field("file_suffix", STRING),
        sequence: schema.add_u64_field("sequence", FAST | tantivy::schema::INDEXED),
    };
    (schema.build(), fields)
}

pub struct ReadIndex {
    index: Index,
    searcher: Searcher,
    fields: Fields,
    semantic_version: String,
    #[cfg(test)]
    pub(crate) record_decodes: std::rc::Rc<std::cell::Cell<usize>>,
    #[cfg(test)]
    summary_decodes: std::cell::Cell<usize>,
}

#[cfg(test)]
thread_local! {
    // Count reads across indexes opened by the operation under test.
    static RECORD_DECODES: std::rc::Rc<std::cell::Cell<usize>> = Default::default();
}

#[derive(Clone, Copy)]
pub enum LexicalMode {
    Events(usize),
    Sessions,
    #[cfg(test)]
    All,
}

pub struct LexicalResults {
    pub rows: Vec<(f32, Record)>,
    addresses: HashMap<String, DocAddress>,
    generators: Vec<SnippetGenerator>,
}

impl LexicalResults {
    pub fn snippet(&self, index: &ReadIndex, id: &str) -> Result<String> {
        let address = self.addresses.get(id).context("lexical result missing")?;
        let doc: TantivyDocument = index.searcher.doc(*address)?;
        let text = doc
            .get_first(index.fields.text)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        Ok(self
            .generators
            .iter()
            .map(|g| g.snippet(text))
            .filter(|s| !s.highlighted().is_empty())
            .max_by_key(|s| s.highlighted().len())
            .map(|s| s.fragment().to_owned())
            .unwrap_or_else(|| {
                text.chars()
                    .take(crate::record::SEARCH_SNIPPET_CHARS)
                    .collect()
            }))
    }
}

#[derive(Serialize, Deserialize)]
struct StoredRecord {
    record: Record,
    references: Vec<(String, usize, usize)>,
}

impl StoredRecord {
    fn new(mut record: Record) -> Self {
        fn dedup(
            value: &mut Value,
            path: &str,
            text: &str,
            refs: &mut Vec<(String, usize, usize)>,
        ) {
            match value {
                Value::String(s) if s.len() >= 32 => {
                    if let Some(start) = text.find(s.as_str()) {
                        refs.push((path.into(), start, start + s.len()));
                        *value = Value::Null;
                    }
                }
                Value::Array(items) => {
                    for (i, item) in items.iter_mut().enumerate() {
                        dedup(item, &format!("{path}/{i}"), text, refs);
                    }
                }
                Value::Object(map) => {
                    for (key, item) in map {
                        let key = key.replace('~', "~0").replace('/', "~1");
                        dedup(item, &format!("{path}/{key}"), text, refs);
                    }
                }
                _ => {}
            }
        }
        let mut references = Vec::new();
        dedup(
            &mut record.structured_content,
            "",
            &record.text,
            &mut references,
        );
        record.text.clear();
        Self { record, references }
    }

    fn restore(mut self, text: &str) -> Result<Record> {
        for (path, start, end) in self.references {
            let value = self
                .record
                .structured_content
                .pointer_mut(&path)
                .context("stored content reference missing")?;
            *value = text
                .get(start..end)
                .context("invalid stored text range")?
                .into();
        }
        self.record.text = text.into();
        Ok(self.record)
    }
}

#[derive(Clone, Debug)]
struct Candidate<T = String> {
    score: f32,
    priority: Priority<T>,
    session: T,
    address: DocAddress,
}

#[derive(Clone, Debug)]
enum Priority<T = String> {
    Relevance(f32, T, T),
    Chronological(T, u64, T),
}

impl Priority {
    fn id(&self) -> &str {
        match self {
            Self::Relevance(_, _, id) | Self::Chronological(_, _, id) => id,
        }
    }
}

impl<T: Ord> Priority<T> {
    fn compare(&self, other: &Self) -> Ordering {
        // Heaps keep the greatest priority. Public ordering puts the best first.
        match (self, other) {
            (Self::Relevance(sa, ta, ia), Self::Relevance(sb, tb, ib)) => sa
                .total_cmp(sb)
                .then_with(|| ta.cmp(tb))
                .then_with(|| ib.cmp(ia)),
            (Self::Chronological(ta, sa, ia), Self::Chronological(tb, sb, ib)) => {
                (tb, sb, ib).cmp(&(ta, sa, ia))
            }
            (Self::Relevance(..), Self::Chronological(..)) => Ordering::Less,
            (Self::Chronological(..), Self::Relevance(..)) => Ordering::Greater,
        }
    }
}

impl<T: Ord> PartialEq for Candidate<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<T: Ord> Eq for Candidate<T> {}
impl<T: Ord> PartialOrd for Candidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Ord> Ord for Candidate<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.compare(&other.priority)
    }
}

struct Winners {
    mode: LexicalMode,
    chronological: bool,
}
struct SegmentWinners {
    segment: u32,
    time: tantivy::columnar::StrColumn,
    id: tantivy::columnar::StrColumn,
    session: tantivy::columnar::StrColumn,
    sequence: tantivy::columnar::Column<u64>,
    candidates: Candidates<u64>,
    error: Option<anyhow::Error>,
    chronological: bool,
}

fn string(column: &tantivy::columnar::StrColumn, doc: u32) -> Result<String> {
    let ordinal = column
        .term_ords(doc)
        .next()
        .context("fast-field value missing")?;
    ordinal_string(column, ordinal)
}

fn ordinal_string(column: &tantivy::columnar::StrColumn, ordinal: u64) -> Result<String> {
    let mut text = String::new();
    ensure!(
        column.ord_to_str(ordinal, &mut text)?,
        "fast-field ordinal missing"
    );
    Ok(text)
}

enum Candidates<T = String> {
    Events(usize, BinaryHeap<Reverse<Candidate<T>>>),
    Sessions(HashMap<T, Candidate<T>>),
    #[cfg(test)]
    All(Vec<Candidate<T>>),
}

impl<T: Ord + Clone + Hash> Candidates<T> {
    fn new(mode: LexicalMode) -> Self {
        match mode {
            LexicalMode::Events(limit) => Self::Events(limit, BinaryHeap::new()),
            LexicalMode::Sessions => Self::Sessions(HashMap::new()),
            #[cfg(test)]
            LexicalMode::All => Self::All(Vec::new()),
        }
    }

    fn push(&mut self, row: Candidate<T>) {
        match self {
            Self::Events(limit, rows) => {
                if rows.len() < *limit {
                    rows.push(Reverse(row));
                } else if rows.peek().is_some_and(|worst| row > worst.0) {
                    rows.pop();
                    rows.push(Reverse(row));
                }
            }
            Self::Sessions(rows) => {
                let entry = rows
                    .entry(row.session.clone())
                    .or_insert_with(|| row.clone());
                if row > *entry {
                    *entry = row;
                }
            }
            #[cfg(test)]
            Self::All(rows) => rows.push(row),
        }
    }

    fn finish(self) -> Vec<Candidate<T>> {
        match self {
            Self::Events(_, rows) => rows.into_iter().map(|row| row.0).collect(),
            Self::Sessions(rows) => rows.into_values().collect(),
            #[cfg(test)]
            Self::All(rows) => rows,
        }
    }
}

impl Collector for Winners {
    type Fruit = Vec<Candidate>;
    type Child = SegmentWinners;
    fn for_segment(&self, segment: u32, reader: &SegmentReader) -> tantivy::Result<Self::Child> {
        let field = |name| {
            reader.fast_fields().str(name)?.ok_or_else(|| {
                tantivy::TantivyError::SchemaError(format!("missing {name} fast field"))
            })
        };
        Ok(SegmentWinners {
            segment,
            time: field("timestamp")?,
            id: field("id")?,
            session: field("session")?,
            sequence: reader.fast_fields().u64("sequence")?,
            candidates: Candidates::new(self.mode),
            error: None,
            chronological: self.chronological,
        })
    }
    fn requires_scoring(&self) -> bool {
        true
    }
    fn merge_fruits(&self, fruits: Vec<Result<Vec<Candidate>>>) -> tantivy::Result<Vec<Candidate>> {
        let mut candidates = Candidates::new(self.mode);
        for rows in fruits {
            for row in rows.map_err(|e| tantivy::TantivyError::InvalidArgument(e.to_string()))? {
                candidates.push(row);
            }
        }
        Ok(candidates.finish())
    }
}

impl SegmentCollector for SegmentWinners {
    type Fruit = Result<Vec<Candidate>>;
    fn collect(&mut self, doc: u32, score: f32) {
        if self.error.is_some() {
            return;
        }
        let row = self.candidate(doc, score);
        match row {
            Err(error) => self.error = Some(error),
            Ok(row) => self.candidates.push(row),
        }
    }
    fn harvest(self) -> Result<Vec<Candidate>> {
        if let Some(error) = self.error {
            return Err(error);
        }
        // Dictionary ordinals preserve string order within this segment. Decode
        // only its winners before comparing values from different dictionaries.
        self.candidates
            .finish()
            .into_iter()
            .map(|row| {
                Ok(Candidate {
                    score: row.score,
                    address: row.address,
                    session: ordinal_string(&self.session, row.session)?,
                    priority: match row.priority {
                        Priority::Relevance(score, time, id) => Priority::Relevance(
                            score,
                            ordinal_string(&self.time, time)?,
                            ordinal_string(&self.id, id)?,
                        ),
                        Priority::Chronological(time, sequence, id) => Priority::Chronological(
                            ordinal_string(&self.time, time)?,
                            sequence,
                            ordinal_string(&self.id, id)?,
                        ),
                    },
                })
            })
            .collect()
    }
}

impl SegmentWinners {
    fn candidate(&self, doc: u32, score: f32) -> Result<Candidate<u64>> {
        let ordinal = |column: &tantivy::columnar::StrColumn| {
            column
                .term_ords(doc)
                .next()
                .context("fast-field value missing")
        };
        let time = ordinal(&self.time)?;
        let id = ordinal(&self.id)?;
        Ok(Candidate {
            score,
            priority: if self.chronological {
                Priority::Chronological(time, self.sequence.first(doc).unwrap_or(0), id)
            } else {
                Priority::Relevance(score, time, id)
            },
            session: ordinal(&self.session)?,
            address: DocAddress::new(self.segment, doc),
        })
    }
}

impl ReadIndex {
    /// Identify this reader's conversation inputs and result metadata.
    pub fn version(&self) -> String {
        self.semantic_version.clone()
    }

    pub fn open(root: &Path) -> Result<Option<Self>> {
        let path = root.join("index");
        if !path.join("meta.json").exists() {
            return Ok(None);
        }
        let index = Index::open_in_dir(&path)?;
        let (expected, fields) = schema();
        ensure!(
            index.schema() == expected,
            "incompatible history index at {}; move that directory aside, then re-import your sources",
            path.display()
        );
        let reader: tantivy::IndexReader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        loop {
            let meta = index.load_metas()?;
            let manifest = decode_manifest(meta.payload.as_deref())?;
            let version = segment_version(
                meta.segments
                    .iter()
                    .map(|s| (s.id().uuid_string(), s.num_deleted_docs())),
            );
            let searcher = reader.searcher();
            // The manifest and searcher must describe the same published records.
            // Retry a concurrent commit without holding up the importer.
            if version
                != segment_version(
                    searcher
                        .segment_readers()
                        .iter()
                        .map(|s| (s.segment_id().uuid_string(), s.max_doc() - s.num_docs())),
                )
            {
                reader.reload()?;
                continue;
            }
            return Ok(Some(Self {
                index,
                searcher,
                fields,
                semantic_version: manifest.semantic_version.unwrap_or(version),
                #[cfg(test)]
                record_decodes: RECORD_DECODES.with(Clone::clone),
                #[cfg(test)]
                summary_decodes: std::cell::Cell::new(0),
            }));
        }
    }

    fn addresses(&self, query: &dyn Query) -> Result<Vec<DocAddress>> {
        Ok(self
            .searcher
            .search(query, &DocSetCollector)?
            .into_iter()
            .collect())
    }

    fn record(&self, address: DocAddress) -> Result<Record> {
        #[cfg(test)]
        self.record_decodes.set(self.record_decodes.get() + 1);
        read_record(&self.searcher, self.fields, address)
    }

    fn query(&self, query: &dyn Query) -> Result<Vec<(f32, Record)>> {
        self.addresses(query)?
            .into_iter()
            .map(|address| Ok((0.0, self.record(address)?)))
            .collect()
    }

    fn summary(&self, address: DocAddress) -> Result<Record> {
        #[cfg(test)]
        self.summary_decodes.set(self.summary_decodes.get() + 1);
        let doc: TantivyDocument = self.searcher.doc(address)?;
        let bytes = doc
            .get_first(self.fields.summary)
            .and_then(|v| v.as_bytes())
            .context("record summary missing")?;
        Ok(serde_json::from_slice(bytes)?)
    }

    pub fn workspace_roots(&self) -> Result<HashSet<workspace::Root>> {
        let mut roots = HashSet::new();
        for segment in self.searcher.segment_readers() {
            let index = segment.inverted_index(self.fields.workspace_root)?;
            let mut terms = index.terms().stream()?;
            while terms.advance() {
                let root = std::str::from_utf8(terms.key())?;
                let term = Term::from_field_text(self.fields.workspace_root, root);
                if let Some(mut postings) = index.read_postings(&term, IndexRecordOption::Basic)? {
                    while postings.doc() != TERMINATED {
                        if segment
                            .alive_bitset()
                            .is_none_or(|alive| alive.is_alive(postings.doc()))
                        {
                            roots.insert(workspace::Root::from_text(root));
                            break;
                        }
                        postings.advance();
                    }
                }
            }
        }
        Ok(roots)
    }

    /// The selection is shared by lexical and semantic search. Checkpoints
    /// never match, including an otherwise empty filter.
    pub fn selection(
        &self,
        filters: &Filters,
        exclude_current: bool,
        workspace: &workspace::Selection,
    ) -> Result<Box<dyn Query>> {
        let f = self.fields;
        let term = |field, value: &str| term_query(Term::from_field_text(field, value));
        let session = |id: &str| {
            Box::new(BooleanQuery::new(vec![
                (Occur::Should, term(f.session, id)),
                (Occur::Should, term(f.native_session, id)),
            ])) as Box<dyn Query>
        };
        let mut clauses = vec![
            (Occur::Must, Box::new(AllQuery) as Box<dyn Query>),
            (Occur::MustNot, term(f.id, CHECKPOINT_ID)),
        ];
        if !filters.provider.is_empty() {
            let providers = filters
                .provider
                .iter()
                .map(|p| (Occur::Should, term(f.provider, p.as_str())))
                .collect();
            clauses.push((Occur::Must, Box::new(BooleanQuery::new(providers))));
        }
        if let Some(id) = &filters.session {
            clauses.push((Occur::Must, session(id)));
        }
        for id in &filters.exclude_session {
            clauses.push((Occur::MustNot, session(id)));
        }
        match workspace {
            workspace::Selection::All => {}
            workspace::Selection::Current(root) => {
                clauses.push((Occur::Must, term(f.workspace_root, root.text())));
            }
            workspace::Selection::Patterns(roots) => {
                let terms = roots
                    .iter()
                    .map(|root| Term::from_field_text(f.workspace_root, root.text()));
                clauses.push((Occur::Must, Box::new(TermSetQuery::new(terms))));
            }
        }
        if filters.since.is_some() || filters.until.is_some() {
            use std::ops::Bound;
            let lower = filters
                .since
                .as_deref()
                .map(|s| crate::query::bound(s, false))
                .transpose()?
                .map(|s| Bound::Included(Term::from_field_text(f.timestamp, &s)))
                .unwrap_or(Bound::Unbounded);
            let upper = filters
                .until
                .as_deref()
                .map(|s| crate::query::bound(s, true))
                .transpose()?
                .map(|s| Bound::Excluded(Term::from_field_text(f.timestamp, &s)))
                .unwrap_or(Bound::Unbounded);
            clauses.push((Occur::Must, Box::new(RangeQuery::new(lower, upper))));
            clauses.push((Occur::MustNot, term(f.timestamp, "")));
        }
        if let Some(kind) = &filters.event_type {
            clauses.push((Occur::Must, term(f.event_type, kind.as_str())));
        }
        let kind = match filters.content_scope {
            crate::record::ContentScope::All => None,
            crate::record::ContentScope::Transcript => Some("message"),
            crate::record::ContentScope::Calls => Some("tool_call"),
            crate::record::ContentScope::Outputs => Some("tool_output"),
        };
        if let Some(kind) = kind {
            clauses.push((Occur::Must, term(f.event_type, kind)));
        }
        if let Some(file) = &filters.file {
            clauses.push((Occur::Must, term(f.file, &file_key(Path::new(file)))));
        }
        if filters.primary_only {
            clauses.push((
                Occur::Must,
                term_query(Term::from_field_bool(f.subagent, false)),
            ));
        }
        if exclude_current && !filters.include_current_session {
            for (provider, key) in [
                ("codex", "CODEX_THREAD_ID"),
                ("opencode", "OPENCODE_SESSION_ID"),
                ("claude", "CLAUDE_SESSION_ID"),
            ] {
                if let Ok(id) = std::env::var(key) {
                    clauses.push((
                        Occur::MustNot,
                        Box::new(BooleanQuery::new(vec![
                            (Occur::Must, term(f.provider, provider)),
                            (Occur::Must, term(f.native_session, &id)),
                        ])),
                    ));
                }
            }
        }
        Ok(Box::new(ConstScoreQuery::new(
            Box::new(BooleanQuery::new(clauses)),
            0.0,
        )))
    }

    pub fn selected_ids(&self, selection: &dyn Query) -> Result<HashSet<String>> {
        let columns = self
            .searcher
            .segment_readers()
            .iter()
            .map(|segment| {
                segment
                    .fast_fields()
                    .str("id")?
                    .context("id fast field missing")
            })
            .collect::<Result<Vec<_>>>()?;
        let mut ids = HashSet::new();
        for address in self.addresses(selection)? {
            ids.insert(string(
                &columns[address.segment_ord as usize],
                address.doc_id,
            )?);
        }
        Ok(ids)
    }

    pub fn page(
        &self,
        filters: &Filters,
        sessions: bool,
        limit: usize,
        cursor: Option<&str>,
        workspace: &workspace::Selection,
    ) -> Result<(Vec<Record>, Option<String>)> {
        let mut selection = self.selection(filters, false, workspace)?;
        if !sessions && let Some(id) = cursor {
            let selected_cursor = BooleanQuery::new(vec![
                (Occur::Must, selection.box_clone()),
                (
                    Occur::Must,
                    term_query(Term::from_field_text(self.fields.id, id)),
                ),
            ]);
            let address = self
                .addresses(&selected_cursor)?
                .into_iter()
                .next()
                .context("cursor no longer matches this selection")?;
            let previous = self.summary(address)?;
            use std::ops::Bound;
            let greater = |term| {
                Box::new(RangeQuery::new(Bound::Excluded(term), Bound::Unbounded)) as Box<dyn Query>
            };
            let at_time = || {
                term_query(Term::from_field_text(
                    self.fields.timestamp,
                    &previous.occurred_at,
                ))
            };
            let after = BooleanQuery::new(vec![
                (
                    Occur::Should,
                    greater(Term::from_field_text(
                        self.fields.timestamp,
                        &previous.occurred_at,
                    )),
                ),
                (
                    Occur::Should,
                    Box::new(BooleanQuery::new(vec![
                        (Occur::Must, at_time()),
                        (
                            Occur::Must,
                            greater(Term::from_field_u64(
                                self.fields.sequence,
                                previous.sequence,
                            )),
                        ),
                    ])),
                ),
                (
                    Occur::Should,
                    Box::new(BooleanQuery::new(vec![
                        (Occur::Must, at_time()),
                        (
                            Occur::Must,
                            term_query(Term::from_field_u64(
                                self.fields.sequence,
                                previous.sequence,
                            )),
                        ),
                        (
                            Occur::Must,
                            greater(Term::from_field_text(self.fields.id, id)),
                        ),
                    ])),
                ),
            ]);
            selection = Box::new(BooleanQuery::new(vec![
                (Occur::Must, selection),
                (Occur::Must, Box::new(after)),
            ]));
        }
        let mut rows = self.searcher.search(
            &*selection,
            &Winners {
                mode: if sessions {
                    LexicalMode::Sessions
                } else {
                    LexicalMode::Events(limit.saturating_add(1))
                },
                chronological: true,
            },
        )?;
        rows.sort_by(|a, b| b.cmp(a));
        let start = match cursor {
            Some(id) if sessions => {
                rows.iter()
                    .position(|r| r.priority.id() == id)
                    .context("cursor no longer matches this selection")?
                    + 1
            }
            _ => 0,
        };
        let end = start.saturating_add(limit).min(rows.len());
        let next = (end < rows.len()).then(|| rows[end - 1].priority.id().to_owned());
        let page = rows[start..end]
            .iter()
            .map(|r| {
                if sessions {
                    self.summary(r.address)
                } else {
                    self.record(r.address)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((page, next))
    }

    pub fn messages(&self) -> Result<Vec<ConversationMessage>> {
        let mut records = Vec::new();
        for (segment_ord, segment) in self.searcher.segment_readers().iter().enumerate() {
            let messages = segment.fast_fields().bool("conversation_message")?;
            for doc_id in segment.doc_ids_alive() {
                // Eligibility belongs to the pinned commit. Skip reasoning, tools,
                // metadata and checkpoints before reading their stored payloads.
                if !messages.first(doc_id).context("message marker missing")? {
                    continue;
                }
                let record = self.record(DocAddress::new(segment_ord as u32, doc_id))?;
                records.push(ConversationMessage::new(&record).context("invalid message marker")?);
            }
        }
        Ok(records)
    }

    /// Whether the pinned commit has no records. Checkpoints do not count.
    pub fn is_empty(&self) -> Result<bool> {
        let checkpoints = self.searcher.search(
            &TermQuery::new(
                Term::from_field_text(self.fields.id, CHECKPOINT_ID),
                IndexRecordOption::Basic,
            ),
            &Count,
        )?;
        Ok(self.searcher.num_docs() == checkpoints as u64)
    }

    #[cfg(test)]
    pub fn all(&self) -> Result<Vec<Record>> {
        let checkpoints = term_query(Term::from_field_text(self.fields.id, CHECKPOINT_ID));
        let events = BooleanQuery::new(vec![
            (Occur::Must, Box::new(AllQuery) as Box<dyn Query>),
            (Occur::MustNot, checkpoints),
        ]);
        Ok(self
            .query(&events)?
            .into_iter()
            .map(|(_, record)| record)
            .collect())
    }

    /// Build a query without field or boolean syntax, so code and URLs search as
    /// text. See [`query_groups`] for the accepted syntax. Returns `None` when
    /// the text has no tokens.
    fn text_query(&self, text: &str) -> Result<Option<Box<dyn Query>>> {
        let mut analyzer = self.index.tokenizer_for_field(self.fields.text)?;
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for (required, group) in query_groups(text) {
            let mut terms = Vec::new();
            analyzer.token_stream(group).process(&mut |token| {
                terms.push(Term::from_field_text(self.fields.text, &token.text));
            });
            // A group that tokenizes into several tokens (`std::fs`, `foo-bar`,
            // a quoted span) matches as a phrase.
            let query: Box<dyn Query> = match terms.len() {
                0 => continue,
                1 => Box::new(TermQuery::new(
                    terms.remove(0),
                    IndexRecordOption::WithFreqs,
                )),
                _ => Box::new(PhraseQuery::new(terms)),
            };
            let occur = if required { Occur::Must } else { Occur::Should };
            clauses.push((occur, query));
        }
        Ok((!clauses.is_empty()).then(|| Box::new(BooleanQuery::new(clauses)) as Box<dyn Query>))
    }

    pub fn lexical(
        &self,
        terms: &[String],
        mode: LexicalMode,
        selection: &dyn Query,
    ) -> Result<LexicalResults> {
        if matches!(mode, LexicalMode::Events(0)) {
            return Ok(LexicalResults {
                rows: Vec::new(),
                addresses: HashMap::new(),
                generators: Vec::new(),
            });
        }
        let mut scores = BTreeMap::<DocAddress, Candidate>::new();
        let mut generators = Vec::new();
        for term in terms {
            let Some(query) = self.text_query(term)? else {
                continue;
            };
            let mut generator =
                SnippetGenerator::create(&self.searcher, &*query, self.fields.text)?;
            generator.set_max_num_chars(crate::record::SEARCH_SNIPPET_CHARS);
            generators.push(generator);
            let filtered = BooleanQuery::new(vec![
                (Occur::Must, query),
                (Occur::Must, selection.box_clone()),
            ]);
            for row in self.searcher.search(
                &filtered,
                &Winners {
                    mode,
                    chronological: false,
                },
            )? {
                let best = scores.entry(row.address).or_insert_with(|| row.clone());
                if row > *best {
                    *best = row;
                }
            }
        }
        let mut candidates: Vec<_> = scores.into_values().collect();
        candidates.sort_by(|a, b| b.cmp(a));
        match mode {
            LexicalMode::Events(limit) => candidates.truncate(limit),
            LexicalMode::Sessions => {
                let mut seen = HashSet::new();
                candidates.retain(|row| seen.insert(row.session.clone()));
            }
            #[cfg(test)]
            LexicalMode::All => {}
        }
        let mut addresses = HashMap::new();
        let rows = candidates
            .into_iter()
            .map(|row| {
                let address = row.address;
                #[cfg(test)]
                self.summary_decodes.set(self.summary_decodes.get() + 1);
                let doc: TantivyDocument = self.searcher.doc(address)?;
                let bytes = doc
                    .get_first(self.fields.summary)
                    .and_then(|v| v.as_bytes())
                    .context("record summary missing")?;
                let record: Record = serde_json::from_slice(bytes)?;
                addresses.insert(record.event_id.clone(), address);
                Ok((row.score, record))
            })
            .collect::<Result<_>>()?;
        Ok(LexicalResults {
            rows,
            addresses,
            generators,
        })
    }

    pub fn by_session(&self, id: &str) -> Result<Vec<Record>> {
        Ok(self
            .query(&TermQuery::new(
                Term::from_field_text(self.fields.session, id),
                IndexRecordOption::Basic,
            ))?
            .into_iter()
            .map(|(_, record)| record)
            .collect())
    }

    pub fn window(&self, record: &Record, radius: usize) -> Result<Vec<Record>> {
        let mut addresses = self.addresses(&*term_query(Term::from_field_text(
            self.fields.session,
            &record.session_id,
        )))?;
        addresses.sort_unstable_by_key(|a| (a.segment_ord, a.doc_id));
        let mut ordered = Vec::with_capacity(addresses.len());
        // Read ordering fields before loading the window's stored payloads.
        for group in addresses.chunk_by(|a, b| a.segment_ord == b.segment_ord) {
            let fields = self
                .searcher
                .segment_reader(group[0].segment_ord)
                .fast_fields();
            let time = fields
                .str("timestamp")?
                .context("timestamp fast field missing")?;
            let id = fields.str("id")?.context("id fast field missing")?;
            let sequence = fields.u64("sequence")?;
            for address in group {
                ordered.push((
                    Priority::Chronological(
                        string(&time, address.doc_id)?,
                        sequence.first(address.doc_id).unwrap_or(0),
                        string(&id, address.doc_id)?,
                    ),
                    *address,
                ));
            }
        }
        ordered.sort_unstable_by(|(a, _), (b, _)| b.compare(a));
        let selected = ordered
            .iter()
            .position(|(order, _)| order.id() == record.event_id)
            .context("event missing from session")?;
        let end = selected
            .saturating_add(radius)
            .saturating_add(1)
            .min(ordered.len());
        ordered[selected.saturating_sub(radius)..end]
            .iter()
            .map(|(_, address)| self.record(*address))
            .collect()
    }

    fn reference_query(&self, field: Field, id: &str) -> Result<Box<dyn Query>> {
        ensure!(
            !id.is_empty() && id.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid trwl reference {id:?}"
        );
        Ok(if id.len() == 64 {
            term_query(Term::from_field_text(field, id))
        } else {
            Box::new(RegexQuery::from_pattern(&format!("{id}.*"), field)?)
        })
    }

    pub fn event_reference(&self, id: &str) -> Result<Vec<Record>> {
        Ok(self
            .query(&*self.reference_query(self.fields.id, id)?)?
            .into_iter()
            .map(|(_, record)| record)
            .collect())
    }

    pub fn session_reference(&self, id: &str) -> Result<Vec<Record>> {
        let mut clauses = vec![(
            Occur::Should,
            term_query(Term::from_field_text(self.fields.native_session, id)),
        )];
        if id.bytes().all(|b| b.is_ascii_hexdigit()) {
            clauses.push((
                Occur::Should,
                self.reference_query(self.fields.session, id)?,
            ));
        }
        self.searcher
            .search(
                &BooleanQuery::new(clauses),
                &Winners {
                    mode: LexicalMode::Sessions,
                    chronological: true,
                },
            )?
            .into_iter()
            .map(|row| self.summary(row.address))
            .collect()
    }
}

/// Split search text into `(required, group)` pairs. Groups are
/// whitespace-separated words or double-quoted spans; an unclosed quote extends
/// to the end. A `+` immediately before a word or opening quote requires it;
/// other groups are alternatives whose scores add up.
fn query_groups(text: &str) -> Vec<(bool, &str)> {
    let mut groups = Vec::new();
    let mut rest = text;
    loop {
        rest = rest.trim_start();
        let required = rest.starts_with('+');
        if required {
            rest = &rest[1..];
        }
        if rest.is_empty() {
            break;
        }
        let (group, next) = if let Some(quoted) = rest.strip_prefix('"') {
            quoted.split_once('"').unwrap_or((quoted, ""))
        } else {
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '"')
                .unwrap_or(rest.len());
            (&rest[..end], &rest[end..])
        };
        // A lone `+` yields an empty group; it requires nothing.
        if !group.is_empty() {
            groups.push((required, group));
        }
        rest = next;
    }
    groups
}

#[cfg(test)]
pub fn records(root: &Path) -> Result<Vec<Record>> {
    ReadIndex::open(root)?.map_or_else(|| Ok(Vec::new()), |index| index.all())
}

fn segment_version(segments: impl IntoIterator<Item = (String, u32)>) -> String {
    let mut segments: Vec<_> = segments
        .into_iter()
        .map(|(id, deleted)| format!("{id}:{deleted}"))
        .collect();
    segments.sort();
    digest(&segments.iter().map(String::as_str).collect::<Vec<_>>())
}

/// Changes to lexical-only content, parser state and segment layout do not
/// invalidate conversation planning. `None` without an index.
pub fn semantic_version(root: &Path) -> Result<Option<String>> {
    let path = root.join("index");
    if !path.join("meta.json").try_exists()? {
        return Ok(None);
    }
    let meta = Index::open_in_dir(&path)?.load_metas()?;
    Ok(Some(
        decode_manifest(meta.payload.as_deref())?
            .semantic_version
            .unwrap_or_else(|| {
                segment_version(
                    meta.segments
                        .iter()
                        .map(|s| (s.id().uuid_string(), s.num_deleted_docs())),
                )
            }),
    ))
}

fn read_record(searcher: &Searcher, fields: Fields, address: DocAddress) -> Result<Record> {
    let doc: TantivyDocument = searcher.doc(address)?;
    let bytes = doc
        .get_first(fields.record)
        .and_then(|v| v.as_bytes())
        .context("record missing from index")?;
    let stored: StoredRecord = serde_json::from_slice(bytes)?;
    stored.restore(
        doc.get_first(fields.text)
            .and_then(|v| v.as_str())
            .unwrap_or(""),
    )
}

fn message_hash(record: &Record) -> Result<Option<String>> {
    ConversationMessage::new(record)
        .map(|message| Ok(digest(&[&serde_json::to_string(&message)?])))
        .transpose()
}

fn removed_message_hashes(
    searcher: &Searcher,
    fields: Fields,
    query: &dyn Query,
    hashes: &mut Vec<String>,
) -> Result<()> {
    for address in searcher.search(query, &DocSetCollector)? {
        let messages = searcher
            .segment_reader(address.segment_ord)
            .fast_fields()
            .bool("conversation_message")?;
        if messages
            .first(address.doc_id)
            .context("message marker missing")?
        {
            let record = read_record(searcher, fields, address)?;
            hashes.push(message_hash(&record)?.context("invalid message marker")?);
        }
    }
    Ok(())
}

fn term_query(term: Term) -> Box<dyn Query> {
    Box::new(TermQuery::new(term, IndexRecordOption::Basic))
}

fn file_key(path: &Path) -> String {
    digest(&[&path.components().as_path().to_string_lossy()])
}

fn checkpoint_query(fields: Fields, key: &str) -> Box<dyn Query> {
    Box::new(BooleanQuery::new(vec![
        (
            Occur::Must,
            term_query(Term::from_field_text(fields.source, key)),
        ),
        (
            Occur::Must,
            term_query(Term::from_field_text(fields.id, CHECKPOINT_ID)),
        ),
    ]))
}

fn load_checkpoint(
    searcher: &Searcher,
    fields: Fields,
    key: &str,
) -> Result<Option<providers::Checkpoint>> {
    let Some((_, address)) = searcher
        .search(
            &*checkpoint_query(fields, key),
            &TopDocs::with_limit(1).order_by_score(),
        )?
        .pop()
    else {
        return Ok(None);
    };
    let doc: TantivyDocument = searcher.doc(address)?;
    let bytes = doc
        .get_first(fields.record)
        .and_then(|v| v.as_bytes())
        .context("checkpoint missing from index")?;
    Ok(Some(
        serde_json::from_slice(bytes).context("decoding input checkpoint")?,
    ))
}

struct PlannedInput {
    key: String,
    provider: Provider,
    path: PathBuf,
    stamp: Stamp,
    changed: bool,
}
struct InputPlan {
    inputs: Vec<PlannedInput>,
    removed: HashSet<String>,
    errors: Vec<Diagnostic>,
    discovered: usize,
}

fn source_listing(source: &Source) -> Result<(Provider, PathBuf, Vec<PathBuf>)> {
    let provider = source.provider;
    Ok((
        provider,
        source.root.canonicalize()?,
        providers::files(provider, &source.root)?,
    ))
}

fn plan_inputs(sources: &[Source], manifest: &Manifest) -> InputPlan {
    let mut visited = HashSet::new();
    let mut plan = InputPlan {
        inputs: Vec::new(),
        removed: HashSet::new(),
        errors: Vec::new(),
        discovered: 0,
    };
    for source in sources {
        let (provider, root, paths) = match source_listing(source) {
            Ok(listing) => listing,
            Err(error) => {
                plan.errors.push(Diagnostic {
                    path: source.root.clone(),
                    detail: format!("{error:#}"),
                });
                continue;
            }
        };
        let present: HashSet<_> = paths.iter().collect();
        // Inaccessible roots preserve history; removing a source registration
        // does not delete its records.
        if root.is_dir() {
            for (key, input) in &manifest.inputs {
                if input.provider == provider
                    && input.path.starts_with(&root)
                    && !present.contains(&input.path)
                {
                    plan.removed.insert(key.clone());
                }
            }
        }
        for path in paths {
            let key = digest(&[provider.as_str(), &path.to_string_lossy()]);
            if !visited.insert(key.clone()) {
                continue;
            }
            match stamp(&path, provider == Provider::Opencode) {
                Ok(stamp) => {
                    let changed = !manifest
                        .inputs
                        .get(&key)
                        .is_some_and(|old| old.unchanged(&stamp));
                    plan.inputs.push(PlannedInput {
                        key,
                        provider,
                        path,
                        stamp,
                        changed,
                    });
                }
                Err(error) => plan.errors.push(Diagnostic {
                    path,
                    detail: format!("{error:#}"),
                }),
            }
        }
    }
    plan.discovered = visited.len();
    plan
}

fn stage_input(
    input: &PlannedInput,
    previous: Option<providers::Checkpoint>,
    spool: &mut Option<BufWriter<File>>,
    root: &Path,
    parse: &mut impl FnMut(
        Provider,
        &Path,
        Option<&providers::Checkpoint>,
        &mut dyn FnMut(&Record) -> Result<()>,
    ) -> Result<providers::Parsed>,
) -> Result<(providers::Parsed, (u64, u64), bool)> {
    // Stage records on disk. Failed sources leave unused byte ranges;
    // one unlinked file bounds memory and open descriptors.
    if spool.is_none() {
        *spool = Some(BufWriter::new(tempfile::tempfile_in(root)?));
    }
    let spool = spool.as_mut().unwrap();
    let start = spool.stream_position()?;
    let parsed = parse(
        input.provider,
        &input.path,
        previous.as_ref(),
        &mut |record| {
            serde_json::to_writer(&mut *spool, record)?;
            spool.write_all(b"\n")?;
            Ok(())
        },
    )?;
    let length = spool.stream_position()? - start;
    // SQLite parsing owns a read transaction. Keep its pre-read stamp so later
    // commits cause another scan, not a rejection of the snapshot just read.
    ensure!(
        input.provider == Provider::Opencode || input.stamp == stamp(&input.path, false)?,
        "input changed during parsing; retry import"
    );
    let unchanged = previous.as_ref() == Some(&parsed.checkpoint);
    Ok((parsed, (start, length), unchanged))
}

pub fn status(config: &Config) -> Result<LexicalStatus> {
    let path = config.root.join("index");
    let initialized = path.join("meta.json").try_exists()?;
    let (manifest, events) = if initialized {
        let index = Index::open_in_dir(&path)?;
        ensure!(
            index.schema() == schema().0,
            "incompatible history index at {}; move that directory aside, then re-import your sources",
            path.display()
        );
        // Counts and source stamps must describe the same published commit.
        let meta = index.load_metas()?;
        let manifest = decode_manifest(meta.payload.as_deref())?;
        let documents: u64 = meta
            .segments
            .iter()
            .map(|segment| u64::from(segment.num_docs()))
            .sum();
        // Each checkpointed input has one checkpoint document.
        let checkpoints = manifest.inputs.values().filter(|i| i.checkpointed).count();
        (manifest, documents - checkpoints as u64)
    } else {
        (Manifest::default(), 0)
    };
    let plan = plan_inputs(&config.sources()?, &manifest);
    let pending = plan.inputs.iter().filter(|input| input.changed).count();
    let active = is_locked(&config.root, "update.lock")?;
    let state = if !plan.errors.is_empty() {
        State::Blocked
    } else if active {
        State::CatchingUp
    } else if !initialized {
        State::Uninitialized
    } else if pending > 0 || !plan.removed.is_empty() {
        State::Pending
    } else {
        State::Ready
    };
    Ok(LexicalStatus {
        state,
        active,
        indexed_events: events,
        indexed_inputs: manifest.inputs.len(),
        discovered_inputs: plan.discovered,
        pending_inputs: pending,
        removed_inputs: plan.removed.len(),
        errors: plan.errors,
    })
}

pub fn update(config: &Config, sources: &[Source], progress: bool) -> Result<ImportReport> {
    update_with_parser(
        config,
        sources,
        progress,
        false,
        |provider, path, previous, emit| providers::parse(provider, path, previous, emit),
    )
}

/// Reparse every input without its stored checkpoint. This catches edits that
/// incremental imports miss, such as an OpenCode row rewritten in place with
/// the same size and `time_updated`.
pub fn update_full(config: &Config, sources: &[Source], progress: bool) -> Result<ImportReport> {
    update_with_parser(
        config,
        sources,
        progress,
        true,
        |provider, path, previous, emit| providers::parse(provider, path, previous, emit),
    )
}

fn update_with_parser(
    config: &Config,
    sources: &[Source],
    progress: bool,
    full: bool,
    mut parse: impl FnMut(
        Provider,
        &Path,
        Option<&providers::Checkpoint>,
        &mut dyn FnMut(&Record) -> Result<()>,
    ) -> Result<providers::Parsed>,
) -> Result<ImportReport> {
    let _guard = lock(&config.root, "update.lock")?;
    if progress {
        eprintln!(
            "{}",
            json!({"type": "trwl_progress", "operation": "import", "status": "started"})
        );
    }
    let path = config.root.join("index");
    fs::create_dir_all(&path)?;
    let (schema, fields) = schema();
    let index = if path.join("meta.json").exists() {
        Index::open_in_dir(&path)?
    } else {
        Index::create_in_dir(&path, schema.clone())?
    };
    ensure!(
        index.schema() == schema,
        "incompatible history index at {}; move that directory aside, then re-import your sources",
        path.display()
    );
    let meta = index.load_metas()?;
    let mut manifest = decode_manifest(meta.payload.as_deref())?;
    let previous_version = manifest.semantic_version.clone().unwrap_or_else(|| {
        segment_version(
            meta.segments
                .iter()
                .map(|s| (s.id().uuid_string(), s.num_deleted_docs())),
        )
    });
    // Checkpoints load from the commit the manifest belongs to; the update
    // lock excludes other writers.
    let searcher = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()?
        .searcher();
    let plan = plan_inputs(sources, &manifest);
    let mut failed = !plan.errors.is_empty();
    let mut reports: Vec<_> = plan.errors.into_iter().map(ImportSource::Failed).collect();
    let mut changes = BTreeMap::new();
    let mut spool: Option<BufWriter<File>> = None;
    for input in plan.inputs {
        let PlannedInput {
            key,
            provider,
            path,
            ..
        } = &input;
        if !input.changed && !full {
            reports.push(ImportSource::Unchanged {
                path: path.clone(),
                unchanged: true,
            });
            continue;
        }
        let previous = match manifest.inputs.get(key) {
            Some(old)
                if old.checkpointed
                    && old.parser_revision == providers::parser_revision(*provider)
                    && !full =>
            {
                load_checkpoint(&searcher, fields, key)
            }
            _ => Ok(None),
        };
        let parsed = previous.and_then(|previous| {
            stage_input(&input, previous, &mut spool, &config.root, &mut parse)
        });
        match parsed {
            Ok((parsed, range, unchanged)) => {
                reports.push(ImportSource::Parsed {
                    path: path.clone(),
                    events: parsed.records,
                    rejected_records: parsed.diagnostics.count,
                    rejection_diagnostics: parsed.diagnostics.first,
                });
                // An unchanged checkpoint is not rewritten, so a stamp-only
                // change leaves the published records untouched. Replacing
                // every record also deletes the old checkpoint.
                let keep = unchanged && !matches!(parsed.replace, providers::Replace::All);
                changes.insert(
                    key.clone(),
                    Change {
                        provider: *provider,
                        replace: parsed.replace,
                        range: Some(range),
                        checkpoint: (!keep).then_some(parsed.checkpoint),
                    },
                );
                manifest.inputs.insert(
                    key.clone(),
                    InputState {
                        provider: *provider,
                        path: path.clone(),
                        parser_revision: providers::parser_revision(*provider),
                        stamp: input.stamp,
                        checkpointed: true,
                    },
                );
            }
            Err(error) => {
                failed = true;
                reports.push(ImportSource::Failed(Diagnostic {
                    path: path.clone(),
                    detail: format!("{error:#}"),
                }));
            }
        }
    }
    // Remove missing files only after successfully listing their directory.
    // An inaccessible root must not erase its indexed history.
    for key in plan.removed {
        let input = manifest
            .inputs
            .remove(&key)
            .context("planned removal missing")?;
        changes.insert(
            key,
            Change {
                provider: input.provider,
                replace: providers::Replace::All,
                range: None,
                checkpoint: None,
            },
        );
    }
    if !changes.is_empty() {
        let mut spool = spool
            .map(|writer| writer.into_inner().map_err(|error| error.into_error()))
            .transpose()?;
        // The budget is split across workers. Small per-worker buffers cause
        // frequent segment flushes and merges for content-heavy transcripts.
        let mut writer = index.writer_with_num_threads(WRITER_THREADS, WRITER_MEMORY_BYTES)?;
        let mut removed_messages = Vec::new();
        let mut added_messages = Vec::new();
        for (key, change) in changes {
            let source = Term::from_field_text(fields.source, &key);
            match &change.replace {
                // Also deletes the checkpoint document.
                providers::Replace::All => {
                    removed_message_hashes(
                        &searcher,
                        fields,
                        &*term_query(source.clone()),
                        &mut removed_messages,
                    )?;
                    writer.delete_term(source.clone());
                }
                providers::Replace::Append => {}
                providers::Replace::Sessions(sessions) => {
                    for suffix in sessions {
                        let session = Term::from_field_text(
                            fields.session,
                            &crate::record::session_id(change.provider, &suffix.native_session),
                        );
                        let query = BooleanQuery::new(vec![
                            (Occur::Must, term_query(source.clone())),
                            (Occur::Must, term_query(session)),
                            (
                                Occur::Must,
                                Box::new(RangeQuery::new(
                                    std::ops::Bound::Included(Term::from_field_u64(
                                        fields.sequence,
                                        suffix.from_sequence,
                                    )),
                                    std::ops::Bound::Unbounded,
                                )),
                            ),
                        ]);
                        removed_message_hashes(&searcher, fields, &query, &mut removed_messages)?;
                        writer.delete_query(Box::new(query))?;
                    }
                }
            }
            if let Some(checkpoint) = &change.checkpoint {
                writer.delete_query(checkpoint_query(fields, &key))?;
                writer.add_document(doc!(
                    fields.source => key.clone(),
                    fields.id => CHECKPOINT_ID,
                    fields.message => false,
                    fields.record => serde_json::to_vec(checkpoint)?
                ))?;
            }
            let Some((start, length)) = change.range else {
                continue;
            };
            let spool = spool.as_mut().context("parsed source spool missing")?;
            spool.seek(SeekFrom::Start(start))?;
            for record in serde_json::Deserializer::from_reader(BufReader::new(spool.take(length)))
                .into_iter::<Record>()
            {
                let mut record = record.context("reading parsed source spool")?;
                record.workspace_root = manifest
                    .workspace_roots
                    .resolve(&record.session_id, &record.workspace);
                if let Some(hash) = message_hash(&record)? {
                    added_messages.push(hash);
                }
                let full = serde_json::to_vec(&StoredRecord::new(record.clone()))?;
                let message = record.is_conversation_message();
                let mut summary = record.clone();
                summary.text.clear();
                summary.structured_content = Value::Null;
                summary.files.clear();
                let mut document = doc!(
                    fields.source => key.clone(),
                    fields.id => record.event_id.clone(),
                    fields.session => record.session_id.clone(),
                    fields.message => message,
                    fields.text => record.text,
                    fields.record => full,
                    fields.summary => serde_json::to_vec(&summary)?,
                    fields.provider => record.provider.as_str(),
                    fields.native_session => record.provider_session_id,
                    fields.timestamp => record.occurred_at,
                    fields.event_type => record.event_type.as_str(),
                    fields.subagent => record.is_subagent,
                    fields.sequence => record.sequence
                );
                if let Some(root) = &record.workspace_root {
                    document.add_text(fields.workspace_root, root.text());
                }
                // Hash normalized suffixes: paths can exceed a term's length
                // limit, and file filters match path components, not substrings.
                for file in record.files {
                    let mut components = Path::new(&file).components();
                    while !components.as_path().as_os_str().is_empty() {
                        document.add_text(fields.file, file_key(components.as_path()));
                        components.next();
                    }
                }
                writer.add_document(document)?;
            }
        }
        // Compare the actual pairing inputs, including result metadata. Sorting
        // makes suffix replay and segment order irrelevant while retaining duplicates.
        removed_messages.sort();
        added_messages.sort();
        manifest.semantic_version = Some(if removed_messages == added_messages {
            previous_version
        } else {
            digest(&[
                &previous_version,
                &serde_json::to_string(&(removed_messages, added_messages))?,
            ])
        });
        // Checkpoints and records become visible together; an interrupted writer
        // cannot mark input as processed without publishing its records.
        let mut commit = writer.prepare_commit()?;
        commit.set_payload(&serde_json::to_string(&manifest)?);
        commit.commit()?;
        // Dropping a writer prevents its merges from publishing but does not stop
        // their work. Finish them so later updates do not repeat discarded merges.
        writer.wait_merging_threads()?;
    }
    Ok(ImportReport {
        outcome: if failed {
            Outcome::CompletedWithSourceFailures
        } else {
            Outcome::Completed
        },
        sources: reports,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        record::EventType,
        test_support::{import, opencode_db, source},
    };

    fn source_stats(report: &ImportReport) -> (usize, usize, usize) {
        match &report.sources[0] {
            ImportSource::Parsed {
                events,
                rejected_records,
                rejection_diagnostics,
                ..
            } => (*events, *rejected_records, rejection_diagnostics.len()),
            _ => panic!("expected parsed source"),
        }
    }

    const HEADER: &str =
        "{\"type\":\"session_meta\",\"payload\":{\"id\":\"session-1\",\"cwd\":\"/workspace\"}}\n";

    fn input(path: &Path, text: &str) {
        let row = json!({"type": "response_item", "timestamp": "2026-01-01T00:00:00Z", "payload": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]
        }});
        fs::write(path, format!("{HEADER}{row}\n")).unwrap();
    }

    #[test]
    fn stored_text_references_restore_native_strings_and_escaped_keys() {
        let text = "日本語 / ~ 🦀 repeated native content ".repeat(100);
        let mut record = Record::new(
            Provider::Codex,
            "session",
            "event",
            EventType::Message,
            Path::new("input"),
        );
        record.text = text.clone();
        record.structured_content =
            json!({"content":[{"text":text}], "a/~b":text, "other":"different"});
        let expected = serde_json::to_value(&record).unwrap();
        let stored = StoredRecord::new(record);
        assert_eq!(stored.references.len(), 2);
        assert!(stored.record.text.is_empty());
        assert!(serde_json::to_vec(&stored).unwrap().len() < text.len());
        let restored = stored.restore(&text).unwrap();
        assert_eq!(serde_json::to_value(restored).unwrap(), expected);
    }

    #[test]
    fn lexical_snippets_show_late_matches_without_loading_native_content() {
        let (_dir, config, path) = crate::test_support::history_file("input.jsonl");
        let text = "prefix filler ".repeat(500) + "late needle match 日本語";
        input(&path, &text);
        let index = import(&config, Provider::Codex, &path);
        let selection = index
            .selection(&Filters::default(), false, &workspace::Selection::All)
            .unwrap();
        let hits = index
            .lexical(&["needle".into()], LexicalMode::Events(1), &*selection)
            .unwrap();
        assert_eq!(hits.rows.len(), 1);
        let snippet = hits.snippet(&index, &hits.rows[0].1.event_id).unwrap();
        assert!(snippet.contains("needle"));
        assert!(snippet.chars().count() <= crate::record::SEARCH_SNIPPET_CHARS);
        assert_eq!(index.record_decodes.get(), 0);
        let full = index
            .event_reference(&hits.rows[0].1.event_id)
            .unwrap()
            .remove(0);
        assert_eq!(full.text, text);
        assert_eq!(full.structured_content["content"][0]["text"], text);
    }

    #[test]
    fn incompatible_indexes_remain_untouched_and_explain_reimport() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(dir.path()).unwrap();
        let path = dir.path().join("index");
        fs::create_dir(&path).unwrap();
        let mut old = Schema::builder();
        old.add_text_field("obsolete", TEXT);
        Index::create_in_dir(&path, old.build()).unwrap();
        let metadata = fs::read(path.join("meta.json")).unwrap();
        for error in [
            ReadIndex::open(dir.path()).err().unwrap(),
            status(&config).err().unwrap(),
            update(&config, &[], false).err().unwrap(),
        ] {
            assert!(
                error
                    .to_string()
                    .contains("move that directory aside, then re-import")
            );
        }
        assert_eq!(fs::read(path.join("meta.json")).unwrap(), metadata);
    }

    fn assert_full_parse(config: &Config, path: &Path) -> Vec<Record> {
        let mut expected = Vec::new();
        providers::parse(Provider::Codex, path, None, |record| {
            expected.push(record.clone());
            Ok(())
        })
        .unwrap();
        let mut actual = records(&config.root).unwrap();
        actual.sort_by_key(|record| record.sequence);
        assert_eq!(
            serde_json::to_value(&actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        actual
    }

    #[test]
    fn jsonl_prefix_edits_and_truncation_replace_records() {
        let (dir, config, path) = crate::test_support::history_file("rollout.jsonl");
        input(&path, "original");
        let row = fs::read_to_string(&path).unwrap()[HEADER.len()..].to_owned();
        let survivor = row.replace("original", "survivor");
        fs::write(&path, format!("{HEADER}{row}{survivor}")).unwrap();
        let source = source(Provider::Codex, &path);
        update(&config, std::slice::from_ref(&source), false).unwrap();
        let pinned = ReadIndex::open(&config.root).unwrap().unwrap();
        let survivor_id = assert_full_parse(&config, &path)[1].event_id.clone();
        // Edit the old prefix while growing the file, then rewrite at the same
        // length, truncate, and replace its inode. None may reuse stale records.
        for text in [
            format!(
                "{HEADER}{}{}{}",
                row.replace("original", "changed!"),
                survivor,
                row
            ),
            format!(
                "{HEADER}{}{}{}",
                row.replace("original", "altered!"),
                survivor,
                row
            ),
            format!("{HEADER}{survivor}"),
        ] {
            fs::write(&path, text).unwrap();
            update(&config, std::slice::from_ref(&source), false).unwrap();
            let actual = assert_full_parse(&config, &path);
            assert_eq!(
                actual
                    .iter()
                    .find(|record| record.text == "survivor")
                    .unwrap()
                    .event_id,
                survivor_id
            );
        }
        let replacement = dir.path().join("replacement");
        fs::write(&replacement, format!("{HEADER}{row}{survivor}")).unwrap();
        fs::rename(replacement, &path).unwrap();
        update(&config, &[source], false).unwrap();
        assert_full_parse(&config, &path);
        assert_eq!(pinned.all().unwrap().len(), 2);
    }

    #[test]
    fn failed_append_preserves_history_and_can_be_retried() {
        let (_dir, config, path) = crate::test_support::history_file("rollout.jsonl");
        input(&path, "original");
        let row = fs::read_to_string(&path).unwrap()[HEADER.len()..].to_owned();
        let source = source(Provider::Codex, &path);
        update(&config, std::slice::from_ref(&source), false).unwrap();
        let pinned = ReadIndex::open(&config.root).unwrap().unwrap();
        fs::write(
            &path,
            format!("{HEADER}{row}{row}{{\"type\":\"session_meta\",\"payload\":{{}}}}\n"),
        )
        .unwrap();
        let report = update(&config, std::slice::from_ref(&source), false).unwrap();
        assert!(!report.completed());
        let current = ReadIndex::open(&config.root).unwrap().unwrap();
        assert_eq!(current.all().unwrap().len(), 1);
        fs::write(&path, format!("{HEADER}{row}{row}")).unwrap();
        update(&config, std::slice::from_ref(&source), false).unwrap();
        assert_eq!(assert_full_parse(&config, &path).len(), 2);
        // Even an append without accepted records publishes its parser state.
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"malformed\n")
            .unwrap();
        let report = update(&config, std::slice::from_ref(&source), false).unwrap();
        assert!(report.completed());
        assert_eq!(source_stats(&report), (2, 1, 1));
        assert_eq!(assert_full_parse(&config, &path).len(), 2);
        assert_eq!(pinned.all().unwrap().len(), 1);
        let unchanged = update(&config, &[source], false).unwrap();
        assert!(matches!(
            unchanged.sources[0],
            ImportSource::Unchanged {
                unchanged: true,
                ..
            }
        ));
    }

    #[test]
    fn checkpoints_are_documents_outside_the_manifest_and_records() {
        let (_dir, config, path) = crate::test_support::history_file("rollout.jsonl");
        let source = source(Provider::Codex, &path);
        let payload = || {
            Index::open_in_dir(config.root.join("index"))
                .unwrap()
                .load_metas()
                .unwrap()
                .payload
                .unwrap()
        };
        // Stamps vary in width; compare the content revision's size, not its value.
        let unstamped = || {
            let mut manifest: Value = serde_json::from_str(&payload()).unwrap();
            manifest["semantic_version"] =
                json!(manifest["semantic_version"].as_str().unwrap().len());
            for input in manifest["inputs"].as_object_mut().unwrap().values_mut() {
                input.as_object_mut().unwrap().remove("stamp");
            }
            manifest
        };
        // An input with session metadata but no records still has a checkpoint.
        fs::write(&path, HEADER).unwrap();
        update(&config, std::slice::from_ref(&source), false).unwrap();
        let index = ReadIndex::open(&config.root).unwrap().unwrap();
        assert!(index.is_empty().unwrap());
        assert!(index.all().unwrap().is_empty());
        assert!(index.messages().unwrap().is_empty());
        assert_eq!(status(&config).unwrap().indexed_events, 0);

        // Rejections and records without native IDs do not grow the manifest,
        // and reports keep a bounded sample of diagnostics.
        input(&path, "record");
        update(&config, std::slice::from_ref(&source), false).unwrap();
        let row = fs::read_to_string(&path).unwrap()[HEADER.len()..].to_owned();
        let small = unstamped();
        fs::write(
            &path,
            format!("{HEADER}{}{}", row.repeat(50), "malformed\n".repeat(50)),
        )
        .unwrap();
        let report = update(&config, std::slice::from_ref(&source), false).unwrap();
        assert_eq!(
            source_stats(&report),
            (50, 50, providers::Diagnostics::LIMIT)
        );
        assert_eq!(unstamped(), small);
        assert!(!payload().contains("fallback_counts"));
        assert_full_parse(&config, &path);
        assert_eq!(status(&config).unwrap().indexed_events, 50);

        // Resuming from the stored checkpoint appends only new records.
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(row.as_bytes())
            .unwrap();
        let mut parsed = 0;
        let report = update_with_parser(
            &config,
            std::slice::from_ref(&source),
            false,
            false,
            |provider, path, previous, emit| {
                assert!(previous.is_some(), "stored checkpoint was not loaded");
                providers::parse(provider, path, previous, |record| {
                    parsed += 1;
                    emit(record)
                })
            },
        )
        .unwrap();
        assert_eq!(parsed, 1);
        assert_eq!(source_stats(&report).1, 50);
        assert_full_parse(&config, &path);

        // A stamp change without new content keeps the published records.
        let version = semantic_version(&config.root).unwrap();
        let file = File::options().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::UNIX_EPOCH)
            .unwrap();
        drop(file);
        let report = update(&config, std::slice::from_ref(&source), false).unwrap();
        assert_eq!(source_stats(&report).0, 51);
        assert_eq!(semantic_version(&config.root).unwrap(), version);

        // A manifest from an earlier version, with inline checkpoints, causes
        // one full reparse that replaces every record and checkpoint.
        let index = Index::open_in_dir(config.root.join("index")).unwrap();
        let mut old: Value = serde_json::from_str(&payload()).unwrap();
        old.as_object_mut().unwrap().remove("semantic_version");
        for input in old["inputs"].as_object_mut().unwrap().values_mut() {
            let input = input.as_object_mut().unwrap();
            input.remove("checkpointed");
            input.insert("checkpoint".into(), json!({"Jsonl": {}}));
        }
        let mut writer = index
            .writer_with_num_threads::<TantivyDocument>(1, 15_000_000)
            .unwrap();
        let mut commit = writer.prepare_commit().unwrap();
        commit.set_payload(&old.to_string());
        commit.commit().unwrap();
        drop(writer);
        let old_version = ReadIndex::open(&config.root).unwrap().unwrap().version();
        let mut previous = Vec::new();
        update_with_parser(
            &config,
            std::slice::from_ref(&source),
            false,
            false,
            |provider, path, checkpoint, emit| {
                previous.push(checkpoint.is_some());
                providers::parse(provider, path, checkpoint, emit)
            },
        )
        .unwrap();
        assert_eq!(previous, [false]);
        assert_eq!(semantic_version(&config.root).unwrap(), Some(old_version));
        assert_full_parse(&config, &path);
        let searcher = index.reader().unwrap().searcher();
        let checkpoints = searcher
            .search(
                &TermQuery::new(
                    Term::from_field_text(schema().1.id, CHECKPOINT_ID),
                    IndexRecordOption::Basic,
                ),
                &Count,
            )
            .unwrap();
        assert_eq!(checkpoints, 1);
        assert_eq!(status(&config).unwrap().indexed_events, 51);
    }

    #[test]
    fn query_groups_split_words_quotes_and_requirements() {
        assert_eq!(
            query_groups(r#" a +b "c d"+"e f" +"g x"#),
            [
                (false, "a"),
                (true, "b"),
                (false, "c d"),
                (true, "e f"),
                (true, "g x")
            ]
        );
        assert_eq!(
            query_groups(r#"a"b"c"#),
            [(false, "a"), (false, "b"), (false, "c")]
        );
        assert_eq!(query_groups("+ a"), [(false, "a")]);
        assert!(query_groups(" + ").is_empty());
    }

    #[test]
    fn code_and_url_queries_search_as_text() {
        let (_dir, config, path) = crate::test_support::history_file("rollout.jsonl");
        input(
            &path,
            "use std::fs::read here; error: see http://example.com (AND NOT)",
        );
        let index = import(&config, Provider::Codex, &path);
        let selection = index
            .selection(&Filters::default(), false, &workspace::Selection::All)
            .unwrap();
        let hits = |query: &str| {
            index
                .lexical(&[query.into()], LexicalMode::All, &*selection)
                .unwrap()
                .rows
                .len()
        };
        for query in [
            "std::fs",
            "read(",
            "error: missing",
            "http://example.com",
            "\"see http\"",
            "\"read here",
            "AND",
            "NOT",
            "fs::read)",
        ] {
            assert_eq!(hits(query), 1, "{query:?}");
        }
        // Multi-token words and quoted spans are phrases, not bags of words.
        for query in ["fs::std", "\"http see\"", "\"unclosed", "", "  :: ()"] {
            assert_eq!(hits(query), 0, "{query:?}");
        }
        // `+` requires a word or phrase; other words stay optional.
        for query in ["+read absent", "absent +\"see http\"", "+ read"] {
            assert_eq!(hits(query), 1, "{query:?}");
        }
        for query in ["+absent read", "read +\"http see\"", "+read +absent"] {
            assert_eq!(hits(query), 0, "{query:?}");
        }
    }

    #[test]
    fn lexical_limit_filters_before_collecting_and_orders_score_ties() {
        let (_dir, config, path) = crate::test_support::history_file("rollout.jsonl");
        let mut input = HEADER.to_owned();
        for n in 0..100 {
            let text = if n < 4 {
                "needle".to_owned()
            } else {
                format!("needle {}", "filler ".repeat(n))
            };
            let row = json!({"type":"response_item", "timestamp":millis_for_test(n as i64), "payload": {
                "id": format!("record-{n}"), "type":"message", "role":"user",
                "content":[{"type":"input_text", "text":text}]
            }});
            input.push_str(&format!("{row}\n"));
        }
        fs::write(&path, input).unwrap();
        let index = import(&config, Provider::Codex, &path);
        let terms = ["needle".into(), "absent".into()];
        for terms in [&terms[..1], &terms[..]] {
            let filters = Filters {
                since: Some(millis_for_test(1)),
                ..Default::default()
            };
            let selection = index
                .selection(&filters, false, &workspace::Selection::All)
                .unwrap();
            index.summary_decodes.set(0);
            let limited = index
                .lexical(terms, LexicalMode::Events(1), &*selection)
                .unwrap()
                .rows;
            // The newest of the equally short matches wins. Long messages
            // have lower relevance despite their newer timestamps.
            assert_eq!(limited.len(), 1);
            assert_eq!(limited[0].1.provider_event_id, "record-3");
            assert_eq!(index.summary_decodes.get(), limited.len());
            assert_eq!(index.record_decodes.get(), 0);
            index.summary_decodes.set(0);
            let grouped = index
                .lexical(terms, LexicalMode::Sessions, &*selection)
                .unwrap()
                .rows;
            assert_eq!(grouped.len(), 1);
            assert_eq!(grouped[0].1.provider_event_id, "record-3");
            assert_eq!(index.summary_decodes.get(), grouped.len());
            assert_eq!(index.record_decodes.get(), 0);
        }
        let selection = index
            .selection(&Filters::default(), false, &workspace::Selection::All)
            .unwrap();
        assert!(
            index
                .lexical(&terms, LexicalMode::Events(0), &*selection)
                .unwrap()
                .rows
                .is_empty()
        );
        let filters = Filters {
            since: Some(millis_for_test(99)),
            ..Default::default()
        };
        let selection = index
            .selection(&filters, false, &workspace::Selection::All)
            .unwrap();
        let last = index
            .lexical(&terms[..1], LexicalMode::Events(1), &*selection)
            .unwrap()
            .rows;
        assert_eq!(
            last.len(),
            1,
            "a low-ranking filtered match must remain reachable"
        );
        assert_eq!(last[0].1.provider_event_id, "record-99");
    }

    fn millis_for_test(seconds: i64) -> String {
        crate::record::millis(1767225600000 + seconds * 1000).unwrap()
    }

    fn hybrid_fallback_before_record_decoding(enabled: bool, cache: Option<&[u8]>, reason: &str) {
        use crate::cli::{Backend, Filters, Refresh, Search};
        let dir = tempfile::tempdir().unwrap();
        let config = Config::edit(dir.path(), |doc| {
            doc["search"]["semantic"] = toml_edit::value(enabled);
            doc["semantic"]["executor"] = toml_edit::value("http://127.0.0.1:1/v1");
            doc["semantic"]["model"] = toml_edit::value("test");
            Ok(())
        })
        .unwrap();
        if let Some(cache) = cache {
            crate::semantic::tests::write_cache(&config, cache);
        }
        let path = dir.path().join("input.jsonl");
        input(&path, "needle");
        let index = import(&config, Provider::Codex, &path);
        let record = index.all().unwrap().remove(0);
        index.record_decodes.set(0);
        index.summary_decodes.set(0);
        let result = crate::query::search_index(
            &config,
            &Search {
                query: "needle".into(),
                term: Vec::new(),
                backend: Backend::Hybrid,
                refresh: Refresh::Off,
                events: true,
                limit: 10,
                filters: Filters {
                    content_scope: crate::record::ContentScope::All,
                    workspace: vec![workspace::Pattern::All],
                    ..Default::default()
                },
            },
            Some(&index),
        )
        .unwrap();
        assert!(result.retrieval.effective_mode == Backend::Lexical);
        assert_eq!(result.results[0].trwl_event_id, record.event_id);
        assert_eq!(index.record_decodes.get(), 0);
        assert_eq!(index.summary_decodes.get(), 1);
        assert!(
            result.retrieval.reason.as_deref().unwrap().contains(reason),
            "fallback must precede record decoding: {:?}",
            result.retrieval.reason
        );
    }

    #[test]
    fn disabled_hybrid_does_not_decode_full_stored_records() {
        hybrid_fallback_before_record_decoding(false, None, "semantic retrieval is disabled");
    }

    #[test]
    fn cached_semantic_refresh_skips_excluded_record_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::edit(dir.path(), |doc| {
            doc["search"]["semantic"] = toml_edit::value(true);
            doc["semantic"]["executor"] = toml_edit::value("http://127.0.0.1:1/v1");
            doc["semantic"]["model"] = toml_edit::value("test");
            Ok(())
        })
        .unwrap();
        let cache = rmp_serde::to_vec(&HashMap::from([(
            "vectors",
            HashMap::from([(digest(&["user:\nneedle"]), vec![1.0f32, 0.0])]),
        )]))
        .unwrap();
        crate::semantic::tests::write_cache(&config, &cache);
        let path = dir.path().join("input.jsonl");
        input(&path, "needle");
        let row = json!({"type":"response_item", "payload":{"type":"reasoning", "summary":[{"text":"excluded"}]}});
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(format!("{row}\n").as_bytes())
            .unwrap();
        let index = import(&config, Provider::Codex, &path);
        assert_eq!(index.all().unwrap().len(), 2);
        index.record_decodes.set(0);
        let messages = index.messages().unwrap();
        assert_eq!(index.record_decodes.get(), 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text, "needle");
        index.record_decodes.set(0);
        crate::semantic::refresh(&config).unwrap();
        assert_eq!(index.record_decodes.get(), 1);
    }

    #[test]
    fn uncached_hybrid_does_not_decode_full_stored_records() {
        hybrid_fallback_before_record_decoding(true, None, "no embeddings");
    }

    #[test]
    fn corrupt_cache_hybrid_does_not_decode_full_stored_records() {
        hybrid_fallback_before_record_decoding(
            true,
            Some(b"invalid cache"),
            "invalid semantic cache",
        );
    }

    #[test]
    fn unavailable_executor_hybrid_does_not_decode_full_stored_records() {
        let cache = rmp_serde::to_vec(&HashMap::from([(
            "vectors",
            HashMap::from([(digest(&["needle"]), vec![1.0f32, 0.0])]),
        )]))
        .unwrap();
        hybrid_fallback_before_record_decoding(true, Some(&cache), "HTTP embedding request failed");
    }

    #[test]
    fn opencode_updates_reparse_only_changed_sessions() {
        let (_dir, config, path) = crate::test_support::history_file("history.db");
        let db = opencode_db(&path);
        db.execute_batch(
            "INSERT INTO session VALUES ('other', '/workspace', NULL), ('gone', '/workspace', NULL);
             INSERT INTO message (id, session_id, data) VALUES
                 ('m1', 'session', '{\"role\":\"user\"}'),
                 ('m2', 'other', '{\"role\":\"user\"}'),
                 ('m3', 'gone', '{\"role\":\"user\"}');",
        )
        .unwrap();
        let part = |id: &str, message: &str, session: &str, text: &str, time: i64| {
            db.execute(
                "INSERT INTO part (id, message_id, session_id, data, time_created) VALUES (?, ?, ?, ?, ?)",
                rusqlite::params![id, message, session, json!({"type":"text", "text":text}).to_string(), time],
            )
            .unwrap();
        };
        part("p1", "m1", "session", "first", 1);
        part("p2", "m2", "other", "second", 2);
        part("p3", "m3", "gone", "third", 3);
        let source = source(Provider::Opencode, &path);
        // Count records each update parses, and compare the index with a full parse.
        let run = || {
            let mut parsed = Vec::new();
            update_with_parser(
                &config,
                std::slice::from_ref(&source),
                false,
                false,
                |provider, path, previous, emit| {
                    providers::parse(provider, path, previous, |record| {
                        parsed.push(record.text.clone());
                        emit(record)
                    })
                },
            )
            .unwrap();
            let mut expected = Vec::new();
            providers::parse(Provider::Opencode, &path, None, |record| {
                expected.push(record.text.clone());
                Ok(())
            })
            .unwrap();
            let mut actual: Vec<_> = records(&config.root)
                .unwrap()
                .into_iter()
                .map(|r| r.text)
                .collect();
            actual.sort();
            expected.sort();
            assert_eq!(actual, expected);
            parsed.sort();
            parsed
        };
        assert_eq!(run(), ["first", "second", "third"]);
        // An unrelated write changes the file stamp but no session.
        db.execute_batch("CREATE TABLE unrelated (x); INSERT INTO unrelated VALUES (1);")
            .unwrap();
        assert!(run().is_empty());
        part("p4", "m1", "session", "appended", 4);
        assert_eq!(run(), ["appended"]);
        db.execute(
            "UPDATE part SET data = ? WHERE id = 'p2'",
            [json!({"type":"text", "text":"edited"}).to_string()],
        )
        .unwrap();
        assert_eq!(run(), ["edited"]);
        db.execute_batch(
            "DELETE FROM part WHERE session_id = 'gone'; DELETE FROM session WHERE id = 'gone';",
        )
        .unwrap();
        assert!(run().is_empty());
        db.execute("DELETE FROM part WHERE id = 'p4'", []).unwrap();
        assert_eq!(run(), ["first"]);
    }

    #[test]
    fn opencode_rewrites_without_timestamp_changes() {
        let (_dir, config, path) = crate::test_support::history_file("history.db");
        let db = opencode_db(&path);
        // Writers below keep `time_updated`, unlike the fixture's trigger.
        db.execute_batch(
            "DROP TRIGGER part_touch;
             INSERT INTO message (id, session_id, data) VALUES ('m', 'session', '{\"role\":\"user\"}');",
        )
        .unwrap();
        let write = |sql: &str, id: &str, text: &str| {
            db.execute(
                sql,
                rusqlite::params![id, json!({"type":"text", "text":text}).to_string()],
            )
            .unwrap();
        };
        let insert = "INSERT INTO part (id, message_id, session_id, data, time_created) VALUES (?1, 'm', 'session', ?2, 1)";
        write(insert, "p1", "one");
        write(insert, "p2", "two");
        let source = source(Provider::Opencode, &path);
        let texts = || {
            let mut texts: Vec<_> = records(&config.root)
                .unwrap()
                .into_iter()
                .map(|r| r.text)
                .collect();
            texts.sort();
            texts
        };
        update(&config, std::slice::from_ref(&source), false).unwrap();
        assert_eq!(texts(), ["one", "two"]);
        // Deleting and re-inserting a row gives it a new rowid.
        db.execute("DELETE FROM part WHERE id = 'p1'", []).unwrap();
        write(insert, "p1", "uno");
        update(&config, std::slice::from_ref(&source), false).unwrap();
        assert_eq!(texts(), ["two", "uno"]);
        // An in-place update keeps every fingerprint component; only a full
        // import reparses it.
        write("UPDATE part SET data = ?2 WHERE id = ?1", "p2", "dos");
        update(&config, std::slice::from_ref(&source), false).unwrap();
        assert_eq!(texts(), ["two", "uno"]);
        assert!(
            update_full(&config, std::slice::from_ref(&source), false)
                .unwrap()
                .completed()
        );
        assert_eq!(texts(), ["dos", "uno"]);
    }

    #[test]
    fn large_lexical_metadata_survives_publication_and_merge() {
        let (_dir, config, path) = crate::test_support::history_file("history.db");
        let db = opencode_db(&path);
        db.execute_batch(
            "INSERT INTO message (id, session_id, data) VALUES ('message', 'session', '{\"role\":\"assistant\"}');",
        )
        .unwrap();
        let source = source(Provider::Opencode, &path);
        let mut files: Vec<_> = (0..2000)
            .map(|i| format!("src/{i}/日本語/{}.rs", "component".repeat(10)))
            .collect();
        // A single oversized value also exercises repeated identical chunks.
        files.push("z".repeat(200_000));
        for n in 0..2 {
            db.execute(
                "INSERT INTO part (id, message_id, session_id, data, time_created) VALUES (?, 'message', 'session', ?, ?)",
                rusqlite::params![
                    format!("part-{n}"),
                    json!({"type":"patch", "hash":"needle", "files": files}).to_string(),
                    n
                ],
            )
            .unwrap();
            update(&config, std::slice::from_ref(&source), false).unwrap();
        }
        let check = || {
            let index = ReadIndex::open(&config.root).unwrap().unwrap();
            let hits = index
                .lexical(
                    &["needle".into()],
                    LexicalMode::All,
                    &*index
                        .selection(&Filters::default(), false, &workspace::Selection::All)
                        .unwrap(),
                )
                .unwrap();
            assert_eq!(hits.rows.len(), 2);
            for (_, hit) in hits.rows {
                let full = index.event_reference(&hit.event_id).unwrap().remove(0);
                assert_eq!(full.files, files);
                for file in [&files[0], files.last().unwrap()] {
                    let filters = Filters {
                        file: Some(file.clone()),
                        ..Default::default()
                    };
                    let selected = index
                        .selected_ids(
                            &*index
                                .selection(&filters, false, &workspace::Selection::All)
                                .unwrap(),
                        )
                        .unwrap();
                    assert!(selected.contains(&hit.event_id));
                }
            }
        };
        check();
        let index = Index::open_in_dir(config.root.join("index")).unwrap();
        let mut writer = index
            .writer_with_num_threads::<TantivyDocument>(1, 15_000_000)
            .unwrap();
        writer
            .merge(&index.searchable_segment_ids().unwrap())
            .wait()
            .unwrap();
        check();
    }

    #[test]
    fn messages_preserve_visible_text_across_commits_and_merges() {
        let (_dir, config, path) = crate::test_support::history_file("history.db");
        let db = opencode_db(&path);
        db.execute_batch(
            "INSERT INTO message (id, session_id, data) VALUES ('user', 'session', '{\"role\":\"user\"}');
             INSERT INTO message (id, session_id, data) VALUES ('assistant', 'session', '{\"role\":\"assistant\"}');",
        )
        .unwrap();
        let text = "a".repeat(200_000) + "最後 🦀";
        for (n, (id, role, part)) in [
            ("user", "user", json!({"type":"text", "text":text})),
            (
                "reasoning",
                "assistant",
                json!({"type":"reasoning", "text":"private reasoning"}),
            ),
            (
                "tool",
                "assistant",
                json!({"type":"tool", "tool":"bash", "state":{"input":{"command":"true"}}}),
            ),
            (
                "answer",
                "assistant",
                json!({"type":"text", "text":"answer"}),
            ),
            ("empty", "user", json!({"type":"text", "text":""})),
        ]
        .into_iter()
        .enumerate()
        {
            db.execute(
                "INSERT INTO part (id, message_id, session_id, data, time_created) VALUES (?, ?, 'session', ?, ?)",
                rusqlite::params![id, role, part.to_string(), n as i64],
            )
            .unwrap();
        }
        let source = source(Provider::Opencode, &path);
        update(&config, std::slice::from_ref(&source), false).unwrap();
        let pinned = ReadIndex::open(&config.root).unwrap().unwrap();
        let messages = pinned.messages().unwrap();
        assert_eq!(messages.len(), 3);
        assert!(messages.iter().any(|message| message.text.is_empty()));
        let user = messages
            .iter()
            .find(|message| {
                message
                    .request
                    .as_ref()
                    .is_some_and(|record| record.provider_event_id == "user")
            })
            .unwrap();
        assert_eq!(user.text, text);
        assert_eq!(
            pinned.event_reference(&user.event_id).unwrap()[0].text,
            text
        );
        assert_eq!(pinned.all().unwrap().len(), 5);
        db.execute(
            "UPDATE part SET data=? WHERE id='user'",
            [json!({"type":"text", "text":"edited"}).to_string()],
        )
        .unwrap();
        db.execute("DELETE FROM part WHERE id IN ('answer', 'empty')", [])
            .unwrap();
        update(&config, std::slice::from_ref(&source), false).unwrap();
        let version = semantic_version(&config.root).unwrap().unwrap();
        assert_ne!(version, pinned.version());
        let check = || {
            let index = ReadIndex::open(&config.root).unwrap().unwrap();
            assert_eq!(index.version(), version);
            let current = index.messages().unwrap();
            assert_eq!(current.len(), 1);
            assert_eq!(current[0].text, "edited");
            assert_eq!(current[0].event_id, user.event_id);
            assert_eq!(pinned.messages().unwrap().len(), 3);
        };
        check();
        update_full(&config, &[source], false).unwrap();
        check();
        let index = Index::open_in_dir(config.root.join("index")).unwrap();
        let mut writer = index
            .writer_with_num_threads::<TantivyDocument>(1, 15_000_000)
            .unwrap();
        writer
            .merge(&index.searchable_segment_ids().unwrap())
            .wait()
            .unwrap();
        check();
    }

    #[test]
    fn sqlite_snapshot_publishes_during_writes_and_catches_up_afterward() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.db");
        let mut db = opencode_db(&path);
        db.execute_batch("PRAGMA journal_mode=WAL;
            INSERT INTO message (id, session_id, data) VALUES ('message', 'session', '{\"role\":\"assistant\"}');").unwrap();
        let tx = db.transaction().unwrap();
        let body = json!({"type":"text", "text":"original"}).to_string();
        for n in 0..2 {
            tx.execute(
                "INSERT INTO part (id, message_id, session_id, data, time_created) VALUES (?, 'message', 'session', ?, ?)",
                rusqlite::params![format!("part-{n}"), body, n],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        let config = Config::load(&dir.path().join("data")).unwrap();
        let source = source(Provider::Opencode, &path);
        update(&config, std::slice::from_ref(&source), false).unwrap();
        db.execute(
            "UPDATE part SET data=?",
            [json!({"type":"text", "text":"busy version 1"}).to_string()],
        )
        .unwrap();
        let mut wrote_during_snapshot = false;
        let result = update_with_parser(
            &config,
            std::slice::from_ref(&source),
            false,
            false,
            |provider, path, previous, emit| {
                providers::parse(provider, path, previous, |record| {
                    if !wrote_during_snapshot {
                        // Commit through another connection while the parser's
                        // read snapshot is open, before parsing can continue.
                        db.execute(
                            "UPDATE part SET data=?",
                            [json!({"type":"text", "text":"busy version 2"}).to_string()],
                        )?;
                        let (frames, checkpointed): (i64, i64) =
                            db.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| {
                                Ok((row.get(1)?, row.get(2)?))
                            })?;
                        assert!(
                            checkpointed < frames,
                            "no commit was held behind the import's SQLite read snapshot"
                        );
                        wrote_during_snapshot = true;
                    }
                    emit(record)
                })
            },
        );
        assert!(wrote_during_snapshot, "changed source was not parsed");
        assert!(result.unwrap().completed());
        let before_catchup = records(&config.root).unwrap();
        assert_eq!(before_catchup.len(), 2);
        let visible = before_catchup
            .iter()
            .find(|r| r.provider_event_id == "part-0")
            .unwrap();
        assert_eq!(visible.text, "busy version 1");
        let last = before_catchup
            .iter()
            .find(|r| r.provider_event_id == "part-1")
            .unwrap();
        assert_eq!(
            last.text, visible.text,
            "one import must not mix source transactions"
        );
        update(&config, &[source], false).unwrap();
        let after = ReadIndex::open(&config.root)
            .unwrap()
            .unwrap()
            .event_reference(&visible.event_id)
            .unwrap();
        assert_eq!(after[0].text, "busy version 2");
    }

    #[test]
    fn sqlite_replacement_and_corrupt_replacement_preserve_committed_history() {
        let (dir, config, path) = crate::test_support::history_file("history.db");
        // Each replacement holds the same session and part, written at a
        // different time.
        let replace = |text: &str, time: i64| {
            let replacement = dir.path().join("replacement.db");
            let db = opencode_db(&replacement);
            db.execute_batch(
                "INSERT INTO message (id, session_id, data) VALUES ('message', 'session', '{\"role\":\"assistant\"}');",
            )
            .unwrap();
            db.execute(
                "INSERT INTO part (id, message_id, session_id, data, time_created, time_updated) VALUES ('part', 'message', 'session', ?, 1, ?)",
                rusqlite::params![json!({"type":"text", "text":text}).to_string(), time],
            )
            .unwrap();
            drop(db);
            fs::rename(replacement, &path).unwrap();
        };
        let source = source(Provider::Opencode, &path);
        replace("first", 1);
        update(&config, std::slice::from_ref(&source), false).unwrap();
        let pinned = ReadIndex::open(&config.root).unwrap().unwrap();
        let id = pinned.all().unwrap()[0].event_id.clone();
        replace("other", 2);
        assert!(
            update(&config, std::slice::from_ref(&source), false)
                .unwrap()
                .completed()
        );
        assert_eq!(records(&config.root).unwrap()[0].text, "other");
        {
            let db = rusqlite::Connection::open(&path).unwrap();
            db.execute(
                "UPDATE part SET data=?",
                [json!({"type":"text", "text":"unpublished replacement"}).to_string()],
            )
            .unwrap();
            db.execute_batch("INSERT INTO message (id, session_id, data) VALUES ('bad-message', 'session', 'invalid JSON');
                INSERT INTO part (id, message_id, session_id, data, time_created) VALUES ('bad-part', 'bad-message', 'session', '{\"type\":\"text\",\"text\":\"bad\"}', 2);").unwrap();
        }
        assert!(
            !update(&config, std::slice::from_ref(&source), false)
                .unwrap()
                .completed()
        );
        assert_eq!(
            records(&config.root).unwrap()[0].text,
            "other",
            "a later parse error must discard earlier spooled records"
        );
        let corrupt = dir.path().join("corrupt.db");
        fs::write(&corrupt, "not a SQLite database").unwrap();
        fs::rename(corrupt, &path).unwrap();
        assert!(
            !update(&config, std::slice::from_ref(&source), false)
                .unwrap()
                .completed()
        );
        assert_eq!(records(&config.root).unwrap()[0].text, "other");
        replace("third", 3);
        update(&config, &[source], false).unwrap();
        let current = records(&config.root).unwrap();
        assert_eq!(current[0].event_id, id);
        assert_eq!(current[0].text, "third");
        assert_eq!(pinned.all().unwrap()[0].text, "first");
    }

    #[test]
    fn pinned_reader_survives_replacement_and_directory_deletion() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&dir.path().join("data")).unwrap();
        let root = dir.path().join("sessions");
        fs::create_dir(&root).unwrap();
        let path = root.join("rollout.jsonl");
        input(&path, "before");
        let source = source(Provider::Codex, &root);
        update(&config, std::slice::from_ref(&source), false).unwrap();
        let pinned = ReadIndex::open(&config.root).unwrap().unwrap();
        input(&path, "after");
        update(&config, std::slice::from_ref(&source), false).unwrap();
        assert_eq!(records(&config.root).unwrap()[0].text, "after");
        assert_eq!(pinned.all().unwrap()[0].text, "before");
        fs::remove_file(path).unwrap();
        update(&config, &[source], false).unwrap();
        assert!(records(&config.root).unwrap().is_empty());
        assert_eq!(pinned.all().unwrap()[0].text, "before");
    }
}
