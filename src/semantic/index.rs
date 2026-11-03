//! SQLite storage for vectors, prepared inputs, document records and chunk mappings.
//!
//! Each executor/model/input-version selection keeps one index-commit mapping.
//! Replacing it removes unreferenced vectors; other selections remain intact.
//!
//! Read-only operation snapshots permit WAL writers to commit during a scan.
//! SQLite manages checkpointing; checkpoint completion is not publication.
//!
//! Writers hold semantic-cache.lock. All operations close their connections
//! before returning; callers never wait for inference with a transaction open.
//! Write failures preserve earlier commits. History remains in Tantivy.
//!
//! MessagePack migration imports only an absent SQL selection, atomically.
//! Its selected source files are removed only after commit.
use super::{Cache, DocumentCache, DocumentPlan, EmbeddingDocument, SelectionIdentity};
use crate::config::{Config, lock};
use crate::store::ConversationRevision;
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OpenFlags, OptionalExtension, Transaction, params};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

const DATABASE: &str = "cache.sqlite3";

const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS selections (
        selection TEXT PRIMARY KEY, version TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS documents (
        selection TEXT NOT NULL REFERENCES selections ON DELETE CASCADE,
        event TEXT NOT NULL, record BLOB NOT NULL, PRIMARY KEY(selection,event));
    CREATE TABLE IF NOT EXISTS chunks (
        selection TEXT NOT NULL, event TEXT NOT NULL,
        ordinal INTEGER NOT NULL, key TEXT NOT NULL,
        PRIMARY KEY(selection,event,ordinal),
        FOREIGN KEY(selection,event) REFERENCES documents ON DELETE CASCADE);
    CREATE INDEX IF NOT EXISTS chunk_keys ON chunks(selection,key);
    CREATE TABLE IF NOT EXISTS vectors (
        selection TEXT NOT NULL REFERENCES selections ON DELETE CASCADE,
        key TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(selection,key));
    CREATE TABLE IF NOT EXISTS builds (
        selection TEXT PRIMARY KEY REFERENCES selections ON DELETE CASCADE,
        complete INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS prepared (
        selection TEXT NOT NULL, event TEXT NOT NULL,
        input TEXT NOT NULL, chunks BLOB NOT NULL, PRIMARY KEY(selection,event),
        FOREIGN KEY(selection,event) REFERENCES documents ON DELETE CASCADE);";

struct Database {
    connection: Connection,
}

#[derive(PartialEq, Eq)]
struct Column {
    name: String,
    kind: String,
    required: bool,
    default: Option<String>,
    primary_key: i64,
}

#[derive(PartialEq, Eq)]
struct ForeignKey {
    table: String,
    from: String,
    to: Option<String>,
    on_update: String,
    on_delete: String,
    matching: String,
}

struct LegacyMigration {
    vectors: Cache,
    documents: Option<DocumentCache>,
    sources: Vec<PathBuf>,
}

struct LockedIndex {
    directory: PathBuf,
    _lock: fs::File,
}

enum Mapping {
    Absent,
    VectorOnly,
    Snapshot(ConversationRevision),
}

impl Mapping {
    fn read(connection: &Connection, selection: &SelectionIdentity) -> Result<Self> {
        let version: Option<String> = connection
            .query_row(
                "SELECT version FROM selections WHERE selection=?",
                [selection.as_str()],
                |r| r.get(0),
            )
            .optional()?;
        Ok(match version {
            None => Self::Absent,
            Some(version) if version.is_empty() => Self::VectorOnly,
            Some(version) => Self::Snapshot(ConversationRevision::from_storage(version)),
        })
    }

    fn matches(&self, version: &ConversationRevision) -> bool {
        match self {
            Self::Absent => false,
            Self::VectorOnly => version.as_str().is_empty(),
            Self::Snapshot(stored) => stored == version,
        }
    }
}

// Build history controls full-build scheduling, not current vector coverage.
pub(super) enum BuildHistory {
    Absent,
    Unfinished,
    Completed,
    LegacyCompleted,
}

impl BuildHistory {
    pub(super) fn needs_full_build(&self) -> bool {
        matches!(self, Self::Absent | Self::Unfinished)
    }
}

impl Database {
    fn open(path: &Path) -> Result<Self> {
        let mut connection = Connection::open(path)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        Self::validate_version(&connection)?;
        connection.execute_batch(
            "PRAGMA foreign_keys=ON;
             PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;",
        )?;
        // The caller holds the application writer lock. Recheck here, rather
        // than trusting the pre-lock reader's initialization decision.
        if Self::validate_version(&connection)? == 0 {
            let tx = connection.transaction()?;
            Self::validate_schema(&tx, true)?;
            let columns = Self::columns(&tx, "selections")?;
            if columns.iter().any(|column| column.name == "touched") {
                tx.execute_batch(
                    "ALTER TABLE selections DROP COLUMN touched;
                                  ALTER TABLE selections DROP COLUMN bytes;",
                )?;
            }
            tx.execute_batch(SCHEMA)?;
            tx.pragma_update(None, "user_version", 1)?;
            tx.commit()?;
        } else {
            Self::validate_schema(&connection, false)?;
        }
        Ok(Self { connection })
    }

    fn validate_version(connection: &Connection) -> Result<i64> {
        let version = connection.pragma_query_value(None, "user_version", |r| r.get(0))?;
        ensure!(
            matches!(version, 0 | 1),
            "unsupported semantic index schema version {version}"
        );
        Ok(version)
    }

    fn columns(connection: &Connection, table: &str) -> Result<Vec<Column>> {
        Ok(connection
            .prepare(&format!("PRAGMA table_info({table})"))?
            .query_map([], |r| {
                Ok(Column {
                    name: r.get(1)?,
                    kind: r.get(2)?,
                    required: r.get(3)?,
                    default: r.get(4)?,
                    primary_key: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    fn validate_schema(connection: &Connection, upgrading: bool) -> Result<()> {
        // Derive expected columns and foreign keys from the one current-schema
        // declaration. Never interpret an unknown layout as an empty index.
        let expected = Connection::open_in_memory()?;
        expected.execute_batch(SCHEMA)?;
        let mut recognized = false;
        for table in [
            "selections",
            "documents",
            "chunks",
            "vectors",
            "builds",
            "prepared",
        ] {
            let mut actual = Self::columns(connection, table)?;
            if actual.is_empty() && upgrading {
                continue;
            }
            recognized = true;
            if upgrading && table == "selections" {
                let old = actual.split_off(actual.len().min(2));
                ensure!(
                    old == vec![
                        Column {
                            name: "touched".into(),
                            kind: "INTEGER".into(),
                            required: true,
                            default: None,
                            primary_key: 0
                        },
                        Column {
                            name: "bytes".into(),
                            kind: "INTEGER".into(),
                            required: true,
                            default: Some("0".into()),
                            primary_key: 0
                        },
                    ],
                    "unrecognized semantic index schema: {table}"
                );
            }
            ensure!(
                actual == Self::columns(&expected, table)?,
                "unrecognized semantic index schema: {table}"
            );
            let foreign_keys = |db: &Connection| -> Result<Vec<ForeignKey>> {
                Ok(db
                    .prepare(&format!("PRAGMA foreign_key_list({table})"))?
                    .query_map([], |r| {
                        Ok(ForeignKey {
                            table: r.get(2)?,
                            from: r.get(3)?,
                            to: r.get(4)?,
                            on_update: r.get(5)?,
                            on_delete: r.get(6)?,
                            matching: r.get(7)?,
                        })
                    })?
                    .collect::<rusqlite::Result<_>>()?)
            };
            ensure!(
                foreign_keys(connection)? == foreign_keys(&expected)?,
                "unrecognized semantic index foreign keys: {table}"
            );
        }
        if upgrading && !recognized {
            let tables: i64 = connection.query_row("SELECT count(*) FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'", [], |r| r.get(0))?;
            ensure!(tables == 0, "unrecognized semantic index schema");
        }
        Ok(())
    }

    fn read(path: &Path) -> Result<Self> {
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        Ok(Self { connection })
    }

    fn begin_write(&mut self) -> Result<Transaction<'_>> {
        Ok(self.connection.transaction()?)
    }

    fn scan_vectors(
        &self,
        selection: &SelectionIdentity,
        mut consume: impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let mut query = self
            .connection
            .prepare("SELECT key,vector FROM vectors WHERE selection=?")?;
        let mut rows = query.query([selection.as_str()])?;
        while let Some(row) = rows.next()? {
            let key = row.get_ref(0)?.as_str()?;
            let bytes = row.get_ref(1)?.as_blob()?;
            ensure!(
                bytes.len().is_multiple_of(4),
                "invalid semantic index embedding length"
            );
            let vector: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            consume(key, &vector)?;
        }
        Ok(())
    }

    fn documents(
        &mut self,
        selection: &SelectionIdentity,
        version: &ConversationRevision,
    ) -> Result<Option<Vec<EmbeddingDocument>>> {
        let tx = &self.connection;
        if !Mapping::read(tx, selection)?.matches(version) {
            return Ok(None);
        }
        let selection = selection.as_str();
        let mut documents = Vec::new();
        {
            let mut query =
                tx.prepare("SELECT event,record FROM documents WHERE selection=? ORDER BY event")?;
            let mut keys = tx
                .prepare("SELECT key FROM chunks WHERE selection=? AND event=? ORDER BY ordinal")?;
            let mut prepared_inputs =
                tx.prepare("SELECT input,chunks FROM prepared WHERE selection=? AND event=?")?;
            let mut rows = query.query([selection])?;
            while let Some(row) = rows.next()? {
                let event: String = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                documents.push(EmbeddingDocument {
                    record: rmp_serde::from_slice(&bytes)
                        .context("invalid semantic index document")?,
                    keys: keys
                        .query_map(params![selection, event], |r| r.get(0))?
                        .collect::<rusqlite::Result<Vec<_>>>()?,
                    prepared: prepared_inputs
                        .query_row(params![selection, event], |r| {
                            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
                        })
                        .optional()?
                        .map(|(input, chunks)| -> Result<_> {
                            Ok(super::PreparedDocument {
                                input,
                                chunks: rmp_serde::from_slice(&chunks)?,
                            })
                        })
                        .transpose()?,
                });
            }
        }
        Ok(Some(documents))
    }

    fn publish(
        &mut self,
        selection: &SelectionIdentity,
        version: &ConversationRevision,
        documents: Option<&[EmbeddingDocument]>,
        vectors: &Cache,
    ) -> Result<()> {
        let tx = self.begin_write()?;
        let selection = selection.as_str();
        let version = version.as_str();
        let inserted = tx.execute(
            "INSERT OR IGNORE INTO selections(selection,version) VALUES (?,?)",
            params![selection, version],
        )?;
        if inserted > 0 {
            tx.execute("INSERT INTO builds VALUES (?,0)", [selection])?;
        }
        tx.execute(
            "UPDATE selections SET version=?2 WHERE selection=?1 AND version!=?2",
            params![selection, version],
        )?;
        if let Some(documents) = documents {
            // Keep unchanged rows in place, including their prepared chunks.
            // The current pinned snapshot still decides metadata and deletions.
            let mut stale: std::collections::HashSet<String> = tx
                .prepare("SELECT event FROM documents WHERE selection=?")?
                .query_map([selection], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            let mut insert = tx.prepare(
                "INSERT INTO documents VALUES (?,?,?)
                ON CONFLICT(selection,event) DO UPDATE SET record=excluded.record
                WHERE documents.record!=excluded.record",
            )?;
            let mut chunk = tx.prepare("INSERT INTO chunks VALUES (?,?,?,?)")?;
            let mut old_keys = tx
                .prepare("SELECT key FROM chunks WHERE selection=? AND event=? ORDER BY ordinal")?;
            let mut old_input =
                tx.prepare("SELECT input FROM prepared WHERE selection=? AND event=?")?;
            for document in documents {
                stale.remove(&document.record.event_id);
                insert.execute(params![
                    selection,
                    document.record.event_id,
                    rmp_serde::to_vec_named(&document.record)?
                ])?;
                let previous = old_keys
                    .query_map(params![selection, document.record.event_id], |r| r.get(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                if previous != document.keys {
                    tx.execute(
                        "DELETE FROM chunks WHERE selection=? AND event=?",
                        params![selection, document.record.event_id],
                    )?;
                    for (ordinal, key) in document.keys.iter().enumerate() {
                        chunk.execute(params![
                            selection,
                            document.record.event_id,
                            ordinal as i64,
                            key
                        ])?;
                    }
                }
                if let Some(prepared) = &document.prepared {
                    let previous: Option<String> = old_input
                        .query_row(params![selection, document.record.event_id], |r| r.get(0))
                        .optional()?;
                    if previous.as_ref() != Some(&prepared.input) {
                        tx.execute("INSERT INTO prepared VALUES (?,?,?,?)
                            ON CONFLICT(selection,event) DO UPDATE SET input=excluded.input,chunks=excluded.chunks",
                            params![selection, document.record.event_id, prepared.input, rmp_serde::to_vec(&prepared.chunks)?])?;
                    }
                } else {
                    tx.execute(
                        "DELETE FROM prepared WHERE selection=? AND event=?",
                        params![selection, document.record.event_id],
                    )?;
                }
            }
            for event in stale {
                tx.execute(
                    "DELETE FROM documents WHERE selection=? AND event=?",
                    params![selection, event],
                )?;
            }
        }
        Self::insert_vectors(&tx, selection, vectors)?;
        // A vector-only legacy cache has no mapping yet. Its first mapping
        // publication removes unused keys just like any subsequent replacement.
        if !version.is_empty() {
            tx.execute("DELETE FROM vectors WHERE selection=? AND NOT EXISTS
                (SELECT 1 FROM chunks WHERE chunks.selection=vectors.selection AND chunks.key=vectors.key)", [selection])?;
        }
        tx.commit()?;
        Ok(())
    }

    fn insert_vectors(tx: &Transaction<'_>, selection: &str, vectors: &Cache) -> Result<()> {
        // Keys identify model and input content; a stored vector is reusable.
        let mut insert = tx.prepare("INSERT OR IGNORE INTO vectors VALUES (?,?,?)")?;
        let mut exists = tx.prepare("SELECT 1 FROM vectors WHERE selection=? AND key=?")?;
        for (key, vector) in &vectors.vectors {
            // Immutable content keys need no rewrite or serialization on a hit.
            if exists.exists(params![selection, key])? {
                continue;
            }
            let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
            insert.execute(params![selection, key, bytes])?;
        }
        Ok(())
    }
}

fn remove(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(e).with_context(|| format!("removing {}", path.display()))
        }
        _ => Ok(()),
    }
}

// Only the exact selected paths belong to this conversion. SQL existence is
// checked under the writer lock before decoding either file.
impl LegacyMigration {
    fn read(directory: &Path, selection: &SelectionIdentity) -> Result<Option<Self>> {
        let selection = selection.as_str();
        let mut sources = Vec::new();
        let mut read = |path: PathBuf| -> Result<Option<Vec<u8>>> {
            match fs::read(&path) {
                Ok(bytes) => {
                    sources.push(path);
                    Ok(Some(bytes))
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(error) => Err(error)
                    .with_context(|| format!("reading legacy semantic index {}", path.display())),
            }
        };
        let vector_path = directory.join(format!("{selection}.msgpack"));
        let vectors = read(vector_path.clone())?
            .map(|bytes| {
                rmp_serde::from_slice(&bytes).with_context(|| {
                    format!("invalid legacy semantic vectors {}", vector_path.display())
                })
            })
            .transpose()?
            .unwrap_or_default();
        let document_path = directory.join(format!("documents-{selection}.msgpack"));
        let documents = read(document_path.clone())?
            .map(|bytes| {
                rmp_serde::from_slice(&bytes).with_context(|| {
                    format!(
                        "invalid legacy semantic documents {}",
                        document_path.display()
                    )
                })
            })
            .transpose()?;
        Ok((!sources.is_empty()).then_some(Self {
            vectors,
            documents,
            sources,
        }))
    }

    fn publish(self, db: &mut Database, selection: &SelectionIdentity) -> Result<()> {
        let vector_only = ConversationRevision::from_storage(String::new());
        db.publish(
            selection,
            self.documents.as_ref().map_or(&vector_only, |d| &d.version),
            self.documents.as_ref().map(|d| d.documents.as_slice()),
            &self.vectors,
        )?;
        for path in self.sources {
            if let Err(error) = remove(&path) {
                eprintln!(
                    "warning: legacy semantic index import committed; cleanup failed for {}: {error:#}",
                    path.display()
                );
            }
        }
        Ok(())
    }
}

impl LockedIndex {
    fn acquire(config: &Config) -> Result<Self> {
        let lock = lock(&config.directories.cache, "semantic-cache.lock")?;
        let directory = config.directories.cache.join("semantic");
        fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            _lock: lock,
        })
    }
}

fn with_database<T>(
    config: &Config,
    selection: &SelectionIdentity,
    operation: impl FnOnce(&mut Database) -> Result<T>,
) -> Result<T> {
    with_writer(config, |db, directory| {
        import_legacy(db, directory, selection)?;
        operation(db)
    })
}

fn import_legacy(db: &mut Database, directory: &Path, selection: &SelectionIdentity) -> Result<()> {
    if matches!(Mapping::read(&db.connection, selection)?, Mapping::Absent)
        && let Some(migration) = LegacyMigration::read(directory, selection)?
    {
        migration.publish(db, selection)?;
    }
    Ok(())
}

fn with_writer<T>(
    config: &Config,
    operation: impl FnOnce(&mut Database, &Path) -> Result<T>,
) -> Result<T> {
    let index = LockedIndex::acquire(config)?;
    let path = index.directory.join(DATABASE);
    (|| {
        let mut db = Database::open(&path)?;
        operation(&mut db, &index.directory)
    })()
    .with_context(|| format!("writing semantic index {}", path.display()))
}

fn with_read<T>(
    config: &Config,
    selection: &SelectionIdentity,
    operation: impl FnOnce(&mut Database) -> Result<T>,
) -> Result<T> {
    let directory = config.directories.cache.join("semantic");
    let path = directory.join(DATABASE);
    // Initialization and conversion are completed before the operation snapshot.
    // No application writer lock is acquired by an initialized SQL read.
    let ready = if path.exists() {
        let db = Database::read(&path)
            .with_context(|| format!("opening semantic index {}", path.display()))?;
        (|| -> Result<bool> {
            if Database::validate_version(&db.connection)? != 1 {
                return Ok(false);
            }
            Database::validate_schema(&db.connection, false)?;
            Ok(
                !matches!(Mapping::read(&db.connection, selection)?, Mapping::Absent)
                    || (!directory
                        .join(format!("{}.msgpack", selection.as_str()))
                        .try_exists()?
                        && !directory
                            .join(format!("documents-{}.msgpack", selection.as_str()))
                            .try_exists()?),
            )
        })()
        .with_context(|| format!("opening semantic index {}", path.display()))?
    } else {
        false
    };
    if !ready {
        with_database(config, selection, |_| Ok(()))?;
    }
    read_snapshot(&path, operation)
}

fn read_snapshot<T>(path: &Path, operation: impl FnOnce(&mut Database) -> Result<T>) -> Result<T> {
    (|| {
        let mut db = Database::read(path)?;
        db.connection.execute_batch("BEGIN")?;
        ensure!(
            Database::validate_version(&db.connection)? == 1,
            "semantic index requires initialization"
        );
        Database::validate_schema(&db.connection, false)?;
        operation(&mut db)
        // Dropping the connection ends this read transaction on every exit.
    })()
    .with_context(|| format!("reading semantic index {}", path.display()))
}

pub(super) fn has_vectors(config: &Config, selection: &SelectionIdentity) -> Result<bool> {
    with_read(config, selection, |db| {
        Ok(db.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM vectors WHERE selection=?)",
            [selection.as_str()],
            |row| row.get(0),
        )?)
    })
}

// Consumers update owned accounting only. Inference, locks and writes must wait
// until this operation returns and its read snapshot has closed.
pub(super) fn scan_vectors(
    config: &Config,
    selection: &SelectionIdentity,
    consume: impl FnMut(&str, &[f32]) -> Result<()>,
) -> Result<()> {
    with_read(config, selection, |db| db.scan_vectors(selection, consume))
}

pub(super) fn scan_keys(
    config: &Config,
    selection: &SelectionIdentity,
    mut consume: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    with_read(config, selection, |db| {
        let mut query = db
            .connection
            .prepare("SELECT key FROM vectors WHERE selection=?")?;
        let mut rows = query.query([selection.as_str()])?;
        while let Some(row) = rows.next()? {
            consume(row.get_ref(0)?.as_str()?)?;
        }
        Ok(())
    })
}

pub(super) fn documents(
    config: &Config,
    selection: &SelectionIdentity,
    version: &ConversationRevision,
) -> Result<Option<Vec<EmbeddingDocument>>> {
    with_read(config, selection, |db| db.documents(selection, version))
}

pub(super) fn covered(
    config: &Config,
    selection: &SelectionIdentity,
    version: &ConversationRevision,
) -> Result<bool> {
    let path = config.directories.cache.join("semantic").join(DATABASE);
    if !path.exists() {
        return Ok(false);
    }
    // Scheduling never imports MessagePack or creates a store on an absent path.
    let version_number = {
        let db = Database::read(&path)
            .with_context(|| format!("opening semantic index {}", path.display()))?;
        Database::validate_version(&db.connection)
            .with_context(|| format!("opening semantic index {}", path.display()))?
    };
    if version_number == 0 {
        with_writer(config, |_, _| Ok(()))?;
    }
    read_snapshot(&path, |db| {
        if !Mapping::read(&db.connection, selection)?.matches(version) {
            return Ok(false);
        }
        let missing: bool = db.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM chunks c WHERE c.selection=? AND NOT EXISTS \
             (SELECT 1 FROM vectors v WHERE v.selection=c.selection AND v.key=c.key))",
            [selection.as_str()],
            |row| row.get(0),
        )?;
        Ok(!missing)
    })
}

pub(super) fn prepared(
    config: &Config,
    selection: &SelectionIdentity,
) -> Result<HashMap<String, super::PreparedDocument>> {
    with_read(config, selection, |db| {
        let mut query = db
            .connection
            .prepare("SELECT input,chunks FROM prepared WHERE selection=?")?;
        let mut rows = query.query([selection.as_str()])?;
        let mut prepared = HashMap::new();
        while let Some(row) = rows.next()? {
            let input: String = row.get(0)?;
            if !prepared.contains_key(&input) {
                let bytes: Vec<u8> = row.get(1)?;
                prepared.insert(
                    input.clone(),
                    super::PreparedDocument {
                        input,
                        chunks: rmp_serde::from_slice(&bytes)
                            .context("invalid prepared document")?,
                    },
                );
            }
        }
        Ok(prepared)
    })
}

pub(super) fn publish_mapping(config: &Config, plan: &DocumentPlan<'_>) -> Result<()> {
    let selection = &plan.selection.key;
    with_database(config, selection, |db| {
        db.publish(
            selection,
            &plan.version,
            Some(&plan.documents),
            &Cache::default(),
        )
    })
}

pub(super) fn build_history(
    config: &Config,
    selection: &SelectionIdentity,
) -> Result<BuildHistory> {
    with_read(config, selection, |db| {
        let complete: Option<Option<bool>> = db
            .connection
            .query_row(
                "SELECT builds.complete FROM selections
                 LEFT JOIN builds USING(selection) WHERE selection=?",
                [selection.as_str()],
                |r| r.get(0),
            )
            .optional()?;
        Ok(match complete {
            None => BuildHistory::Absent,
            Some(None) => BuildHistory::LegacyCompleted,
            Some(Some(false)) => BuildHistory::Unfinished,
            Some(Some(true)) => BuildHistory::Completed,
        })
    })
}

pub(super) fn mark_build_complete(config: &Config, plan: &DocumentPlan<'_>) -> Result<()> {
    let selection = &plan.selection.key;
    with_writer(config, |db, _| {
        let tx = db.begin_write()?;
        ensure!(
            Mapping::read(&tx, selection)?.matches(&plan.version),
            "semantic index mapping changed before completion"
        );
        let mut exists = tx.prepare("SELECT 1 FROM vectors WHERE selection=? AND key=?")?;
        for document in &plan.documents {
            for key in &document.keys {
                ensure!(
                    exists.exists(params![selection.as_str(), key])?,
                    "semantic index missing required embedding {key} before completion"
                );
            }
        }
        drop(exists);
        // History still controls full-build scheduling, not query readiness.
        tx.execute(
            "UPDATE builds SET complete=1 WHERE selection=? AND complete=0",
            [selection.as_str()],
        )?;
        tx.commit()?;
        Ok(())
    })
}

pub(super) fn begin_rebuild(config: &Config, selection: &SelectionIdentity) -> Result<()> {
    // Bypass legacy decoding even before history/plan reads. An unfinished SQL
    // build resumes; a replacement row suppresses replay across crashes/retries.
    with_writer(config, |db, _| {
        let tx = db.begin_write()?;
        let unfinished: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM builds WHERE selection=? AND complete=0)",
            [selection.as_str()],
            |r| r.get(0),
        )?;
        if unfinished {
            return Ok(());
        }
        tx.execute(
            "DELETE FROM selections WHERE selection=?",
            [selection.as_str()],
        )?;
        tx.execute("INSERT INTO selections VALUES (?,'')", [selection.as_str()])?;
        tx.execute("INSERT INTO builds VALUES (?,0)", [selection.as_str()])?;
        tx.commit()?;
        Ok(())
    })
}

pub(super) fn publish_batch(
    config: &Config,
    plan: &DocumentPlan<'_>,
    vectors: &Cache,
) -> Result<()> {
    with_writer(config, |db, _| {
        let tx = db.begin_write()?;
        let selection = plan.selection.key.as_str();
        ensure!(
            Mapping::read(&tx, &plan.selection.key)?.matches(&plan.version),
            "semantic index mapping changed before batch publication"
        );
        let mut required = tx.prepare("SELECT 1 FROM chunks WHERE selection=? AND key=?")?;
        for key in vectors.vectors.keys() {
            ensure!(
                required.exists(params![selection, key])?,
                "embedding {key} is not in the published semantic mapping"
            );
        }
        drop(required);
        Database::insert_vectors(&tx, selection, vectors)?;
        tx.commit()?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(config: &Config, selection: &SelectionIdentity) -> Result<Cache> {
        let mut vectors = HashMap::new();
        scan_vectors(config, selection, |key, vector| {
            vectors.insert(key.to_owned(), vector.to_vec());
            Ok(())
        })?;
        Ok(Cache { vectors })
    }
    use crate::record::{EventType, Provider, Record};

    fn selection(value: &str) -> SelectionIdentity {
        SelectionIdentity(value.to_owned())
    }

    fn revision(value: &str) -> ConversationRevision {
        ConversationRevision::from_storage(value.to_owned())
    }

    #[test]
    fn opening_an_initialized_index_does_not_write_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DATABASE);
        drop(Database::open(&path).unwrap());
        // Separate the timestamps even on filesystems with coarse resolution.
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::UNIX_EPOCH)
            .unwrap();
        let before = fs::metadata(&path).unwrap().modified().unwrap();
        drop(Database::open(&path).unwrap());
        assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), before);
    }

    fn fixture(key: &str, dimensions: usize) -> (Vec<EmbeddingDocument>, Cache) {
        let record = Record::new(
            Provider::Codex,
            "session",
            key,
            EventType::Message,
            Path::new("history"),
        );
        (
            vec![EmbeddingDocument {
                record,
                keys: vec![key.into()],
                prepared: None,
            }],
            Cache {
                vectors: [(key.into(), vec![1.0; dimensions])].into(),
            },
        )
    }

    struct SnapshotFixture {
        selection: super::super::Selection,
        index: crate::store::ReadIndex,
    }

    impl SnapshotFixture {
        fn new(config: &Config, label: &str) -> Self {
            let path = config.directories.cache.join("history.jsonl");
            fs::create_dir_all(&config.directories.cache).unwrap();
            let message = crate::test_support::codex_message("user", label);
            fs::write(
                &path,
                format!(
                    "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"snapshot\"}}}}\n{message}\n"
                ),
            )
            .unwrap();
            Self {
                selection: super::super::Selection {
                    key: SelectionIdentity("model".into()),
                    executor: "http".into(),
                    model: "test".into(),
                    directory: None,
                },
                index: crate::test_support::import(config, Provider::Codex, &path),
            }
        }

        fn plan(&self, documents: Vec<EmbeddingDocument>) -> DocumentPlan<'_> {
            DocumentPlan {
                selection: &self.selection,
                index: &self.index,
                version: self.index.version(),
                documents,
                materialization: super::super::Materialization::Deferred,
            }
        }
    }

    fn seed(
        config: &Config,
        selection: &str,
        version: &str,
        documents: &[EmbeddingDocument],
        vectors: &Cache,
    ) {
        let selection = SelectionIdentity(selection.to_owned());
        let version = ConversationRevision::from_storage(version.to_owned());
        with_database(config, &selection, |db| {
            db.publish(&selection, &version, Some(documents), vectors)
        })
        .unwrap();
    }

    #[test]
    fn finishing_a_completed_build_does_not_update_it() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let (documents, vectors) = fixture("first", 2);
        let snapshot = SnapshotFixture::new(&config, "first");
        let plan = snapshot.plan(documents);
        publish_mapping(&config, &plan).unwrap();
        publish_batch(&config, &plan, &vectors).unwrap();
        mark_build_complete(&config, &plan).unwrap();
        let observer =
            Connection::open(config.directories.cache.join("semantic").join(DATABASE)).unwrap();
        observer
            .execute_batch(
                "CREATE TRIGGER no_completed_build_updates BEFORE UPDATE ON builds
            WHEN old.complete=1 BEGIN SELECT RAISE(ABORT, 'updated a completed build'); END;",
            )
            .unwrap();
        mark_build_complete(&config, &plan).unwrap();
        assert!(matches!(
            build_history(&config, &selection("model")).unwrap(),
            BuildHistory::Completed
        ));
    }

    #[test]
    fn operation_snapshot_allows_writes_without_observing_half_a_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let (mut first, mut vectors) = fixture("first", 2);
        first[0].keys.push("tail".into());
        vectors.vectors.insert("tail".into(), vec![0.0, 1.0]);
        seed(&config, "model", "old", &first, &vectors);
        let mut observed = HashMap::new();
        scan_vectors(&config, &selection("model"), |key, vector| {
            if observed.is_empty() {
                // Test-only pause after the scan has established its snapshot.
                let writer_config = config.clone();
                let (send, receive) = std::sync::mpsc::channel();
                let writer = std::thread::spawn(move || {
                    let (next, vectors) = fixture("next", 2);
                    seed(&writer_config, "model", "new", &next, &vectors);
                    seed(&writer_config, "another", "new", &next, &vectors);
                    send.send(()).unwrap();
                });
                receive.recv_timeout(Duration::from_secs(5)).unwrap();
                writer.join().unwrap();
            }
            observed.insert(key.to_owned(), vector.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(observed, vectors.vectors);
        assert!(
            documents(&config, &selection("model"), &revision("old"))
                .unwrap()
                .is_none()
        );
        let mut later = Vec::new();
        scan_vectors(&config, &selection("model"), |key, _| {
            later.push(key.to_owned());
            Ok(())
        })
        .unwrap();
        assert_eq!(later, ["next"]);
    }

    #[test]
    fn batches_and_completion_require_the_published_mapping_and_persistent_keys() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let (documents, vectors) = fixture("first", 2);
        let snapshot = SnapshotFixture::new(&config, "first");
        let plan = snapshot.plan(documents);
        let directory = config.directories.cache.join("semantic");
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("documents-model.msgpack"),
            rmp_serde::to_vec_named(&DocumentCache {
                version: plan.version.clone(),
                documents: fixture("first", 2).0,
            })
            .unwrap(),
        )
        .unwrap();
        fs::write(
            directory.join("model.msgpack"),
            rmp_serde::to_vec(&vectors).unwrap(),
        )
        .unwrap();
        assert!(mark_build_complete(&config, &plan).is_err());
        assert!(publish_batch(&config, &plan, &vectors).is_err());
        assert!(directory.join("model.msgpack").exists());
        assert!(directory.join("documents-model.msgpack").exists());
        fs::remove_file(directory.join("model.msgpack")).unwrap();
        fs::remove_file(directory.join("documents-model.msgpack")).unwrap();
        publish_mapping(&config, &plan).unwrap();
        assert!(mark_build_complete(&config, &plan).is_err());
        assert!(matches!(
            build_history(&config, &selection("model")).unwrap(),
            BuildHistory::Unfinished
        ));
        publish_batch(&config, &plan, &vectors).unwrap();
        let other = SnapshotFixture::new(&config, "next");
        let (documents, other_vectors) = fixture("next", 2);
        let other_plan = other.plan(documents);
        publish_mapping(&config, &other_plan).unwrap();
        assert!(publish_batch(&config, &plan, &vectors).is_err());
        assert!(mark_build_complete(&config, &plan).is_err());
        assert!(
            load(&config, &selection("model"))
                .unwrap()
                .vectors
                .is_empty()
        );
        publish_batch(&config, &other_plan, &other_vectors).unwrap();
        mark_build_complete(&config, &other_plan).unwrap();
        assert!(covered(&config, &selection("model"), &other_plan.version).unwrap());
    }

    #[test]
    fn concurrent_first_access_rechecks_initialization() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for n in 0..4 {
                let config = &config;
                let barrier = &barrier;
                scope.spawn(move || {
                    let (documents, vectors) = fixture("first", 2);
                    barrier.wait();
                    seed(config, &format!("model{n}"), "first", &documents, &vectors);
                });
            }
        });
        for n in 0..4 {
            assert_eq!(
                load(&config, &selection(&format!("model{n}")))
                    .unwrap()
                    .vectors
                    .len(),
                1
            );
        }
    }

    #[test]
    fn publishing_a_snapshot_does_not_rewrite_existing_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let (documents, vectors) = fixture("first", 2);
        let snapshot = SnapshotFixture::new(&config, "first");
        let plan = snapshot.plan(documents);
        publish_mapping(&config, &plan).unwrap();
        publish_batch(&config, &plan, &vectors).unwrap();
        let observer =
            Connection::open(config.directories.cache.join("semantic").join(DATABASE)).unwrap();
        observer
            .execute_batch(
                "CREATE TRIGGER no_vector_rewrites AFTER INSERT ON vectors
            BEGIN SELECT RAISE(ABORT, 'rewrote a completed vector'); END;",
            )
            .unwrap();
        publish_batch(&config, &plan, &vectors).unwrap();
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors,
            vectors.vectors
        );
    }

    #[test]
    fn full_build_state_survives_partial_batches_and_resets_per_selection() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let (mut documents, first) = fixture("first", 2);
        let (second_documents, second) = fixture("second", 2);
        documents.extend(second_documents);
        let snapshot = SnapshotFixture::new(&config, "first");
        let plan = snapshot.plan(documents);
        assert!(matches!(
            build_history(&config, &selection("model")).unwrap(),
            BuildHistory::Absent
        ));
        publish_mapping(&config, &plan).unwrap();
        publish_batch(&config, &plan, &first).unwrap();
        assert!(matches!(
            build_history(&config, &selection("model")).unwrap(),
            BuildHistory::Unfinished
        ));
        assert_eq!(load(&config, &selection("model")).unwrap().vectors.len(), 1);
        publish_batch(&config, &plan, &second).unwrap();
        mark_build_complete(&config, &plan).unwrap();
        assert!(
            !build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
        assert_eq!(load(&config, &selection("model")).unwrap().vectors.len(), 2);
        assert!(
            build_history(&config, &selection("changed-model"))
                .unwrap()
                .needs_full_build()
        );
        // An ordinary index update does not make the selection a full build.
        let next = SnapshotFixture::new(&config, "next");
        publish_mapping(&config, &next.plan(plan.documents)).unwrap();
        assert!(
            !build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
        begin_rebuild(&config, &selection("model")).unwrap();
        assert!(
            build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
        assert!(
            load(&config, &selection("model"))
                .unwrap()
                .vectors
                .is_empty()
        );
    }

    #[test]
    fn an_existing_selection_without_a_build_is_incremental() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let (documents, vectors) = fixture("first", 2);
        let snapshot = SnapshotFixture::new(&config, "first");
        let plan = snapshot.plan(documents);
        publish_mapping(&config, &plan).unwrap();
        publish_batch(&config, &plan, &vectors).unwrap();
        with_database(&config, &selection("model"), |db| {
            db.connection.execute("DELETE FROM builds", [])?;
            Ok(())
        })
        .unwrap();
        assert!(matches!(
            build_history(&config, &selection("model")).unwrap(),
            BuildHistory::LegacyCompleted
        ));
        let next = SnapshotFixture::new(&config, "next");
        let plan = next.plan(plan.documents);
        publish_mapping(&config, &plan).unwrap();
        publish_batch(&config, &plan, &vectors).unwrap();
        assert!(
            !build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
    }

    #[test]
    fn replacement_drops_old_versions_and_unreferenced_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(&dir.path().join(DATABASE)).unwrap();
        let (first, vectors) = fixture("first", 2);
        db.publish(
            &selection("model"),
            &revision("commit1"),
            Some(&first),
            &vectors,
        )
        .unwrap();
        let (next, vectors) = fixture("next", 2);
        db.publish(
            &selection("model"),
            &revision("commit2"),
            Some(&next),
            &vectors,
        )
        .unwrap();
        assert!(
            db.documents(&selection("model"), &revision("commit1"))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            db.documents(&selection("model"), &revision("commit2"))
                .unwrap()
                .unwrap()[0]
                .keys,
            ["next"]
        );
        let mut vectors = HashMap::new();
        db.scan_vectors(&selection("model"), |key, vector| {
            vectors.insert(key.to_owned(), vector.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(vectors.len(), 1);
        assert!(vectors.contains_key("next"));
    }

    #[test]
    fn sqlite_full_preserves_commits_and_next_publication_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let history = dir.path().join("history.jsonl");
        let message = crate::test_support::codex_message("user", "cache recovery history");
        fs::write(
            &history,
            format!(
                "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"session\"}}}}\n{message}\n"
            ),
        )
        .unwrap();
        let index = crate::test_support::import(&config, Provider::Codex, &history);
        let event = index.all().unwrap().remove(0);
        let (documents, vectors) = fixture("first", 2);
        seed(&config, "model", "old", &documents, &vectors);
        seed(&config, "other", "old", &documents, &vectors);
        with_database(&config, &selection("other"), |db| {
            db.connection
                .execute("UPDATE builds SET complete=1 WHERE selection='other'", [])?;
            Ok(())
        })
        .unwrap();
        assert!(matches!(
            build_history(&config, &selection("model")).unwrap(),
            BuildHistory::Unfinished
        ));
        assert!(matches!(
            build_history(&config, &selection("other")).unwrap(),
            BuildHistory::Completed
        ));
        let error = with_database(&config, &selection("model"), |db| {
            let pages: i64 = db
                .connection
                .query_row("PRAGMA page_count", [], |r| r.get(0))?;
            db.connection.pragma_update(None, "max_page_count", pages)?;
            let (large, vectors) = fixture("huge", 100_000);
            db.publish(
                &selection("model"),
                &revision("new"),
                Some(&large),
                &vectors,
            )
        })
        .unwrap_err();
        assert!(error.chain().any(|cause| matches!(cause.downcast_ref::<rusqlite::Error>(), Some(rusqlite::Error::SqliteFailure(code, _)) if code.code == rusqlite::ErrorCode::DiskFull)), "{error:#}");
        assert!(format!("{error:#}").contains("cache.sqlite3"), "{error:#}");
        let stored = crate::query::resolve(
            &config.directories.data,
            crate::cli::Entity::Event,
            &event.event_id,
        )
        .unwrap();
        assert_eq!(stored.text, event.text);
        assert!(matches!(
            build_history(&config, &selection("model")).unwrap(),
            BuildHistory::Unfinished
        ));
        assert!(matches!(
            build_history(&config, &selection("other")).unwrap(),
            BuildHistory::Completed
        ));
        for model in ["model", "other"] {
            assert_eq!(
                load(&config, &selection(model)).unwrap().vectors,
                vectors.vectors
            );
            assert!(
                super::documents(&config, &selection(model), &revision("old"))
                    .unwrap()
                    .is_some()
            );
        }
        seed(&config, "model", "recovered", &documents, &vectors);
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors["first"],
            [1.0, 1.0]
        );
    }

    #[test]
    fn migration_retries_publication_and_never_replays_over_sql() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let directory = config.directories.cache.join("semantic");
        fs::create_dir_all(&directory).unwrap();
        let (documents, mut vectors) = fixture("useful", 2);
        vectors.vectors.insert("obsolete".into(), vec![1.0, 0.0]);
        fs::write(
            directory.join("model.msgpack"),
            rmp_serde::to_vec(&vectors).unwrap(),
        )
        .unwrap();
        fs::write(
            directory.join("documents-model.msgpack"),
            rmp_serde::to_vec_named(&DocumentCache {
                version: revision("commit"),
                documents,
            })
            .unwrap(),
        )
        .unwrap();
        for name in [
            "other.msgpack",
            "documents-model-old.msgpack",
            "model.tmp-123",
        ] {
            fs::write(directory.join(name), b"obsolete").unwrap();
        }
        with_writer(&config, |db, _| {
            db.connection.execute_batch("CREATE TRIGGER fail_import BEFORE INSERT ON vectors BEGIN SELECT RAISE(ABORT, 'injected import failure'); END;")?;
            Ok(())
        }).unwrap();
        assert!(load(&config, &selection("model")).is_err());
        assert!(directory.join("model.msgpack").exists());
        assert!(directory.join("documents-model.msgpack").exists());
        with_writer(&config, |db, _| {
            assert!(matches!(
                Mapping::read(&db.connection, &selection("model"))?,
                Mapping::Absent
            ));
            db.connection.execute_batch("DROP TRIGGER fail_import")?;
            Ok(())
        })
        .unwrap();
        let cached = load(&config, &selection("model")).unwrap();
        assert_eq!(cached.vectors.len(), 1);
        assert_eq!(cached.vectors["useful"], [1.0, 1.0]);
        assert_eq!(
            super::documents(&config, &selection("model"), &revision("commit"))
                .unwrap()
                .unwrap()[0]
                .keys,
            ["useful"]
        );
        assert!(!directory.join("model.msgpack").exists());
        assert!(directory.join("model.tmp-123").exists());
        assert!(directory.join("other.msgpack").exists());
        assert!(directory.join("documents-model-old.msgpack").exists());
        // Leftovers, including corrupt files, cannot replace a later SQL map.
        fs::write(directory.join("model.msgpack"), [0xc1]).unwrap();
        let (replacement, replacement_vectors) = fixture("replacement", 2);
        seed(
            &config,
            "model",
            "replacement",
            &replacement,
            &replacement_vectors,
        );
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors,
            replacement_vectors.vectors
        );
        assert!(directory.join("model.msgpack").exists());
    }

    #[test]
    fn vector_only_legacy_cache_keeps_vectors_until_mapping_publication() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let directory = config.directories.cache.join("semantic");
        fs::create_dir_all(&directory).unwrap();
        let (mapped, mut vectors) = fixture("useful", 2);
        vectors.vectors.insert("obsolete".into(), vec![0.0, 1.0]);
        fs::write(
            directory.join("model.msgpack"),
            rmp_serde::to_vec_named(&vectors).unwrap(),
        )
        .unwrap();
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors,
            vectors.vectors
        );
        let snapshot = SnapshotFixture::new(&config, "first");
        let plan = snapshot.plan(mapped);
        assert!(
            documents(&config, &selection("model"), &plan.version)
                .unwrap()
                .is_none()
        );
        assert!(
            build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
        publish_mapping(&config, &plan).unwrap();
        let cached = load(&config, &selection("model")).unwrap();
        assert_eq!(cached.vectors.len(), 1);
        assert_eq!(cached.vectors["useful"], [1.0, 1.0]);
        assert!(
            build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
        mark_build_complete(&config, &plan).unwrap();
        assert!(
            !build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
    }

    #[test]
    fn document_only_legacy_cache_migrates_with_missing_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let directory = config.directories.cache.join("semantic");
        fs::create_dir_all(&directory).unwrap();
        let (mapped, _) = fixture("missing", 2);
        let path = directory.join("documents-model.msgpack");
        fs::write(
            &path,
            rmp_serde::to_vec_named(&DocumentCache {
                version: revision("commit"),
                documents: mapped,
            })
            .unwrap(),
        )
        .unwrap();
        assert!(
            load(&config, &selection("model"))
                .unwrap()
                .vectors
                .is_empty()
        );
        assert_eq!(
            documents(&config, &selection("model"), &revision("commit"))
                .unwrap()
                .unwrap()[0]
                .keys,
            ["missing"]
        );
        assert!(!path.exists());
        assert!(
            build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
    }

    #[test]
    fn rebuild_bypasses_invalid_legacy_and_resumes_its_durable_selection() {
        for corrupt in ["model.msgpack", "documents-model.msgpack"] {
            let dir = tempfile::tempdir().unwrap();
            let config =
                Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
            let directory = config.directories.cache.join("semantic");
            fs::create_dir_all(&directory).unwrap();
            let (documents, vectors) = fixture("useful", 2);
            fs::write(
                directory.join("model.msgpack"),
                rmp_serde::to_vec(&vectors).unwrap(),
            )
            .unwrap();
            fs::write(
                directory.join("documents-model.msgpack"),
                rmp_serde::to_vec_named(&DocumentCache {
                    version: revision("commit"),
                    documents,
                })
                .unwrap(),
            )
            .unwrap();
            fs::write(directory.join(corrupt), [0xc1]).unwrap();
            fs::write(directory.join("other.msgpack"), b"unrelated").unwrap();
            fs::write(directory.join("model.tmp-123"), b"abandoned").unwrap();
            let error = load(&config, &selection("model")).err().unwrap();
            let expected = if corrupt == "model.msgpack" {
                "invalid legacy semantic vectors"
            } else {
                "invalid legacy semantic documents"
            };
            assert!(format!("{error:#}").contains(expected), "{error:#}");
            for name in [
                "model.msgpack",
                "documents-model.msgpack",
                "other.msgpack",
                "model.tmp-123",
            ] {
                assert!(directory.join(name).exists(), "{name}");
            }
            begin_rebuild(&config, &selection("model")).unwrap();
            assert!(
                load(&config, &selection("model"))
                    .unwrap()
                    .vectors
                    .is_empty()
            );
            let snapshot = SnapshotFixture::new(&config, "rebuilt");
            let (documents, vectors) = fixture("new", 2);
            let plan = snapshot.plan(documents);
            publish_mapping(&config, &plan).unwrap();
            publish_batch(&config, &plan, &vectors).unwrap();
            begin_rebuild(&config, &selection("model")).unwrap();
            assert_eq!(
                load(&config, &selection("model")).unwrap().vectors,
                vectors.vectors
            );
            mark_build_complete(&config, &plan).unwrap();
            begin_rebuild(&config, &selection("model")).unwrap();
            assert!(
                load(&config, &selection("model"))
                    .unwrap()
                    .vectors
                    .is_empty()
            );
        }
    }
}
