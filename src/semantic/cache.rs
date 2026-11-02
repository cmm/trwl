//! SQLite storage for vectors, prepared inputs, document records and chunk mappings.
//!
//! Each executor/model/input-version selection keeps one index-commit mapping.
//! Replacing it removes unreferenced vectors. Retention is limited to four
//! selections and 128 MiB of encoded row data plus per-row overhead. The least
//! recently used selections are evicted first; a selection larger than the budget fails
//! publication. Access times have daily resolution and expire after 30 days.
//! Maintenance runs on every cache access, including hits and status checks.
//!
//! The database has a 256 MiB page limit. FULL auto-vacuum reclaims freed pages;
//! checkpoints truncate the WAL before and after writes. Disabling page spilling
//! limits one transaction's WAL to less than 258 MiB. A pinned external reader
//! blocks subsequent writes rather than allowing WAL accumulation.
//!
//! Operations hold semantic-cache.lock and close their connections before
//! returning; callers never wait for inference with a transaction open. On
//! SQLITE_FULL the database can be removed without leaving a trwl reader on an
//! unlinked file.
//! History remains in Tantivy, outside this cache.
//!
//! MessagePack migration reads the selected vectors and document mapping into
//! memory, then removes file caches and abandoned temporary files before SQLite
//! publication. Cleanup does not need space for another complete cache file.
use super::{Cache, DocumentCache, DocumentPlan, EmbeddingDocument, SelectionIdentity};
use crate::config::{Config, lock};
use crate::store::ConversationRevision;
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

const DATABASE: &str = "cache.sqlite3";
const MAX_BYTES: i64 = 128 * 1024 * 1024;
const MAX_PAGES: i64 = 65536; // 256 MiB, including SQLite indexes and free pages.
const MAX_SELECTIONS: i64 = 4;
const MAX_AGE: i64 = 30 * 24 * 60 * 60;

struct Database {
    connection: Connection,
    budget: i64,
}

enum Eviction<'a> {
    AnySelection,
    ProtectPublication(&'a SelectionIdentity),
}

struct LegacyMigration {
    vectors: Cache,
    documents: Option<DocumentCache>,
}

struct LockedCache {
    directory: PathBuf,
    _lock: fs::File,
}

// A successful commit does not imply that its WAL could be checkpointed.
// Keep that outcome separate from errors before commit, while returning the
// original checkpoint error to callers that require the whole write to finish.
#[must_use]
enum CommitOutcome {
    Checkpointed,
    CommittedButNotCheckpointed(anyhow::Error),
}

impl CommitOutcome {
    fn into_result(self) -> Result<()> {
        match self {
            Self::Checkpointed => Ok(()),
            Self::CommittedButNotCheckpointed(error) => Err(error),
        }
    }
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
    fn open(path: &Path, budget: i64, pages: i64) -> Result<Self> {
        let connection = Connection::open(path)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.pragma_update(None, "page_size", 4096)?;
        let auto_vacuum: i64 = connection.pragma_query_value(None, "auto_vacuum", |r| r.get(0))?;
        if auto_vacuum != 1 {
            // Setting FULL even when it is already enabled writes the database
            // header and forces a durable commit on every cache open.
            connection.pragma_update(None, "auto_vacuum", "FULL")?;
        }
        connection.execute_batch(
            "PRAGMA foreign_keys=ON;
             PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA cache_spill=OFF;
             PRAGMA journal_size_limit=0;",
        )?;
        connection.pragma_update(None, "max_page_count", pages)?;
        let mut db = Self { connection, budget };
        // No spilling: a transaction writes each dirty page once, bounding its
        // WAL to max_page_count frames (page bytes plus 24 bytes per frame).
        let schema = db.begin_write()?;
        schema.execute_batch(
            "CREATE TABLE IF NOT EXISTS selections (
                 selection TEXT PRIMARY KEY, version TEXT NOT NULL,
                 touched INTEGER NOT NULL, bytes INTEGER NOT NULL DEFAULT 0);
             CREATE TABLE IF NOT EXISTS documents (
                 selection TEXT NOT NULL REFERENCES selections ON DELETE CASCADE,
                 event TEXT NOT NULL, record BLOB NOT NULL,
                 PRIMARY KEY(selection,event));
             CREATE TABLE IF NOT EXISTS chunks (
                 selection TEXT NOT NULL, event TEXT NOT NULL,
                 ordinal INTEGER NOT NULL, key TEXT NOT NULL,
                 PRIMARY KEY(selection,event,ordinal),
                 FOREIGN KEY(selection,event) REFERENCES documents ON DELETE CASCADE);
             CREATE INDEX IF NOT EXISTS chunk_keys ON chunks(selection,key);
             CREATE TABLE IF NOT EXISTS vectors (
                 selection TEXT NOT NULL REFERENCES selections ON DELETE CASCADE,
                 key TEXT NOT NULL, vector BLOB NOT NULL,
                 PRIMARY KEY(selection,key));
             CREATE TABLE IF NOT EXISTS builds (
                 selection TEXT PRIMARY KEY REFERENCES selections ON DELETE CASCADE,
                 complete INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS prepared (
                 selection TEXT NOT NULL, event TEXT NOT NULL,
                 input TEXT NOT NULL, chunks BLOB NOT NULL,
                 PRIMARY KEY(selection,event),
                 FOREIGN KEY(selection,event) REFERENCES documents ON DELETE CASCADE);",
        )?;
        schema.commit()?;
        db.checkpoint_committed().into_result()?;
        Ok(db)
    }

    fn begin_write(&mut self) -> Result<Transaction<'_>> {
        // Do not append another transaction behind a pinned WAL. With trwl's
        // short, serialized snapshots this only fails for an external reader.
        self.checkpoint()?;
        Ok(self.connection.transaction()?)
    }

    fn checkpoint(&self) -> Result<()> {
        let busy: i64 = self
            .connection
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))?;
        ensure!(
            busy == 0,
            "semantic cache has a pinned reader; retry after it closes"
        );
        Ok(())
    }

    fn checkpoint_committed(&self) -> CommitOutcome {
        match self.checkpoint() {
            Ok(()) => CommitOutcome::Checkpointed,
            Err(error) => CommitOutcome::CommittedButNotCheckpointed(error),
        }
    }

    fn expire_and_evict(&mut self) -> Result<CommitOutcome> {
        let budget = self.budget;
        let tx = self.begin_write()?;
        tx.execute(
            "DELETE FROM selections WHERE touched < unixepoch() - ?",
            [MAX_AGE],
        )?;
        Self::evict_to_limits(&tx, budget, Eviction::AnySelection)?;
        tx.commit()?;
        // FULL auto-vacuum truncates freed tail pages at commit; checkpointing
        // transfers that truncation to the database and removes the WAL bytes.
        Ok(self.checkpoint_committed())
    }

    fn touch(&self, selection: &SelectionIdentity) -> Result<CommitOutcome> {
        // Avoid a metadata write for every query.
        self.connection.execute(
            "UPDATE selections SET touched=unixepoch() WHERE selection=? AND touched < unixepoch()-86400",
            [selection.as_str()],
        )?;
        Ok(self.checkpoint_committed())
    }

    fn evict_to_limits(tx: &Connection, budget: i64, eviction: Eviction<'_>) -> Result<()> {
        let keep = match eviction {
            Eviction::AnySelection => None,
            Eviction::ProtectPublication(selection) => Some(selection.as_str()),
        };
        loop {
            let (bytes, count): (i64, i64) = tx.query_row(
                "SELECT coalesce(sum(bytes),0), count(*) FROM selections",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            if bytes <= budget && count <= MAX_SELECTIONS {
                break;
            }
            let victim: Option<String> = tx
                .query_row(
                    "SELECT selection FROM selections WHERE ? IS NULL OR selection != ?
                 ORDER BY touched, selection LIMIT 1",
                    params![keep, keep],
                    |r| r.get(0),
                )
                .optional()?;
            let victim = victim.with_context(|| {
                format!("semantic cache selection exceeds its {budget}-byte budget")
            })?;
            tx.execute("DELETE FROM selections WHERE selection=?", [victim])?;
        }
        Ok(())
    }

    fn vectors(&mut self, selection: &SelectionIdentity) -> Result<Cache> {
        let tx = self.connection.transaction()?;
        let mut vectors = HashMap::new();
        {
            let mut query = tx.prepare("SELECT key,vector FROM vectors WHERE selection=?")?;
            let mut rows = query.query([selection.as_str()])?;
            while let Some(row) = rows.next()? {
                let bytes: Vec<u8> = row.get(1)?;
                ensure!(
                    bytes.len().is_multiple_of(4),
                    "invalid cached embedding length"
                );
                let vector = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                vectors.insert(row.get(0)?, vector);
            }
        }
        tx.commit()?;
        Ok(Cache { vectors })
    }

    fn documents(
        &mut self,
        selection: &SelectionIdentity,
        version: &ConversationRevision,
    ) -> Result<Option<Vec<EmbeddingDocument>>> {
        let tx = self.connection.transaction()?;
        if !Mapping::read(&tx, selection)?.matches(version) {
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
                    record: rmp_serde::from_slice(&bytes).context("invalid cached document")?,
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
        tx.commit()?;
        Ok(Some(documents))
    }

    fn publish_snapshot(
        &mut self,
        plan: &DocumentPlan<'_>,
        vectors: &Cache,
    ) -> Result<CommitOutcome> {
        let selection = &plan.selection.key;
        let mapping = Mapping::read(&self.connection, selection)?;
        self.publish(
            selection,
            &plan.version,
            (!mapping.matches(&plan.version)).then_some(plan.documents.as_slice()),
            vectors,
        )
    }

    fn publish(
        &mut self,
        selection: &SelectionIdentity,
        version: &ConversationRevision,
        documents: Option<&[EmbeddingDocument]>,
        vectors: &Cache,
    ) -> Result<CommitOutcome> {
        let budget = self.budget;
        let tx = self.begin_write()?;
        let eviction = Eviction::ProtectPublication(selection);
        let selection = selection.as_str();
        let version = version.as_str();
        let inserted = tx.execute(
            "INSERT OR IGNORE INTO selections(selection,version,touched) VALUES (?,?,unixepoch())",
            params![selection, version],
        )?;
        if inserted > 0 {
            tx.execute("INSERT INTO builds VALUES (?,0)", [selection])?;
        }
        tx.execute(
            "UPDATE selections SET version=?2,touched=unixepoch() WHERE selection=?1
             AND (version!=?2 OR touched<unixepoch()-86400)",
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
        {
            // Keys identify model and input content; a stored vector is reusable.
            let mut insert = tx.prepare("INSERT OR IGNORE INTO vectors VALUES (?,?,?)")?;
            let mut exists = tx.prepare("SELECT 1 FROM vectors WHERE selection=? AND key=?")?;
            for (key, vector) in &vectors.vectors {
                // Final pinned-reader repair must restore evicted batches, but
                // need not serialize vectors still present in this transaction.
                if exists.exists(params![selection, key])? {
                    continue;
                }
                let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                insert.execute(params![selection, key, bytes])?;
            }
        }
        // A vector-only legacy cache has no mapping yet. Its first mapping
        // publication removes unused keys just like any subsequent replacement.
        if !version.is_empty() {
            tx.execute("DELETE FROM vectors WHERE selection=? AND NOT EXISTS
                (SELECT 1 FROM chunks WHERE chunks.selection=vectors.selection AND chunks.key=vectors.key)", [selection])?;
        }
        tx.execute("UPDATE selections SET bytes=
            (SELECT coalesce(sum(length(record)+length(event)+128),0) FROM documents WHERE selection=?1) +
            (SELECT coalesce(sum(length(key)+length(event)+128),0) FROM chunks WHERE selection=?1) +
            (SELECT coalesce(sum(length(input)+length(event)+length(chunks)+128),0) FROM prepared WHERE selection=?1) +
            (SELECT coalesce(sum(length(key)+length(vector)+128),0) FROM vectors WHERE selection=?1)
            WHERE selection=?1", [selection])?;
        Self::evict_to_limits(&tx, budget, eviction)?;
        tx.commit()?;
        Ok(self.checkpoint_committed())
    }
}

fn remove(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

// Import only the selected legacy cache. Drop other selections, old per-commit
// maps and abandoned atomic-write files before any SQLite allocation. Useful
// vectors stay in memory during migration, so cleanup needs no second full copy.
impl LegacyMigration {
    fn read_and_remove(directory: &Path, selection: &SelectionIdentity) -> Result<Option<Self>> {
        let selection = selection.as_str();
        let vector_name = format!("{selection}.msgpack");
        let document_name = format!("documents-{selection}.msgpack");
        let mut vector_path = None;
        let mut document_path = None;
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == vector_name {
                vector_path = Some(entry.path());
            } else if name == document_name {
                document_path = Some(entry.path());
            } else if name.ends_with(".msgpack") || name.contains(".tmp-") {
                remove(&entry.path())?;
            }
        }
        if vector_path.is_none() && document_path.is_none() {
            return Ok(None);
        }
        let decoded = (|| {
            let vectors = match &vector_path {
                Some(path) => {
                    rmp_serde::from_slice(&fs::read(path)?).context("invalid semantic cache")?
                }
                None => Cache::default(),
            };
            let documents = document_path
                .as_ref()
                .map(|path| -> Result<DocumentCache> {
                    rmp_serde::from_slice(&fs::read(path)?)
                        .context("invalid semantic document cache")
                })
                .transpose()?;
            Ok(Self { vectors, documents })
        })();
        for path in [vector_path, document_path].into_iter().flatten() {
            remove(&path)?;
        }
        decoded.map(Some)
    }

    fn publish(self, db: &mut Database, selection: &SelectionIdentity) -> Result<CommitOutcome> {
        let vector_only = ConversationRevision::from_storage(String::new());
        db.publish(
            selection,
            self.documents.as_ref().map_or(&vector_only, |d| &d.version),
            self.documents.as_ref().map(|d| d.documents.as_slice()),
            &self.vectors,
        )
    }
}

impl LockedCache {
    fn acquire(config: &Config) -> Result<Self> {
        let lock = lock(&config.directories.cache, "semantic-cache.lock")?;
        let directory = config.directories.cache.join("semantic");
        fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            _lock: lock,
        })
    }

    // Own the operation connection here so every return closes it before the
    // caller classifies failures or removes files under the same cache lock.
    fn run_connected<T>(
        &self,
        selection: &SelectionIdentity,
        migration: Option<LegacyMigration>,
        operation: impl FnOnce(&mut Database) -> Result<T>,
    ) -> Result<T> {
        let mut db = Database::open(&self.directory.join(DATABASE), MAX_BYTES, MAX_PAGES)?;
        db.expire_and_evict()?.into_result()?;
        if let Some(migration) = migration {
            migration.publish(&mut db, selection)?.into_result()?;
        }
        // Metadata operations and touch share this access's preflight and final
        // checkpoint, but retain their separate autocommits. An external reader
        // can pin the WAL between them; the next access must pass preflight again.
        let value = operation(&mut db)?;
        db.touch(selection)?.into_result()?;
        Ok(value)
    }

    fn reset_if_disk_full<T>(&self, result: Result<T>) -> Result<T> {
        if result.as_ref().err().is_some_and(|e| {
            e.chain().any(|e| {
                matches!(
                    e.downcast_ref::<rusqlite::Error>(),
                    Some(rusqlite::Error::SqliteFailure(code, _))
                        if code.code == rusqlite::ErrorCode::DiskFull
                )
            })
        }) {
            // The connection has closed and semantic-cache.lock excludes all trwl readers.
            // Unlinking needs no new database/WAL pages, even on a full filesystem.
            for name in [
                DATABASE.to_owned(),
                format!("{DATABASE}-wal"),
                format!("{DATABASE}-shm"),
            ] {
                remove(&self.directory.join(name))?;
            }
            return result.context("semantic cache exhausted disk space or its page limit; cache cleared, history unchanged; retry --refresh wait");
        }
        result
    }
}

fn with_database<T>(
    config: &Config,
    selection: &SelectionIdentity,
    operation: impl FnOnce(&mut Database) -> Result<T>,
) -> Result<T> {
    let cache = LockedCache::acquire(config)?;
    let migration = LegacyMigration::read_and_remove(&cache.directory, selection)?;
    let result = cache.run_connected(selection, migration, operation);
    cache.reset_if_disk_full(result)
}

pub(super) fn load(config: &Config, selection: &SelectionIdentity) -> Result<Cache> {
    with_database(config, selection, |db| db.vectors(selection))
}

pub(super) fn documents(
    config: &Config,
    selection: &SelectionIdentity,
    version: &ConversationRevision,
) -> Result<Option<Vec<EmbeddingDocument>>> {
    with_database(config, selection, |db| db.documents(selection, version))
}

pub(super) fn covered(
    config: &Config,
    selection: &SelectionIdentity,
    version: &ConversationRevision,
) -> Result<bool> {
    let cache = LockedCache::acquire(config)?;
    let result = cache.run_connected(selection, None, |db| {
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
    });
    cache.reset_if_disk_full(result)
}

pub(super) fn prepared(
    config: &Config,
    selection: &SelectionIdentity,
) -> Result<HashMap<String, super::PreparedDocument>> {
    with_database(config, selection, |db| {
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
        )?
        .into_result()
    })
}

pub(super) fn build_history(
    config: &Config,
    selection: &SelectionIdentity,
) -> Result<BuildHistory> {
    with_database(config, selection, |db| {
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
    with_database(config, selection, |db| {
        // Completion records build history, not current mapping/vector coverage.
        db.connection.execute(
            "UPDATE builds SET complete=1 WHERE selection=? AND complete=0",
            [selection.as_str()],
        )?;
        Ok(())
    })
}

pub(super) fn reset(config: &Config, selection: &SelectionIdentity) -> Result<()> {
    with_database(config, selection, |db| {
        db.connection.execute(
            "DELETE FROM selections WHERE selection=?",
            [selection.as_str()],
        )?;
        Ok(())
    })
}

pub(super) fn publish_batch(
    config: &Config,
    plan: &DocumentPlan<'_>,
    vectors: &Cache,
) -> Result<()> {
    with_database(config, &plan.selection.key, |db| {
        db.publish_snapshot(plan, vectors)?.into_result()
    })
}

pub(super) fn repair_snapshot(
    config: &Config,
    plan: &DocumentPlan<'_>,
    vectors: &Cache,
) -> Result<()> {
    with_database(config, &plan.selection.key, |db| {
        db.publish_snapshot(plan, vectors)?.into_result()
    })
}

#[cfg(test)]
mod tests {
    use super::super::VectorCoverage;
    use super::*;
    use crate::record::{EventType, Provider, Record};

    fn selection(value: &str) -> SelectionIdentity {
        SelectionIdentity(value.to_owned())
    }

    fn revision(value: &str) -> ConversationRevision {
        ConversationRevision::from_storage(value.to_owned())
    }

    #[test]
    fn opening_an_initialized_cache_does_not_write_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DATABASE);
        drop(Database::open(&path, MAX_BYTES, MAX_PAGES).unwrap());
        // Separate the timestamps even on filesystems with coarse resolution.
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::UNIX_EPOCH)
            .unwrap();
        let before = fs::metadata(&path).unwrap().modified().unwrap();
        drop(Database::open(&path, MAX_BYTES, MAX_PAGES).unwrap());
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
            db.publish(&selection, &version, Some(documents), vectors)?
                .into_result()
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
    fn publishing_a_snapshot_does_not_rewrite_existing_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let (documents, vectors) = fixture("first", 2);
        let snapshot = SnapshotFixture::new(&config, "first");
        let plan = snapshot.plan(documents);
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
    fn final_snapshot_restores_vectors_evicted_by_another_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let (mut first_documents, first) = fixture("first", 2);
        first_documents[0].prepared = Some(super::super::PreparedDocument {
            input: "first-input".into(),
            chunks: vec![(
                "first".into(),
                super::super::ChunkInput {
                    text: "first".into(),
                    tokens: 1,
                },
            )],
        });
        let (second_documents, second) = fixture("second", 2);
        let first_snapshot = SnapshotFixture::new(&config, "first");
        let first_plan = first_snapshot.plan(first_documents);
        let second_snapshot = SnapshotFixture::new(&config, "second");
        let second_plan = second_snapshot.plan(second_documents);
        publish_batch(&config, &first_plan, &first).unwrap();
        publish_batch(&config, &second_plan, &second).unwrap();
        assert!(prepared(&config, &selection("model")).unwrap().is_empty());
        assert!(
            !load(&config, &selection("model"))
                .unwrap()
                .vectors
                .contains_key("first")
        );
        // A batch restores its own map after inference, but does not mark the
        // build complete, even when its vectors happen to cover that map.
        publish_batch(&config, &first_plan, &first).unwrap();
        assert!(
            documents(&config, &selection("model"), &first_plan.version)
                .unwrap()
                .is_some()
        );
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors,
            first.vectors
        );
        assert!(
            build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
        publish_batch(&config, &second_plan, &second).unwrap();
        let observer =
            Connection::open(config.directories.cache.join("semantic").join(DATABASE)).unwrap();
        observer
            .execute_batch(
                "CREATE TRIGGER complete_only_after_repair BEFORE UPDATE OF complete ON builds
                 WHEN new.complete=1 AND NOT EXISTS
                 (SELECT 1 FROM vectors WHERE selection=new.selection AND key='first')
                 BEGIN SELECT RAISE(ABORT, 'completed before pinned snapshot repair'); END;",
            )
            .unwrap();
        repair_snapshot(&config, &first_plan, &first).unwrap();
        mark_build_complete(&config, &first_plan).unwrap();
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors,
            first.vectors
        );
        assert_eq!(
            documents(&config, &selection("model"), &first_plan.version)
                .unwrap()
                .unwrap()[0]
                .keys,
            ["first"]
        );
        assert!(
            documents(&config, &selection("model"), &second_plan.version)
                .unwrap()
                .is_none()
        );
        let restored = prepared(&config, &selection("model")).unwrap();
        assert_eq!(restored["first-input"].chunks[0].1.text, "first");
        assert!(matches!(
            build_history(&config, &selection("model")).unwrap(),
            BuildHistory::Completed
        ));
        // Completed history does not imply coverage, even for the same map.
        observer.execute("DELETE FROM vectors", []).unwrap();
        assert!(matches!(
            first_plan.vector_coverage(&load(&config, &selection("model")).unwrap()),
            VectorCoverage::Missing
        ));
        repair_snapshot(&config, &first_plan, &first).unwrap();
        mark_build_complete(&config, &first_plan).unwrap();
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors,
            first.vectors
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
        repair_snapshot(&config, &plan, &load(&config, &selection("model")).unwrap()).unwrap();
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
        reset(&config, &selection("model")).unwrap();
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
        publish_batch(&config, &next.plan(plan.documents), &vectors).unwrap();
        assert!(
            !build_history(&config, &selection("model"))
                .unwrap()
                .needs_full_build()
        );
    }

    #[test]
    fn replacement_drops_old_versions_and_unreferenced_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(&dir.path().join(DATABASE), MAX_BYTES, MAX_PAGES).unwrap();
        let (first, vectors) = fixture("first", 2);
        db.publish(
            &selection("model"),
            &revision("commit1"),
            Some(&first),
            &vectors,
        )
        .and_then(CommitOutcome::into_result)
        .unwrap();
        let (next, vectors) = fixture("next", 2);
        db.publish(
            &selection("model"),
            &revision("commit2"),
            Some(&next),
            &vectors,
        )
        .and_then(CommitOutcome::into_result)
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
        let vectors = db.vectors(&selection("model")).unwrap().vectors;
        assert_eq!(vectors.len(), 1);
        assert!(vectors.contains_key("next"));
    }

    #[test]
    fn global_budget_expiry_and_selection_count_reclaim_physical_storage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DATABASE);
        let mut db = Database::open(&path, 180_000, MAX_PAGES).unwrap();
        let (documents, vectors) = fixture("large", 32_000);
        for n in 0..8 {
            let selection = format!("model{n}");
            db.publish(
                &SelectionIdentity(selection),
                &revision("commit"),
                Some(&documents),
                &vectors,
            )
            .and_then(CommitOutcome::into_result)
            .unwrap();
            let count: i64 = db
                .connection
                .query_row("SELECT count(*) FROM selections", [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, 1, "budget is global, not per selection");
            assert!(fs::metadata(&path).unwrap().len() < 220_000);
            assert_eq!(
                fs::metadata(path.with_file_name(format!("{DATABASE}-wal")))
                    .unwrap()
                    .len(),
                0
            );
        }
        let grown = fs::metadata(&path).unwrap().len();
        db.connection
            .execute("UPDATE selections SET touched=0", [])
            .unwrap();
        db.expire_and_evict()
            .and_then(CommitOutcome::into_result)
            .unwrap();
        assert!(db.vectors(&selection("model7")).unwrap().vectors.is_empty());
        assert!(fs::metadata(&path).unwrap().len() < grown / 2);
        let (documents, vectors) = fixture("small", 2);
        for n in 0..8 {
            db.publish(
                &selection(&format!("model{n}")),
                &revision("commit"),
                Some(&documents),
                &vectors,
            )
            .and_then(CommitOutcome::into_result)
            .unwrap();
        }
        let count: i64 = db
            .connection
            .query_row("SELECT count(*) FROM selections", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, MAX_SELECTIONS);
        // Prepared text shares the vector budget, even when vectors are tiny.
        let (mut documents, vectors) = fixture("oversized", 2);
        documents[0].prepared = Some(super::super::PreparedDocument {
            input: "oversized-input".into(),
            chunks: vec![(
                "oversized".into(),
                super::super::ChunkInput {
                    text: "x".repeat(180_000),
                    tokens: 1,
                },
            )],
        });
        assert!(
            db.publish(
                &selection("oversized"),
                &revision("commit"),
                Some(&documents),
                &vectors
            )
            .is_err()
        );
    }

    #[test]
    fn failed_publication_rolls_back_and_a_pinned_reader_stops_wal_growth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DATABASE);
        let mut db = Database::open(&path, 10_000, MAX_PAGES).unwrap();
        let (first, vectors) = fixture("first", 2);
        db.publish(
            &selection("model"),
            &revision("old"),
            Some(&first),
            &vectors,
        )
        .and_then(CommitOutcome::into_result)
        .unwrap();
        let (large, vectors) = fixture("large", 10_000);
        assert!(
            db.publish(
                &selection("model"),
                &revision("new"),
                Some(&large),
                &vectors
            )
            .is_err()
        );
        assert!(
            db.documents(&selection("model"), &revision("old"))
                .unwrap()
                .is_some()
        );
        assert!(
            db.documents(&selection("model"), &revision("new"))
                .unwrap()
                .is_none()
        );
        assert_eq!(db.vectors(&selection("model")).unwrap().vectors.len(), 1);

        let reader = Connection::open(&path).unwrap();
        reader
            .execute_batch("BEGIN; SELECT * FROM selections;")
            .unwrap();
        db.connection.busy_timeout(Duration::ZERO).unwrap();
        let (next, vectors) = fixture("next", 2);
        // Publication is atomic even if an external snapshot prevents its final
        // checkpoint. The next writer must not append behind that snapshot.
        let outcome = db
            .publish(&selection("model"), &revision("new"), Some(&next), &vectors)
            .unwrap();
        assert!(matches!(
            &outcome,
            CommitOutcome::CommittedButNotCheckpointed(_)
        ));
        assert!(
            outcome
                .into_result()
                .unwrap_err()
                .to_string()
                .contains("pinned reader")
        );
        let version: String = reader
            .query_row("SELECT version FROM selections", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, "old");
        let wal = path.with_file_name(format!("{DATABASE}-wal"));
        let size = fs::metadata(&wal).unwrap().len();
        assert!(size > 0);
        for _ in 0..3 {
            assert!(
                db.publish(
                    &selection("model"),
                    &revision("newer"),
                    Some(&first),
                    &Cache::default()
                )
                .is_err()
            );
            assert_eq!(fs::metadata(&wal).unwrap().len(), size);
        }
        reader.execute_batch("COMMIT").unwrap();
        db.checkpoint().unwrap();
        assert_eq!(fs::metadata(&wal).unwrap().len(), 0);
        assert!(
            db.documents(&selection("model"), &revision("new"))
                .unwrap()
                .is_some()
        );
        let (repaired, vectors) = fixture("repaired", 2);
        db.publish(
            &selection("model"),
            &revision("repaired"),
            Some(&repaired),
            &vectors,
        )
        .and_then(CommitOutcome::into_result)
        .unwrap();
        assert_eq!(
            db.vectors(&selection("model")).unwrap().vectors,
            vectors.vectors
        );
        assert!(
            db.documents(&selection("model"), &revision("repaired"))
                .unwrap()
                .is_some()
        );
        assert_eq!(fs::metadata(&wal).unwrap().len(), 0);
    }

    #[test]
    fn sqlite_full_clears_only_cache_and_next_publication_recovers() {
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
            )?
            .into_result()
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("cache cleared"), "{error:#}");
        let stored = crate::query::resolve(
            &config.directories.data,
            crate::cli::Entity::Event,
            &event.event_id,
        )
        .unwrap();
        assert_eq!(stored.text, event.text);
        for name in [
            DATABASE.to_owned(),
            format!("{DATABASE}-wal"),
            format!("{DATABASE}-shm"),
        ] {
            assert!(
                !config
                    .directories
                    .cache
                    .join("semantic")
                    .join(name)
                    .exists()
            );
        }
        for selection in ["model", "other"] {
            assert!(matches!(
                build_history(&config, &SelectionIdentity(selection.to_owned())).unwrap(),
                BuildHistory::Absent
            ));
            assert!(
                load(&config, &SelectionIdentity(selection.to_owned()))
                    .unwrap()
                    .vectors
                    .is_empty()
            );
        }
        assert!(
            load(&config, &selection("model"))
                .unwrap()
                .vectors
                .is_empty()
        );
        seed(&config, "model", "recovered", &documents, &vectors);
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors["first"],
            [1.0, 1.0]
        );
    }

    #[test]
    fn migration_preserves_current_vectors_and_cleans_before_a_failed_write() {
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
        assert!(!directory.join("model.tmp-123").exists());
        assert!(!directory.join("other.msgpack").exists());

        fs::write(directory.join("other.msgpack"), b"obsolete").unwrap();
        fs::write(directory.join("model.tmp-123"), b"abandoned").unwrap();
        let failed = with_database(&config, &selection("model"), |db| {
            db.budget = 1;
            let (documents, vectors) = fixture("replacement", 2);
            db.publish(
                &selection("model"),
                &revision("replacement"),
                Some(&documents),
                &vectors,
            )?
            .into_result()
        })
        .unwrap_err();
        assert!(failed.to_string().contains("budget"), "{failed:#}");
        assert!(!directory.join("other.msgpack").exists());
        assert!(!directory.join("model.tmp-123").exists());
        assert_eq!(
            load(&config, &selection("model")).unwrap().vectors["useful"],
            [1.0, 1.0]
        );
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
        repair_snapshot(&config, &plan, &cached).unwrap();
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
    fn invalid_legacy_data_is_removed_before_returning_the_decode_error() {
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
                "invalid semantic cache"
            } else {
                "invalid semantic document cache"
            };
            assert!(error.to_string().contains(expected), "{error:#}");
            for name in [
                "model.msgpack",
                "documents-model.msgpack",
                "other.msgpack",
                "model.tmp-123",
                DATABASE,
            ] {
                assert!(!directory.join(name).exists(), "{name}");
            }
            assert!(
                load(&config, &selection("model"))
                    .unwrap()
                    .vectors
                    .is_empty()
            );
        }
    }

    #[test]
    fn recently_used_selection_survives_count_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(&dir.path().join(DATABASE), MAX_BYTES, MAX_PAGES).unwrap();
        let (documents, vectors) = fixture("chunk", 2);
        for n in 0..4 {
            db.publish(
                &selection(&format!("model{n}")),
                &revision("commit"),
                Some(&documents),
                &vectors,
            )
            .and_then(CommitOutcome::into_result)
            .unwrap();
            db.connection
                .execute(
                    "UPDATE selections SET touched=unixepoch()-172800+? WHERE selection=?",
                    params![n, format!("model{n}")],
                )
                .unwrap();
        }
        db.touch(&selection("model0"))
            .and_then(CommitOutcome::into_result)
            .unwrap();
        db.publish(
            &selection("model4"),
            &revision("commit"),
            Some(&documents),
            &vectors,
        )
        .and_then(CommitOutcome::into_result)
        .unwrap();
        assert!(
            db.documents(&selection("model1"), &revision("commit"))
                .unwrap()
                .is_none()
        );
        for model in ["model0", "model2", "model3", "model4"] {
            assert!(
                db.documents(&selection(model), &revision("commit"))
                    .unwrap()
                    .is_some(),
                "evicted {model}"
            );
        }
    }
}
