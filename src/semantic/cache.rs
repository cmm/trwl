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
use super::{Cache, DocumentCache, EmbeddingDocument};
use crate::config::{Config, lock};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use std::{collections::HashMap, fs, path::Path, time::Duration};

const DATABASE: &str = "cache.sqlite3";
const MAX_BYTES: i64 = 128 * 1024 * 1024;
const MAX_PAGES: i64 = 65536; // 256 MiB, including SQLite indexes and free pages.
const MAX_SELECTIONS: i64 = 4;
const MAX_AGE: i64 = 30 * 24 * 60 * 60;

struct Database {
    connection: Connection,
    budget: i64,
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
        let db = Self { connection, budget };
        // No spilling: a transaction writes each dirty page once, bounding its
        // WAL to max_page_count frames (page bytes plus 24 bytes per frame).
        // Do not append another transaction behind a pinned WAL. With trwl's
        // short, serialized snapshots this only fails for an external reader.
        db.checkpoint()?;
        let schema = db.connection.unchecked_transaction()?;
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
        db.checkpoint()?;
        Ok(db)
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

    fn maintain(&mut self) -> Result<()> {
        self.checkpoint()?;
        let tx = self.connection.transaction()?;
        tx.execute(
            "DELETE FROM selections WHERE touched < unixepoch() - ?",
            [MAX_AGE],
        )?;
        Self::trim(&tx, self.budget, None)?;
        tx.commit()?;
        // FULL auto-vacuum truncates freed tail pages at commit; checkpointing
        // transfers that truncation to the database and removes the WAL bytes.
        self.checkpoint()
    }

    fn touch(&self, selection: &str) -> Result<()> {
        // Avoid a metadata write for every query.
        self.connection.execute(
            "UPDATE selections SET touched=unixepoch() WHERE selection=? AND touched < unixepoch()-86400",
            [selection],
        )?;
        self.checkpoint()
    }

    fn trim(tx: &Connection, budget: i64, keep: Option<&str>) -> Result<()> {
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

    fn vectors(&mut self, selection: &str) -> Result<Cache> {
        let tx = self.connection.transaction()?;
        let mut vectors = HashMap::new();
        {
            let mut query = tx.prepare("SELECT key,vector FROM vectors WHERE selection=?")?;
            let mut rows = query.query([selection])?;
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
        selection: &str,
        version: &str,
    ) -> Result<Option<Vec<EmbeddingDocument>>> {
        let tx = self.connection.transaction()?;
        let stored: Option<String> = tx
            .query_row(
                "SELECT version FROM selections WHERE selection=?",
                [selection],
                |r| r.get(0),
            )
            .optional()?;
        if stored.as_deref() != Some(version) {
            return Ok(None);
        }
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

    fn publish(
        &mut self,
        selection: &str,
        version: &str,
        documents: Option<&[EmbeddingDocument]>,
        vectors: &Cache,
    ) -> Result<()> {
        self.checkpoint()?;
        let tx = self.connection.transaction()?;
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
        Self::trim(&tx, self.budget, Some(selection))?;
        tx.commit()?;
        self.checkpoint()
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
fn legacy(directory: &Path, selection: &str) -> Result<Option<(Cache, Option<DocumentCache>)>> {
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
                rmp_serde::from_slice(&fs::read(path)?).context("invalid semantic document cache")
            })
            .transpose()?;
        Ok((vectors, documents))
    })();
    for path in [vector_path, document_path].into_iter().flatten() {
        remove(&path)?;
    }
    decoded.map(Some)
}

fn with_database<T>(
    config: &Config,
    selection: &str,
    operation: impl FnOnce(&mut Database) -> Result<T>,
) -> Result<T> {
    let _lock = lock(&config.root, "semantic-cache.lock")?;
    let directory = config.root.join("semantic");
    fs::create_dir_all(&directory)?;
    let old = legacy(&directory, selection)?;
    let path = directory.join(DATABASE);
    let result: Result<T> = (|| {
        let mut db = Database::open(&path, MAX_BYTES, MAX_PAGES)?;
        db.maintain()?;
        if let Some((vectors, documents)) = old {
            db.publish(
                selection,
                documents.as_ref().map_or("", |d| d.version.as_str()),
                documents.as_ref().map(|d| d.documents.as_slice()),
                &vectors,
            )?;
        }
        let value = operation(&mut db)?;
        db.touch(selection)?;
        Ok(value)
    })();
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
            remove(&directory.join(name))?;
        }
        return result.context("semantic cache exhausted disk space or its page limit; cache cleared, history unchanged; retry --refresh wait");
    }
    result
}

pub(super) fn load(config: &Config, selection: &str) -> Result<Cache> {
    with_database(config, selection, |db| db.vectors(selection))
}

pub(super) fn documents(
    config: &Config,
    selection: &str,
    version: &str,
) -> Result<Option<Vec<EmbeddingDocument>>> {
    with_database(config, selection, |db| db.documents(selection, version))
}

pub(super) fn prepared(
    config: &Config,
    selection: &str,
) -> Result<HashMap<String, super::PreparedDocument>> {
    with_database(config, selection, |db| {
        let mut query = db
            .connection
            .prepare("SELECT input,chunks FROM prepared WHERE selection=?")?;
        let mut rows = query.query([selection])?;
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

pub(super) fn publish(
    config: &Config,
    selection: &str,
    version: &str,
    documents: &[EmbeddingDocument],
    vectors: &Cache,
) -> Result<()> {
    with_database(config, selection, |db| {
        db.publish(selection, version, Some(documents), vectors)
    })
}

pub(super) fn full_build(config: &Config, selection: &str) -> Result<bool> {
    with_database(config, selection, |db| {
        let complete: Option<bool> = db
            .connection
            .query_row(
                "SELECT coalesce(builds.complete,1) FROM selections
                 LEFT JOIN builds USING(selection) WHERE selection=?",
                [selection],
                |r| r.get(0),
            )
            .optional()?;
        Ok(!complete.unwrap_or(false))
    })
}

pub(super) fn finish_build(config: &Config, selection: &str) -> Result<()> {
    with_database(config, selection, |db| {
        db.connection.execute(
            "UPDATE builds SET complete=1 WHERE selection=? AND complete=0",
            [selection],
        )?;
        Ok(())
    })
}

pub(super) fn reset(config: &Config, selection: &str) -> Result<()> {
    with_database(config, selection, |db| {
        db.connection
            .execute("DELETE FROM selections WHERE selection=?", [selection])?;
        Ok(())
    })
}

pub(super) fn publish_batch(
    config: &Config,
    selection: &str,
    version: &str,
    documents: &[EmbeddingDocument],
    vectors: &Cache,
) -> Result<()> {
    with_database(config, selection, |db| {
        let stored: Option<String> = db
            .connection
            .query_row(
                "SELECT version FROM selections WHERE selection=?",
                [selection],
                |r| r.get(0),
            )
            .optional()?;
        // A status check or another pinned reader may have replaced the map
        // during inference. Otherwise only write the newly completed batch.
        db.publish(
            selection,
            version,
            (stored.as_deref() != Some(version)).then_some(documents),
            vectors,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{EventType, Provider, Record};

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

    #[test]
    fn finishing_a_completed_build_does_not_update_it() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(dir.path()).unwrap();
        let (documents, vectors) = fixture("first", 2);
        publish(&config, "model", "v1", &documents, &vectors).unwrap();
        finish_build(&config, "model").unwrap();
        let observer = Connection::open(config.root.join("semantic").join(DATABASE)).unwrap();
        observer
            .execute_batch(
                "CREATE TRIGGER no_completed_build_updates BEFORE UPDATE ON builds
            WHEN old.complete=1 BEGIN SELECT RAISE(ABORT, 'updated a completed build'); END;",
            )
            .unwrap();
        finish_build(&config, "model").unwrap();
        assert!(!full_build(&config, "model").unwrap());
    }

    #[test]
    fn publishing_a_snapshot_does_not_rewrite_existing_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(dir.path()).unwrap();
        let (documents, vectors) = fixture("first", 2);
        publish_batch(&config, "model", "v1", &documents, &vectors).unwrap();
        let observer = Connection::open(config.root.join("semantic").join(DATABASE)).unwrap();
        observer
            .execute_batch(
                "CREATE TRIGGER no_vector_rewrites AFTER INSERT ON vectors
            BEGIN SELECT RAISE(ABORT, 'rewrote a completed vector'); END;",
            )
            .unwrap();
        publish_batch(&config, "model", "v1", &documents, &vectors).unwrap();
        assert_eq!(load(&config, "model").unwrap().vectors, vectors.vectors);
    }

    #[test]
    fn final_snapshot_restores_vectors_evicted_by_another_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(dir.path()).unwrap();
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
        publish_batch(&config, "model", "v1", &first_documents, &first).unwrap();
        publish(&config, "model", "v2", &second_documents, &second).unwrap();
        assert!(prepared(&config, "model").unwrap().is_empty());
        assert!(
            !load(&config, "model")
                .unwrap()
                .vectors
                .contains_key("first")
        );
        publish_batch(&config, "model", "v1", &first_documents, &first).unwrap();
        assert_eq!(load(&config, "model").unwrap().vectors, first.vectors);
        assert_eq!(
            documents(&config, "model", "v1").unwrap().unwrap()[0].keys,
            ["first"]
        );
        assert!(documents(&config, "model", "v2").unwrap().is_none());
        let restored = prepared(&config, "model").unwrap();
        assert_eq!(restored["first-input"].chunks[0].1.text, "first");
    }

    #[test]
    fn full_build_state_survives_partial_batches_and_resets_per_selection() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(dir.path()).unwrap();
        let (mut documents, first) = fixture("first", 2);
        let (second_documents, second) = fixture("second", 2);
        documents.extend(second_documents);
        assert!(full_build(&config, "model").unwrap());
        publish(&config, "model", "v1", &documents, &Cache::default()).unwrap();
        publish_batch(&config, "model", "v1", &documents, &first).unwrap();
        assert!(full_build(&config, "model").unwrap());
        assert_eq!(load(&config, "model").unwrap().vectors.len(), 1);
        publish_batch(&config, "model", "v1", &documents, &second).unwrap();
        finish_build(&config, "model").unwrap();
        assert!(!full_build(&config, "model").unwrap());
        assert_eq!(load(&config, "model").unwrap().vectors.len(), 2);
        assert!(full_build(&config, "changed-model").unwrap());
        // An ordinary index update does not make the selection a full build.
        publish(&config, "model", "v2", &documents, &Cache::default()).unwrap();
        assert!(!full_build(&config, "model").unwrap());
        reset(&config, "model").unwrap();
        assert!(full_build(&config, "model").unwrap());
        assert!(load(&config, "model").unwrap().vectors.is_empty());
    }

    #[test]
    fn an_existing_selection_without_a_build_is_incremental() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(dir.path()).unwrap();
        let (documents, vectors) = fixture("first", 2);
        publish(&config, "model", "v1", &documents, &vectors).unwrap();
        with_database(&config, "model", |db| {
            db.connection.execute("DELETE FROM builds", [])?;
            Ok(())
        })
        .unwrap();
        assert!(!full_build(&config, "model").unwrap());
        publish_batch(&config, "model", "v2", &documents, &vectors).unwrap();
        assert!(!full_build(&config, "model").unwrap());
    }

    #[test]
    fn replacement_drops_old_versions_and_unreferenced_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(&dir.path().join(DATABASE), MAX_BYTES, MAX_PAGES).unwrap();
        let (first, vectors) = fixture("first", 2);
        db.publish("model", "commit1", Some(&first), &vectors)
            .unwrap();
        let (next, vectors) = fixture("next", 2);
        db.publish("model", "commit2", Some(&next), &vectors)
            .unwrap();
        assert!(db.documents("model", "commit1").unwrap().is_none());
        assert_eq!(
            db.documents("model", "commit2").unwrap().unwrap()[0].keys,
            ["next"]
        );
        let vectors = db.vectors("model").unwrap().vectors;
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
            db.publish(&selection, "commit", Some(&documents), &vectors)
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
        db.maintain().unwrap();
        assert!(db.vectors("model7").unwrap().vectors.is_empty());
        assert!(fs::metadata(&path).unwrap().len() < grown / 2);
        let (documents, vectors) = fixture("small", 2);
        for n in 0..8 {
            db.publish(&format!("model{n}"), "commit", Some(&documents), &vectors)
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
            db.publish("oversized", "commit", Some(&documents), &vectors)
                .is_err()
        );
    }

    #[test]
    fn failed_publication_rolls_back_and_a_pinned_reader_stops_wal_growth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DATABASE);
        let mut db = Database::open(&path, 10_000, MAX_PAGES).unwrap();
        let (first, vectors) = fixture("first", 2);
        db.publish("model", "old", Some(&first), &vectors).unwrap();
        let (large, vectors) = fixture("large", 10_000);
        assert!(db.publish("model", "new", Some(&large), &vectors).is_err());
        assert!(db.documents("model", "old").unwrap().is_some());
        assert!(db.documents("model", "new").unwrap().is_none());
        assert_eq!(db.vectors("model").unwrap().vectors.len(), 1);

        let reader = Connection::open(&path).unwrap();
        reader
            .execute_batch("BEGIN; SELECT * FROM selections;")
            .unwrap();
        db.connection.busy_timeout(Duration::ZERO).unwrap();
        let (next, vectors) = fixture("next", 2);
        // Publication is atomic even if an external snapshot prevents its final
        // checkpoint. The next writer must not append behind that snapshot.
        assert!(db.publish("model", "new", Some(&next), &vectors).is_err());
        let version: String = reader
            .query_row("SELECT version FROM selections", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, "old");
        let wal = path.with_file_name(format!("{DATABASE}-wal"));
        let size = fs::metadata(&wal).unwrap().len();
        assert!(size > 0);
        assert!(
            db.publish("model", "newer", Some(&first), &Cache::default())
                .is_err()
        );
        assert_eq!(fs::metadata(&wal).unwrap().len(), size);
        reader.execute_batch("COMMIT").unwrap();
        db.checkpoint().unwrap();
        assert_eq!(fs::metadata(&wal).unwrap().len(), 0);
        assert!(db.documents("model", "new").unwrap().is_some());
    }

    #[test]
    fn sqlite_full_clears_only_cache_and_next_publication_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&dir.path().join("data")).unwrap();
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
        publish(&config, "model", "old", &documents, &vectors).unwrap();
        let error = with_database(&config, "model", |db| {
            let pages: i64 = db
                .connection
                .query_row("PRAGMA page_count", [], |r| r.get(0))?;
            db.connection.pragma_update(None, "max_page_count", pages)?;
            let (large, vectors) = fixture("huge", 100_000);
            db.publish("model", "new", Some(&large), &vectors)
        })
        .unwrap_err();
        assert!(format!("{error:#}").contains("cache cleared"), "{error:#}");
        let stored =
            crate::query::resolve(&config.root, crate::cli::Entity::Event, &event.event_id)
                .unwrap();
        assert_eq!(stored.text, event.text);
        assert!(!config.root.join("semantic/cache.sqlite3").exists());
        assert!(load(&config, "model").unwrap().vectors.is_empty());
        publish(&config, "model", "recovered", &documents, &vectors).unwrap();
        assert_eq!(load(&config, "model").unwrap().vectors["first"], [1.0, 1.0]);
    }

    #[test]
    fn migration_preserves_current_vectors_and_cleans_before_a_failed_write() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&dir.path().join("data")).unwrap();
        let directory = config.root.join("semantic");
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
                version: "commit".into(),
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
        let cached = load(&config, "model").unwrap();
        assert_eq!(cached.vectors.len(), 1);
        assert_eq!(cached.vectors["useful"], [1.0, 1.0]);
        assert_eq!(
            super::documents(&config, "model", "commit")
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
        let failed = with_database(&config, "model", |db| {
            db.budget = 1;
            let (documents, vectors) = fixture("replacement", 2);
            db.publish("model", "replacement", Some(&documents), &vectors)
        })
        .unwrap_err();
        assert!(failed.to_string().contains("budget"), "{failed:#}");
        assert!(!directory.join("other.msgpack").exists());
        assert!(!directory.join("model.tmp-123").exists());
        assert_eq!(
            load(&config, "model").unwrap().vectors["useful"],
            [1.0, 1.0]
        );
    }

    #[test]
    fn recently_used_selection_survives_count_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(&dir.path().join(DATABASE), MAX_BYTES, MAX_PAGES).unwrap();
        let (documents, vectors) = fixture("chunk", 2);
        for n in 0..4 {
            db.publish(&format!("model{n}"), "commit", Some(&documents), &vectors)
                .unwrap();
            db.connection
                .execute(
                    "UPDATE selections SET touched=unixepoch()-172800+? WHERE selection=?",
                    params![n, format!("model{n}")],
                )
                .unwrap();
        }
        db.touch("model0").unwrap();
        db.publish("model4", "commit", Some(&documents), &vectors)
            .unwrap();
        assert!(db.documents("model1", "commit").unwrap().is_none());
        for model in ["model0", "model2", "model3", "model4"] {
            assert!(
                db.documents(model, "commit").unwrap().is_some(),
                "evicted {model}"
            );
        }
    }
}
