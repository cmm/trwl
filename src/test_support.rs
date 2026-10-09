use crate::{
    config::{Config, Source},
    record::Provider,
    store::{self, ReadIndex},
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub fn history_file(name: &str) -> (tempfile::TempDir, Config, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let config = Config::load(&dir.path().join("data")).unwrap();
    let path = dir.path().join(name);
    (dir, config, path)
}

pub fn source(provider: Provider, path: &Path) -> Source {
    Source {
        name: "test".into(),
        provider,
        root: path.into(),
    }
}

pub fn import(config: &Config, provider: Provider, path: &Path) -> ReadIndex {
    assert!(
        store::update(config, &[source(provider, path)], false)
            .unwrap()
            .completed()
    );
    ReadIndex::open(&config.root).unwrap().unwrap()
}

pub fn codex_message(role: &str, text: &str) -> Value {
    let kind = if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    };
    json!({
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": kind, "text": text}],
        },
    })
}

/// Create the OpenCode tables trwl reads, with one session. Triggers stand
/// in for OpenCode, which sets `time_updated` on every write.
pub fn opencode_db(path: &Path) -> rusqlite::Connection {
    let db = rusqlite::Connection::open(path).unwrap();
    db.execute_batch(
        "CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT, parent_id TEXT);
         CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT,
             time_created INTEGER DEFAULT 0, time_updated INTEGER DEFAULT 0, data TEXT);
         CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
             time_created INTEGER, time_updated INTEGER DEFAULT 0, data TEXT);
         CREATE TRIGGER part_touch AFTER UPDATE OF data ON part BEGIN
             UPDATE part SET time_updated = time_updated + 1 WHERE id = new.id; END;
         CREATE TRIGGER message_touch AFTER UPDATE OF data ON message BEGIN
             UPDATE message SET time_updated = time_updated + 1 WHERE id = new.id; END;
         INSERT INTO session VALUES ('session', '/workspace', NULL);",
    )
    .unwrap();
    db
}
