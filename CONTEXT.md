# trwl implementation

trwl is one Rust crate. Provider readers normalize history into records; a
Tantivy index stores them for lexical search, semantic search and retrieval.
User commands are in [README.md](README.md); build and test entry points are in
[INSTALL.md](INSTALL.md). Design notes are module comments at the top of each
source file.

| File | Contents |
| --- | --- |
| [main.rs](src/main.rs) | Command dispatch and background lexical/embedding updates |
| [cli.rs](src/cli.rs) | Command-line arguments |
| [config.rs](src/config.rs) | `config.toml`, source discovery, file locks |
| [providers.rs](src/providers.rs) | Provider readers and parser checkpoints |
| [record.rs](src/record.rs) | Record type and identities |
| [report.rs](src/report.rs) | Command reports and readiness types |
| [store.rs](src/store.rs) | Index schema, import, lexical search, record lookup |
| [query.rs](src/query.rs) | Search filters, grouping, rank fusion, `show`, `list` |
| [semantic.rs](src/semantic.rs) | Embedding documents, executors, vector cache |
| [semantic/cache.rs](src/semantic/cache.rs) | SQLite cache transactions, retention limits and file-cache migration |
| [semantic/worker.rs](src/semantic/worker.rs) | GPU subprocess, transport, lock ownership and CPU recovery |
| [output.rs](src/output.rs) | Text, JSON, JSONL and Markdown rendering |

The data root holds `config.toml`, `index/` (Tantivy), `semantic/cache.sqlite3`
(embeddings and document mappings), `semantic/model-hashes.json`, lock files,
`daemon.log` and the `daemon.stop` request file.
