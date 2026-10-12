# trwl

trwl searches coding-agent history from OpenCode, Codex, Pi, Claude Code and
GitHub Copilot CLI. It provides lexical and semantic search, complete transcript
retrieval and source locations.  It can update the index in the background.

trwl is inspired by [ctx](https://github.com/ctxrs/ctx), around
version 2.0.3. It borrows most of the CLI contract and several key
design choices, but is not a drop-in replacement.

See [INSTALL.md](INSTALL.md) for build & install instructions.

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Searching](#searching)
  - [Workspace selection](#workspace-selection)
- [Reading transcripts](#reading-transcripts)
- [Indexing](#indexing)
- [Status](#status)
- [Semantic search](#semantic-search)
- [Output formats](#output-formats)
- [Data and configuration](#data-and-configuration)
- [Motivation](#motivation)

## Quick start

```sh
trwl setup --wait
trwl search "connection pool"
trwl show session SESSION_ID
```

`setup --wait` imports every configured and discovered history and waits for
the import to finish. Discovery covers OpenCode's XDG data directory,
`$CODEX_HOME/sessions`, `~/.pi/agent/sessions`, `$CLAUDE_CONFIG_DIR/projects`
and `~/.copilot/session-state`; unset Codex and Claude paths default to
`~/.codex` and `~/.claude`.

By default a search outputs one match per session, with event and
session IDs and a snippet. Copy a session ID into `show session` to
read the whole conversation, or `show event` with `--window 5` for
surrounding context. Searches cover the current workspace by default;
`--workspace '**'` searches all history (see [Workspace
selection](#workspace-selection)).

## How it works

`setup` imports each provider's native history — OpenCode v1/v2 SQLite, and
Codex, Pi, Claude Code and Copilot CLI JSONL — into a local index
under the data directory. Every search, transcript and location lookup
reads that index, so retrieval keeps working even if the original
history moves away.

There are three search backends: `lexical` (default) matches words and
phrases, `semantic` (if enabled in configuration) matches meaning, and
`hybrid` (if `semantic` is enabled) combines both.

## Searching

```sh
trwl search "connection pool" --events --refresh wait
trwl search "connection pool" --provider opencode --since YYYY-MM-DD
trwl search "connection pool" --session SESSION_ID
trwl search "connection pool" --file src/database.rs
trwl search "why do requests wait for connections?" --backend hybrid
```

- **Result modes:** session results (default) list one match per
  session; `--events` lists each match. `--limit` caps how many
  results are returned — sessions in session mode, events with
  `--events` — and `--term "alternative wording"` adds another query.
- **Words:** query words are alternatives; prefix `+word` to require
  it and put `"two words"` in the query for a phrase. Other
  punctuation has no special meaning.
- **Filters:** `--provider`, `--session`, `--workspace`,
  `--since`/`--until`, `--event-type`, `--content-scope`, `--file`
  (records that mention a file) and `--primary-only`. Dates are
  `YYYY-MM-DD` UTC days, inclusive on both ends, or RFC 3339
  timestamps.
- **Session IDs:** `--session` and `--exclude-session` take either
  the trwl session ID or the provider's native one; the value is
  matched against both, and the complete ID is required — prefixes
  are not accepted.
- **Refresh:** `off` reads the current index — instant, but with no
  indication that history is missing; `wait` imports before searching
  — fresh, but the search waits out the import; `background` (the
  default) searches indexed history immediately and restarts the
  updater in automatic mode, so freshness is restored without
  blocking the search. In automatic mode the updater is supposed to
  be running; the search only revives it if it died.
- **Current session:** excluded when `OPENCODE_SESSION_ID`,
  `CODEX_THREAD_ID` or `CLAUDE_SESSION_ID` identifies it;
  `--include-current-session` restores it. `--exclude-session` omits
  any other session.  Limitation: exclusions do not extend to child
  sessions automatically.

Text results contain event and session IDs, the workspace root,
retrieval mode and snippets around matches. Hybrid search combines
lexical and semantic matches (see [Semantic
search](#semantic-search)).

### Workspace selection

`search` and `list` restrict results to the current workspace by
default.

`--workspace PATTERN` replaces the default restriction. Repeat it to
combine patterns with OR. Matching is case-insensitive and uses [globset
pathname syntax](https://docs.rs/globset/latest/globset/#syntax):

- Globs match complete canonical roots, so select a root by name with
  a leading `**/`, such as `--workspace '**/my-project'`.
- Literal directory arguments, including `.` and `../project`, resolve
  to workspace roots; an absent absolute path can still match a stored
  root.
- `/work/project/**` selects descendant roots, excluding `/work/project` itself.
- `--workspace '**'` removes the restriction and includes unresolved roots.

A session's root is fixed at import time. A directory that is missing,
unreadable or relative at import time gets no root, and only
`--workspace '**'` will still find that history.

## Reading transcripts

```sh
trwl show event EVENT_ID --window 5
trwl show session SESSION_ID
trwl locate event EVENT_ID
trwl show session SESSION_ID --format markdown --out conversation.md
```

`show` and `locate` accept unambiguous ID prefixes. `show` reads complete
stored records without reopening the original input. `locate` reports the trwl
and provider identities, the source path, the original workspace directory
and the canonical root. `--out PATH` writes any format to a file (see
[Output formats](#output-formats)).

## Indexing

```sh
trwl sources list
trwl sources add work --provider opencode --root /path/to/opencode.db
trwl import --all
trwl import --provider codex --path /path/to/rollout.jsonl
trwl index mode auto
trwl status
```

Provider selectors are `opencode`, `codex`, `pi`, `claude` and
`copilot-cli`.  `sources add` registers a history for repeated
imports, and `sources remove` drops the registration; one-time `import
--path` accepts a file or directory, and automatic discovery can
re-include a removed input. Imports pick up new, changed, and deleted
history. `trwl import --all --full` forces a complete re-import.

There is no command that removes records from the index. Records leave
only when the source drops them — a session deleted in OpenCode, or
lines removed from a rollout file, picked up on the next import — or
when the index is rebuilt from scratch (see [Data and
configuration](#data-and-configuration)). `sources remove` removes
only the registration, and one-time imports are not monitored: their
history stays indexed after the input file disappears.

In automatic mode a polling updater imports changes in the background;
in manual mode imports happen only when you run them.  `daemon run`
runs the updater in the foreground; `daemon stop` stops it and the
local inference process without changing the mode. A later background
search or `setup` will restart the updater.

See [Status](#status) for readiness checks and states.

## Status

- `status` — the data directory, indexing mode, updater activity, both
  indexes, and overall `ready`.
- `index status` — lexical readiness: configured and discovered inputs
  compared with the index's source stamps, including SQLite WAL
  changes; reports counts, pending changes and removals.
- `semantic status` — embedding coverage: documents and chunks
  embedded for the selected executor/model. Other selections and old
  content do not count.
- `daemon status` — whether the updater is running.

| State | Meaning |
| --- | --- |
| `ready` | nothing pending for that index |
| `pending` | work remains, no worker active |
| `catching_up` | a worker is active |
| `uninitialized` | no lexical index exists |
| `blocked` | inputs inaccessible (lexical only) |
| `unavailable` | index, model or cache error |
| `disabled` | retrieval off (semantic only) |

Overall `ready` in `status` requires lexical readiness plus semantic
readiness (if semantic is enabled). Every readiness is a snapshot of
the stored index: it says nothing about source changes still awaiting
import, an HTTP executor's availability, or rejected records — import
reports count those.

## Semantic search

```sh
trwl semantic enable --executor builtin
trwl search "Why are database requests waiting for free connections?" --backend semantic --refresh wait
```

### What gets embedded

Each user message is paired with the last assistant reply before the
next user message, and a match always points at the user message, even
when the matching text is in the reply. User messages with no reply
are embedded too. Tool calls, their outputs, metadata and reasoning
are not embedded; use lexical search for those. Everything appears in
full transcripts.

### Executors

- `builtin` uses the configured multilingual-E5-small model; queries
  and incremental fills use CPU, bulk fills (initial history imports
  and full rebuilds) attempt to use GPU first.
- `http` uses an OpenAI-compatible endpoint:

  ```sh
  trwl semantic enable --executor http://localhost:8080/v1 --model MODEL
  ```

  trwl appends `/embeddings` to the base URL; include `/v1` if the
  server uses that prefix. `TRWL_SEMANTIC_EMBEDDING_TOKEN` supplies an
  optional bearer token. The server receives query text and
  conversation chunks from the index.

### Availability and fallback

A waited query fills missing embeddings before searching. During
background catch-up, a query will use already-embedded history and
report `retrieval.semantic_status: "partial"` in JSON. Without
embeddings, a semantic-only query will fail and a hybrid query will
fall back to lexical results, reporting the reason in
`retrieval.reason`.

## Output formats

`--format text` is the default. Search and transcript commands print
readable entries; status commands print readiness summaries; source
and import commands print source paths and results. `--format
markdown` adds headings to entries and status sections. `--out PATH`
writes the selected format to a file.

`--format json` emits one JSON document, including result arrays and metadata.
`--format jsonl` emits one JSON object per row followed by a completion record:

| Commands | Row type / payload field | Completion type |
| --- | --- | --- |
| `search` | `search_result` / `result` | `search_completion` |
| `show`, `list events` | `event_range_event` / `event` | `event_range_completion` |
| `list sessions` | `session_range_session` / `session` | `session_range_completion` |
| `sources list`, `import`, `setup --wait` | `source` / `source` | `sources_completion` |

Completion records carry the collection metadata; empty collections
still emit one, and single-object commands such as `status` and
`locate` emit the same single line for JSON and JSONL.

For `list events` and `list sessions`, pass the completion record's `next_cursor`
to the next `--cursor` request until `terminal` is true. Keep filters unchanged
between pages. Changing filters does not restart pagination: if the cursor still
identifies a row in the new selection, the command continues after it using the
new filters; otherwise it fails with `cursor no longer matches this selection`.
Concurrent updates can change later pages.

## Data and configuration

The default data directory is `$XDG_DATA_HOME/trwl` or
`~/.local/share/trwl`. To override it, use `--data-root` or set
`TRWL_DATA_ROOT`.

Configuration is read from `<data-directory>/config.toml`.  It
controls indexing, source discovery and embeddings. For automatic
indexing with built-in embeddings:

```toml
[indexing]
mode = "auto"

[sources]
automatic = true

[search]
semantic = true

[semantic]
executor = "builtin"
```

Automatic source discovery and indexing default to enabled; semantic
retrieval defaults to disabled.

 `TRWL_SEMANTIC_MODEL_DIR` overrides the model directory configured at
`trwl` build time.

 `ORT_DYLIB_PATH` overrides the ONNX Runtime shared library configured
at `trwl` build time.

Indexes and exports contain private session history data, so access to
files created by trwl is restricted to the current user.

If trwl reports an incompatible history index, remove (or move aside) the data
directory's `index/` and re-setup or re-build as appropriate.  There
is no automatic rebuild at the moment.  We'll also endeavor to avoid
frivolously changing the index format.

# Motivation

I wanted `ctx` but without the unhinged barely-checked (just my
impression, yours may be different!) slop churn, scope creep
(CodeGraph is fine as it is, thank you, I'm only here for reliable
transcript search!), half-million-line over-engineered code base, or
unconscionably-long build (using Bazel to boot).

I also wanted:
- Semantic index that is usable during catch-up.
- Some better UX defaults here and there.
- User-facing documentation that has been read and edited by a human
  and is not a scattered-around repetitive mess.
