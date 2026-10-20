# CLI end-to-end tests

`e2e.py` supplies native provider inputs to a trwl executable and checks CLI
results. It uses POSIX process groups and Python's standard library, with
additional dependencies for the [HTTP model server](#http-embeddings).

## Contents

- [Cases and formats](#cases-and-formats)
- [HTTP embeddings](#http-embeddings)
- [Manual GPU check](#manual-gpu-check)
- [Results](#results)
- [Performance comparison](#performance-comparison)
  - [Semantic measurements](#semantic-measurements)
- [History benchmarks](#history-benchmarks)
- [OpenCode corpora](#opencode-corpora)
- [Comparison policy](#comparison-policy)
- [Harness tests](#harness-tests)

From the repository root:

```sh
tests/e2e.py --binary /absolute/path/to/trwl --model-dir /absolute/path/to/model
```

All cases run by default, including semantic and hybrid retrieval.
`--model-dir` selects the supplied model directory through
`TRWL_SEMANTIC_MODEL_DIR`, overriding the executable's default model directory.
Missing files in a supplied model directory mark the semantic case **unrun** and
make the run exit nonzero.
The HTTP cases also need an explicit model directory and the Python dependencies
listed under [HTTP embeddings](#http-embeddings).

Select individual cases by repeating `--case`:

```sh
tests/e2e.py --binary /absolute/path/to/trwl --case retrieval --case updates
```

`selection` and `workspaces` require Git on the runner's `PATH` to create native
repository fixtures. The trwl executable discovers roots without invoking Git.

`--source-revision LABEL` records a caller-supplied binary provenance label.

## Cases and formats

| Cases | Provider input and behavior |
| --- | --- |
| `hybrid-grouping` | Controlled HTTP vectors and 120 strong matches in one session; grouped retrieval still returns the ten other matching sessions; different event anchors reinforce session scores but remain separate in event mode |
| `semantic-cache` | Controlled HTTP vectors; unchanged fills reuse embeddings, inference holds no SQLite transaction, eviction reports partial/unavailable coverage without losing history, and waited queries refill |
| `benchmark-restore` | Prepared benchmark history survives restoration and its next import reports unchanged sources |
| `discovery` | Data-root flag and environment override, XDG/home defaults; setup discovers OpenCode, Codex, Pi, Claude Code and Copilot CLI histories; repeating setup preserves content and references |
| `retrieval` | OpenCode SQLite messages, tool input/output, parent/child sessions, windows, list, locate, text and Markdown transcripts |
| `opencode-v2` | Hand-built OpenCode v1/v2 SQLite; mixed and v2-only databases, streaming tools, output/errors, metadata, children, edits/deletions, migration precedence, and incremental/fresh-import agreement |
| `selection` | OpenCode SQLite and Codex rollout JSONL; matching counterexamples for filters, grouping, children and current-session exclusion |
| `workspaces` | Git roots, subdirectories, nested repositories, symlinks, SHA-256 repositories and linked worktrees; all five providers; default, glob and global selection across all backends and both result modes; root preservation through updates and deleted checkouts |
| `file-search` | Codex file operations versus prose and other-file counterexamples; alternative search terms |
| `long-content` | Complete Unicode messages and tool output, searchable tails, and retrieval after the original input moves away |
| `updates` | OpenCode append/edit/delete, tool completion, insertion order, role changes, rejection diagnostics, stable IDs, repeated imports, and unrelated Codex history; successive incremental results compared with separate full imports, including long prefixes, suffix sequence changes and truncation |
| `busy-source` | SQLite publication during continuous writes; WAL checkpoint observations establish commits during the import's read transaction, followed by catch-up |
| `many-inputs` | Eighty changed JSONL inputs import under a 64-descriptor limit; complete content and unique record IDs survive |
| `jsonl-updates` | Codex partial-line completion, same-length rewrite, truncation, unchanged-repeat deduplication, and surviving references |
| `bad-input` | Codex rollout JSONL with malformed records; diagnostics and preservation after an invalid replacement |
| `refresh-restart` | OpenCode SQLite; `off`/`wait`/`background` refresh, restarting the background process, fresh query processes, and reopening a copy of populated data |
| `interruption`, `overlap` | OpenCode SQLite and Codex rollout JSONL; SIGINT after public progress, concurrent imports, recovery and deduplication |
| `pi-jsonl` | Pi session-v3 JSONL messages and tool activity; repeated import, append, correction and deletion |
| `claude-code-jsonl` | Claude Code project JSONL messages and tool activity; repeated import, append, correction and deletion |
| `copilot-cli-jsonl` | Copilot CLI `session.start`, user and assistant events in `events.jsonl`; repeated import, append, correction and deletion |
| `provider-details` | Pi/Claude thinking, Codex reasoning summaries, OpenCode errors, `patchText` file filters, metadata parts and oversized file lists; full retrieval after inputs disappear |
| `semantic-unavailable`, `semantic` | OpenCode SQLite; lexical fallback, technical paraphrase retrieval, session exclusion, append/edit/delete catch-up, a match late in a long message, and missing-model errors |
| `semantic-passive` | A successful waited semantic query followed by the same query with refresh disabled |
| `semantic-background` | Existing history remains searchable by meaning as a background updater ingests new messages |
| `semantic-device-policy` | CPU queries and catch-up, worker-based full builds, shared-lock contention, CPU-only override, interrupted-build resumption, and model-selection changes |
| `semantic-executor` | Real-model inference across fresh CLI processes, concurrent startup, malformed-client isolation, crash recovery, idle expiry, and shutdown through `daemon stop` and `semantic disable` |
| `semantic-http` | E5 model behind an authenticated OpenAI-compatible endpoint; paraphrase retrieval, appended history and unavailable-service fallback |
| `semantic-live` | Later lexical arrivals and existing semantic/hybrid reads during a held embedding request, catch-up, and shutdown with a request pending |
| `semantic-recovery` | Failed background requests retry; model/executor selection changes during a fill re-embed all documents; restart completes an interrupted fill |

## HTTP embeddings

`semantic-http`, `semantic-live` and `semantic-recovery` run the supplied E5 ONNX
model in a local HTTP server. They need
Python packages `numpy`, `onnxruntime` and `tokenizers`, plus `--model-dir`.
Missing dependencies produce an **unrun** result and a nonzero run status.
The server uses a synthetic bearer token, masked mean pooling and normalized
vectors, with E5's
[`query:` prefix for paraphrase similarity](https://huggingface.co/intfloat/multilingual-e5-small#faq).

With Nix:

```sh
timeout --signal=INT --kill-after=5s 110s \
  nix-shell \
  -p 'python3.withPackages (p: [p.onnxruntime p.tokenizers])' \
  --run 'tests/e2e.py --binary /absolute/path/to/trwl --model-dir /absolute/path/to/model --case semantic-http'
```

The cases select HTTP embeddings with
`trwl semantic enable --executor BASE_URL --model MODEL`, where `BASE_URL`
includes `/v1`. The token comes from `TRWL_SEMANTIC_EMBEDDING_TOKEN`. Requests use
`POST BASE_URL/embeddings` with OpenAI's `model`, `input` and float encoding;
responses associate vectors with inputs by `data[].index`.

The server returns vectors in reversed array order to exercise input-index
association. After semantic/hybrid retrieval and an append, the server stops;
semantic search must fail and hybrid must report lexical fallback while
preserving stored content.

The live and recovery cases hold document requests while allowing query
requests. Recovery also returns one HTTP 503, changes the selected model name
and endpoint during fills, and stops/restarts the updater before publication.
Both model names use E5 inference; recorded request inputs verify that each
selection embeds every document. The restart check interrupts inference and
verifies that the next updater completes the fill.

## Manual GPU check

`gpu.py` is an explicit hardware check, separate from the default e2e run and CI.
The ordinary `semantic-device-policy` case checks worker execution and accepts
CPU. This command fails if the requested device is unavailable, the worker falls
back to CPU, or no model operation runs on the requested provider. `--device`
asserts the device; it does not change the executable's device selection.

For an already-built package containing the requested execution provider,
using its compiled-in model and runtime:

```sh
python3 tests/gpu.py --binary /path/to/gpu-package/bin/trwl --device openvino-gpu
```

For a Cargo-built binary, supply the model and ONNX Runtime library:

```sh
python3 tests/gpu.py --binary /path/to/target/release/trwl --device cuda \
  --model-dir /path/to/model \
  --ort-dylib /path/to/libonnxruntime.so
```

Replace the paths with existing files/directories. For Intel GPU checking, use
`--device openvino-gpu`; add `--opencl-vendors /path/to/OpenCL/vendors` when an
explicit OpenCL ICD location is needed. `--ort-dylib` and `--opencl-vendors` set
`ORT_DYLIB_PATH` and `OCL_ICD_VENDORS` in both cases. The cases do not inherit
those variables or other runtime overrides from the invoking shell.

For Intel inference, the runtime must include OpenVINO. To use a different
nixpkgs input, set `GPU_NIXPKGS` to its flake reference. Build and test in the
same environment; this leaves `flake.lock` unchanged:

```sh
nix develop \
  --override-input nixpkgs "$GPU_NIXPKGS" \
  --no-write-lock-file -c bash -euc '
    cargo build --locked
    python3 tests/gpu.py --binary target/debug/trwl --device openvino-gpu \
      --model-dir "$TRWL_SEMANTIC_MODEL_DIR" \
      --ort-dylib "$ORT_LIB_PATH/libonnxruntime.so" \
      --opencl-vendors "$OCL_ICD_VENDORS"
  '
```

The harness imports nine fixture documents into separate CPU/GPU workspaces,
rebuilds embeddings, and checks ready semantic retrieval of the paraphrase target
among the top three results. The CPU baseline sets `TRWL_SEMANTIC_CPU_ONLY=1`;
the GPU case has no CPU override. Both use the same runtime paths and each
case's isolated GPU lock directory. Cached vectors must have identical nonempty
key sets, equal nonempty dimensions, finite normalized GPU values (squared norm
within `1e-5` of one), and CPU/GPU dot products greater than `0.999`.

Both cases set `TRWL_SEMANTIC_PROFILE_DIR` to an existing artifact directory.
This opt-in diagnostic profiles the first inference batch of each built-in
session and flushes the ONNX Runtime trace before returning embeddings, before
the GPU worker can be killed. The harness reads these traces and requires a
model-operation kernel event from the requested execution provider; registration,
initialization and memory transfers do not count. CPU-only execution fails the
GPU check. Unsupported operations may still run on CPU. Profiling is off when
the variable is unset.

Commands have timeouts. Artifacts remain under `.cache/assistant/tmp/trwl-gpu-*`,
including traces and per-case command logs. The command prints the artifact path,
compared-vector count and minimum CPU/GPU cosine, and exits nonzero on failure
or missing hardware.

## Results

Each run prints its artifact directory under `.cache/trwl-e2e-*`.
Each case has an independent home, configuration, input and data directory. The
runner copies the executable into each workspace, outside the data directory,
and records the original path, invoked path, SHA-256 and version.

`report.json` contains selection, invocation, exit status and case results.
Per-case results record checks and failure tracebacks. `commands/` records each
command's argv, environment, working directory, exit status, stdout, stderr and
timeout. Artifacts remain after the run. Owned process groups are stopped on
completion, failure, SIGINT, SIGTERM and SIGHUP.

The runner attempts every selected case. Exit zero means they all passed;
failures or missing prerequisites return nonzero.

## Performance comparison

[Reference measurements and reproduction commands](benchmarks/README.md) cover
generated import, embedding, history-read and update workloads. Their inputs are
generated by tracked scripts; resource samples and input parameters are archived.

`perf.py` measures one executable or compares two through the CLI. It requires
Linux and GNU `time` on `PATH`. It runs foreground commands with automatic
indexing disabled. It launches the supplied
binaries under GNU `time` and checks their hashes before and after the run.

For measurements without a reference comparison:

```sh
tests/perf.py --binary /absolute/path/to/trwl \
  --profile reference --model-dir /absolute/path/to/model
```

This runs each workload once as an excluded warm-up, then five measured samples.
The report summarizes wall time, user/system CPU time, peak RSS, filesystem I/O,
page faults and context switches. It makes no regression decision. Exit zero
means every selected workload completed and passed its content checks.

```sh
tests/perf.py \
  --baseline /absolute/path/to/reference/trwl \
  --candidate /absolute/path/to/candidate/trwl \
  --profile reference --model-dir /absolute/path/to/model
```

`--profile reference` selects 88 operations, or 88 paired comparisons when two
executables are supplied:

| Profile | Inputs | Measured operations | Count |
| --- | --- | --- | ---: |
| Import | OpenCode/Codex × long/many sessions × 1,000/10,000 messages × 256/4,096 bytes | Initial import, unchanged import, three-message append, lexical search, show | 80 |
| Embedding | 128 short messages; 32 messages alternating short and long user/reply pairs | Empty-cache embedding fill and query | 2 |
| Semantic query | 100 and 1,000 short messages | Ready semantic query, ready hybrid query, three-message append and query | 6 |

Semantic profiles use OpenCode and one conversation. Short messages have 256
ASCII bytes; long messages have 4,096. The embedding cases exercise multiple
inference batches, multi-chunk documents and unequal document lengths. The query
sizes exercise cached lookup and append work against different amounts of
history.

Each corpus has one excluded warm-up pair and five measured pairs. Trial order
alternates between baseline-first and candidate-first. There are no additional
per-sample query warm-ups. Import trials start with empty data directories;
unchanged and append imports follow the initial import. Full CLI enumeration
checks complete content, native session identity, duplicates and stable IDs
after each import. Enumeration requests the whole fixture in one page, with
cursor continuation if needed; pagination behavior has separate tests.
Search and show check the target's identity and full text.

Input hashing warms provider pages. Each command runs in a fresh process with
warm filesystem caches. Wall time covers GNU `time` launch through completion,
including launcher overhead. CPU time includes `wait4` for GNU `time` and its
waited command/descendants, plus detached inference workers' `/proc/PID/stat`
deltas. Worker counters have `SC_CLK_TCK` resolution. Fixture work, correctness
checks, output compression and process cleanup are outside timing.

For commands without detached inference workers, `max_rss_kib` is GNU `time`'s
`%M`: the maximum Linux RSS high-water mark across the command and its waited
descendants. With workers, their lifetime `VmHWM` values are added. This sum of
peaks bounds concurrent RSS; it is not a sampled simultaneous peak. Reports keep
`client_resources` and `worker_before`/`worker_after` separately, including worker
resident RSS. CPU and RSS are diagnostics; comparison policy uses wall time.
A missing or invalid RSS report fails the trial.

Filesystem input/output bytes combine Linux `wait4` block counts, multiplied
by 512, with detached workers' `/proc/PID/io` `read_bytes`/`write_bytes` deltas.
They describe charged storage I/O, not logical read/write bytes. Cached reads can report zero input;
mmap activity is also reflected in page faults. Output includes the command's
stdout/stderr and GNU `time`'s resource file. Syscall tracing and CPU/allocator
profiling should run separately, so their overhead is not included in samples.

`--profile import` is the default; it selects both providers/shapes, sizes 1,000
and 10,000, and 256-byte messages. For individual profiles, `--sizes N [N ...]`
and `--message-bytes` select inputs. Provider/shape overrides apply only to import.
`--repetitions` selects the measured pair count; fewer than five is inconclusive.
`--timeout` bounds each command; use an outer `timeout` for the complete run.
Optional
`--baseline-label` and `--candidate-label` record caller-supplied source/build
descriptions; use comparable build configurations.

Artifacts live under `.cache/assistant/tmp/trwl-perf-*`. `report.json` records
binary identities, corpus manifests, every sample, decisions and growth between
sizes. Per-command JSON records timing, resources, command and launcher arguments,
environment and exit status. The recorded PID is the GNU `time` process-group
leader. Its raw RSS report is in an adjacent `.rss-kib` file; stdout/stderr are
preserved in adjacent gzip files. Successful
import trials' data directories are removed after verification. Semantic data
and prepared snapshots are removed after each successful scenario; `--keep-data`
preserves them. Failed scenarios retain their data. Provider inputs and logs
remain. Progress is printed after each trial and before semantic preparation.

### Semantic measurements

```sh
tests/perf.py \
  --baseline /absolute/path/to/reference/trwl \
  --candidate /absolute/path/to/candidate/trwl \
  --profile semantic-index --sizes 128 --model-dir /absolute/path/to/model

tests/perf.py \
  --baseline /absolute/path/to/reference/trwl \
  --candidate /absolute/path/to/candidate/trwl \
  --profile semantic-query --sizes 100 1000 --model-dir /absolute/path/to/model
```

`semantic-index` prepares imported history once per executable and restores that
state, without embeddings, before each sample. Its default is 128 messages.
`--long-message-bytes 4096` alternates short and long user/reply pairs; paired
messages have the same length, so complete documents have different lengths.

`semantic-query` prepares embeddings once per executable/size and records that
preparation separately. Each sample restores its own executable's ready data and
unchanged provider input. An untimed import reconciles the restored source's
ctime, and lexical readiness is checked before measurement. It measures semantic
and hybrid queries, then publishes
three new messages and measures catch-up/query. Input timestamps are restored
along with content. Snapshot copying is outside timing. Complete history
enumeration verifies the appended corpus and stable existing identities.

Size labels describe history before semantic work; only query trials append
messages. All results must report semantic participation and refer to known
complete records. Queries use fresh processes and `--refresh wait`, including
the unchanged-history check. Restoration stops inference before copying data:
the first query in each trial starts the model; later queries can reuse it.
Built-in inference is unthrottled.

Semantic comparisons use the same resource measurements and
[comparison policy](#comparison-policy) as lexical comparisons.

`perf_executor.py` separates cold-executor queries from reuse across CLI
processes. Prepare one synthetic size with `--binary --keep-data`, then pass its
report. The preparation records the data root, query and workspace as named inputs:

```sh
python3 tests/perf.py --binary /path/to/before/trwl --profile semantic-query \
  --sizes 100 --repetitions 1 --model-dir /path/to/model --keep-data
taskset -c 0,2 python3 tests/perf_executor.py \
  --prepared-report /path/to/report.json \
  --baseline /path/to/before/trwl --candidate /path/to/after/trwl \
  --model-dir /path/to/model
```

It copies the prepared data to each executable's data root. Five alternating
pairs and one excluded warm-up pair check byte-for-byte query output. Cold
samples stop inference before each command; reuse samples run an untimed query
before measurement. Filesystem caches are not evicted. Resources include the
detached worker. Reports remain under `.cache/assistant/tmp/executor-pairs-*`.

## History benchmarks

Generate tool-heavy synthetic history:

```sh
python3 tests/perf_history_corpus.py --output .cache/assistant/tmp/history-v1
python3 tests/perf_history.py --binary /absolute/path/to/trwl \
  --corpus "$PWD/.cache/assistant/tmp/history-v1/opencode.db" \
  --workspace "$PWD/.cache/assistant/tmp/history-v1/workspaces/project-00" \
  --query cache --query parser --query timeout
```

`--sessions` and `--length-multiplier` scale conversation count and individual
length independently. The [corpus description](benchmarks/README.md#generated-tool-heavy-history)
lists generated structures, measured input characteristics and limitations.

The text-only profiles measure import scaling and embedding throughput. Their
uniform message bodies and unique-marker lexical query do not cover tool-heavy
histories or broad grouped retrieval. `perf_history.py` also accepts a frozen
OpenCode database:

```sh
tests/realm_corpus.py --source /path/to/opencode.db \
  --output .cache/assistant/tmp/history-corpus
tests/perf_history.py --binary /absolute/path/to/trwl \
  --corpus "$PWD/.cache/assistant/tmp/history-corpus/opencode.db" \
  --workspace "$PWD" --semantic --model-dir /absolute/path/to/model
```

| Workload | Selection |
| --- | --- |
| Initial and unchanged import | All frozen source rows; a new data root for each trial |
| Broad lexical retrieval | `nix`, `benchmark`, `workspace`; event and session results, limit 10 |
| Lexical query forms and filters | Required words, phrase, alternative term, transcript, session, primary-only, default/target workspace and workspace glob |
| Missing match and disabled-semantic fallback | No-match lexical query; hybrid query with semantic disabled |
| Follow-up reads | Complete large event, five-event window, largest session in JSON and Markdown, event location |
| Enumeration | All events in 1,000-record JSONL pages, including continuation |
| Optional semantic work | Empty-cache CPU fill; ready semantic/hybrid queries, events/grouping and refresh off/wait |

Repeat `--query` to replace the three broad lexical queries with terms from the
supplied corpus. For a single ASCII word up to 40 characters, a literal match in
a selected source record requires nonempty results. Filtered queries also check
that returned records satisfy the filter. Reports label each query `matched` or
`empty` and record result and session counts; empty queries do not exercise ranking
or grouping. Required-word and phrase queries use two adjacent words from the
source, require matches, and are marked `skipped` if no such phrase exists.
The semantic topic query asks why workspace filtering missed session history.
Ready semantic queries must return results; they measure retrieval cost, not
relevance quality.

Each run has one excluded warm-up and five measured trials by default. Automatic
indexing and discovery are disabled; each operation uses a fresh process with
warm host caches. Native parts, full message text and tool outputs are checked
against the frozen database outside timing. Window/session/list results are
checked against complete CLI enumeration. Embedding fill is separate from ready
queries, which must report semantic participation. Foreground built-in inference
uses CPU; this profile does not measure background GPU builds.

Reports and data remain under `.cache/assistant/tmp/trwl-perf-history-*` for
separate profiling. `report.json` includes corpus,
binary and supplied-model hashes, all command/resource samples, result counts,
and per-workload medians and ranges. A changed corpus or executable fails the run.
`--repetitions` and `--timeout` control sample count and per-command deadline.
Use a frozen corpus rather than a database being written by a running agent.

Coverage depends on the input: one workspace does not measure exclusion of other
workspaces; a corpus without children does not measure session-tree grouping.
Only OpenCode is accepted here. Use the synthetic import profiles for the Codex
format and long/many-session scaling.

### Long-session updates

`perf_history_updates.py` compares edits and tool completion in the session with
the most parts in a generated corpus. It works on a copy, removes other sessions,
and sets the last tool to running before preparing each executable's index.
Each sample restores that index and either edits the final text part or completes
the last tool. Source preparation and restoration are outside timing.

```sh
python3 tests/perf_history_corpus.py --output .cache/assistant/tmp/history-updates \
  --sessions 64 --length-multiplier 2
taskset -c 0,2 python3 tests/perf_history_updates.py \
  --corpus "$PWD/.cache/assistant/tmp/history-updates/opencode.db" \
  --baseline /path/to/before/trwl --candidate /path/to/after/trwl
```

The default is five alternating pairs and one excluded warm-up pair. Omit
`--candidate` to measure one executable. `--repetitions` and `--timeout` control
sample count and per-command deadlines. The runner reports resources but makes
no regression decision. It enumerates complete source content after every sample,
compares normalized event output across executables and samples, and compares
warm-up results with full import. Artifacts remain under
`.cache/assistant/tmp/edit-pairs-*`; `report.json` records hashes, commands,
samples and summaries. See the [update workloads](benchmarks/README.md#long-session-updates).

## OpenCode corpora

`realm_corpus.py` opens an OpenCode database read-only and copies its `session`,
`message` and `part` tables from one read transaction. It preserves their columns,
row values and indexes.

```sh
tests/realm_corpus.py --source /path/to/opencode.db \
  --output .cache/trwl-corpus-full
tests/realm_corpus.py --source /path/to/opencode.db \
  --output .cache/trwl-corpus-small --root-limit 20
```

The default includes every history row. `--root-limit` selects root tasks by a
stable hash of their native IDs and includes all descendants. Sampling preserves
complete selected conversations, including full messages and tool output. Use the
same frozen source for a pair of full and sampled corpora. Cyclic parent
relationships fail generation.

Each destination must be new. It contains `opencode.db` and `manifest.json`,
which records selection, row counts, encoded row sizes, and per-table content
hashes. Generated databases contain private history; keep them in ignored storage.

Import the generated database with `trwl import --provider opencode --path PATH`.
The [history runner](#history-benchmarks) measures reads and imports against these
inputs.

## Comparison policy

With the default policy, a regression requires:

- Candidate median greater than **3×** the baseline median, **and** an added delay
  greater than **250 ms for imports** or **50 ms for search/show**.
- Both thresholds exceeded in at least **80% of paired samples** (four of five).

The noise check discards the lowest and highest fifth only when calculating
spread. If either side's remaining range exceeds the larger of half its median
and the minimum added-delay threshold, the comparison is inconclusive. Medians
and regression counts use all samples. Borderline or inconsistently repeated
slowdowns, missing samples and invalid measurements are also inconclusive.

`--ratio`, `--import-slack`, `--read-slack`, and `--max-relative-spread` change the
policy and are recorded in the report.

Exit **0** means all selected comparisons passed; **1** means a regression,
command/correctness failure, or runner error; **2** means inconclusive.

## Harness tests

```sh
tests/test-perf-compare.py
tests/test-perf-runner.py
tests/test-realm-corpus.py
```

These check the comparison policy with specified timing samples, the runner
with child processes, and native corpus preservation and sampling.
