//! Embedding documents, the vector cache and semantic search.
//!
//! A document pairs a user message with the last nonempty assistant reply
//! before the next user message in the same session, using visible message
//! text; a user message without a reply is a document on its own. Control
//! messages are not anchors, and an empty user message still ends a turn. A
//! fast field selects conversation messages before their payloads load, so
//! tools, metadata and standalone reasoning are never embedded.
//!
//! Built-in chunks use the model's token budget; HTTP chunks use a character
//! budget. SQLite stores one document-to-chunk mapping per selection and vectors
//! keyed by chunk content. Cache hits do not load transcripts. Readers with different
//! conversation inputs plan locally from their pinned index. Selection includes the input
//! version and executor/model; built-in selection includes a hash of the model
//! files. File hashes are cached by path, inode, size, mtime and ctime.
//! Unchanged chunks reuse vectors. Obsolete mappings and unreferenced vectors
//! are removed on replacement; older pinned readers can report missing coverage.
//! Transactions and legacy-file migration live in `index`.
//!
//! The built-in executor uses masked mean pooling and E5's `passage: ` and
//! `query: ` prefixes
//! (<https://huggingface.co/intfloat/multilingual-e5-small#faq>), and batches
//! inputs by token count. Queries and incremental fills use CPU. Full builds
//! use a short-lived worker trying CUDA, Intel OpenVINO GPU, then CPU.
//! HTTP executors receive unprefixed text at
//! `POST BASE_URL/embeddings`, batched by character count; response vectors
//! are matched by `data[].index` and validated. All vectors are normalized.
//!
//! Hybrid search checks the selection, cache and query embedding before
//! loading conversation text, so a failure falls back to lexical results
//! without that cost. A semantic result is the user message anchoring its
//! document, and filters apply to that record.

use crate::{
    config::{Config, atomic_write, is_locked, lock},
    record::{ConversationMessage, Record, digest},
    report::{Coverage, SemanticStatus, State},
    store::{self, ReadIndex},
};
use anyhow::{Context, Result, bail, ensure};
use ort::{
    environment::Environment,
    ep::{ArbitrarilyConfigurableExecutionProvider, ExecutionProvider, ExecutionProviderDispatch},
    session::Session,
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    io::Read,
    os::unix::fs::{FileTypeExt, MetadataExt},
    path::{Path, PathBuf},
    time::Duration,
};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

mod index;
pub(crate) mod server;
pub(crate) mod worker;

const BUILTIN_MODEL: &str = "multilingual-e5-small";
const INPUT_VERSION: &str = "embedding-v4";
// E5's 512-token limit includes special tokens and its retrieval prefix.
const MODEL_MAX_TOKENS: usize = 512;
const DOCUMENT_PREFIX: &str = "passage: ";
const QUERY_PREFIX: &str = "query: ";
// Character limits also bound HTTP payloads without the remote tokenizer.
// A quarter-chunk overlap keeps context on both sides of a chunk boundary.
const DOCUMENT_MAX_CHARS: usize = 64 * 1024;
const CHUNK_CHARS: usize = 480;
const CHUNK_OVERLAP_CHARS: usize = 120;
// Limit inference memory and CPU use per executor, not per machine.
const BATCH_SIZE: usize = 16;
const INFERENCE_THREADS: usize = 2;
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const HTTP_IO_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ChunkInput {
    text: String,
    tokens: usize,
}

type ChunkInputs = BTreeMap<String, ChunkInput>;
#[cfg(test)]
thread_local! { static HASHED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

pub fn validate_selection<'a>(executor: &str, model: Option<&'a str>) -> Result<&'a str> {
    ensure!(
        executor == "builtin"
            || executor.starts_with("http://")
            || executor.starts_with("https://"),
        "embedding executor must be builtin or an HTTP base URL"
    );
    ensure!(
        executor == "builtin" || model.is_some_and(|name| !name.trim().is_empty()),
        "HTTP embeddings require --model"
    );
    Ok(model.unwrap_or(BUILTIN_MODEL))
}

#[derive(Serialize, Deserialize, Default)]
struct Cache {
    vectors: HashMap<String, Vec<f32>>,
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct ModelStamp {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl From<&fs::Metadata> for ModelStamp {
    fn from(m: &fs::Metadata) -> Self {
        Self {
            device: m.dev(),
            inode: m.ino(),
            size: m.len(),
            modified: (m.mtime(), m.mtime_nsec()),
            changed: (m.ctime(), m.ctime_nsec()),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ModelHash {
    stamp: ModelStamp,
    hash: String,
}

impl ModelHash {
    fn read_stable(mut file: File, stamp: ModelStamp) -> Result<Self> {
        let mut hash = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
            #[cfg(test)]
            HASHED_BYTES.with(|bytes| bytes.set(bytes.get() + count));
        }
        ensure!(
            ModelStamp::from(&file.metadata()?) == stamp,
            "model file changed while hashing; retry"
        );
        Ok(Self {
            stamp,
            hash: hex::encode(hash.finalize()),
        })
    }
}

struct ModelHashMemo {
    path: PathBuf,
    entries: BTreeMap<PathBuf, ModelHash>,
    selected: HashSet<PathBuf>,
    changed: bool,
    hashes: Vec<String>,
}

impl ModelHashMemo {
    fn load(path: PathBuf) -> Result<Self> {
        let entries = match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("invalid model hash cache")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Self {
            path,
            entries,
            selected: HashSet::new(),
            changed: false,
            hashes: Vec::new(),
        })
    }

    fn select(&mut self, directory: &Path, name: &str) -> Result<()> {
        let file_path = directory
            .join(name)
            .canonicalize()
            .with_context(|| format!("opening model file {}", directory.join(name).display()))?;
        let file = File::open(&file_path)?;
        let stamp = ModelStamp::from(&file.metadata()?);
        if !self
            .entries
            .get(&file_path)
            .is_some_and(|entry| entry.stamp == stamp)
        {
            let entry = ModelHash::read_stable(file, stamp)?;
            self.entries.insert(file_path.clone(), entry);
            self.changed = true;
        }
        self.hashes
            .push(format!("{name}:{}", self.entries[&file_path].hash));
        self.selected.insert(file_path);
        Ok(())
    }

    fn finish(mut self) -> Result<String> {
        let previous = self.entries.len();
        self.entries.retain(|path, _| self.selected.contains(path));
        self.changed |= self.entries.len() != previous;
        if self.changed {
            // Hash memoization must not prevent the database from reclaiming space
            // on a full filesystem. The computed hash remains usable without it.
            if atomic_write(&self.path, &serde_json::to_vec(&self.entries)?).is_err() {
                let _ = fs::remove_file(
                    self.path
                        .with_extension(format!("tmp-{}", std::process::id())),
                );
            }
        }
        Ok(digest(
            &self.hashes.iter().map(String::as_str).collect::<Vec<_>>(),
        ))
    }
}

fn model_hash(config: &Config, directory: &Path) -> Result<String> {
    let _lock = lock(&config.directories.cache, "semantic-cache.lock")?;
    let mut memo =
        ModelHashMemo::load(config.directories.cache.join("semantic/model-hashes.json"))?;
    for name in [
        "onnx/model.onnx",
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ] {
        memo.select(directory, name)?;
    }
    memo.finish()
}

#[derive(Serialize, Deserialize)]
struct EmbeddingDocument {
    record: Record,
    keys: Vec<String>,
    #[serde(default)]
    prepared: Option<PreparedDocument>,
}

#[derive(Clone, Serialize, Deserialize)]
struct PreparedDocument {
    input: String,
    chunks: Vec<(String, ChunkInput)>,
}

#[derive(Serialize, Deserialize)]
struct DocumentCache {
    version: crate::store::ConversationRevision,
    documents: Vec<EmbeddingDocument>,
}

fn load_tokenizer(directory: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer in {}: {e}", directory.display()))
}

fn chunk_tokenizer(selection: &Selection) -> Result<Option<Tokenizer>> {
    selection
        .directory
        .as_ref()
        .map(|directory| {
            let mut tokenizer = load_tokenizer(directory)?;
            tokenizer
                .with_truncation(None)
                .map_err(|e| anyhow::anyhow!("configuring chunk tokenizer: {e}"))?;
            tokenizer.with_padding(None);
            Ok(tokenizer)
        })
        .transpose()
}

enum PlanningTokenizer {
    Owned(PlanningState),
    Shared {
        selection: SelectionIdentity,
        client: server::Client,
    },
}

#[derive(Default)]
enum PlanningState {
    #[default]
    Uninitialized,
    Local {
        selection: SelectionIdentity,
        tokenizer: Box<Tokenizer>,
    },
    Remote {
        selection: SelectionIdentity,
    },
}

impl PlanningState {
    fn prepare(&mut self, selection: &Selection, texts: &[String]) -> Result<Vec<Vec<ChunkInput>>> {
        let key = match self {
            Self::Uninitialized => None,
            Self::Local { selection, .. } | Self::Remote { selection } => Some(&*selection),
        };
        if key != Some(&selection.key) {
            // Drop the old vocabulary before initialization so failure is retryable.
            *self = Self::Uninitialized;
            *self = match chunk_tokenizer(selection)? {
                Some(tokenizer) => Self::Local {
                    selection: selection.key.clone(),
                    tokenizer: Box::new(tokenizer),
                },
                None => Self::Remote {
                    selection: selection.key.clone(),
                },
            };
        }
        match self {
            Self::Local { tokenizer, .. } => prepare_chunks(texts, Some(tokenizer)),
            Self::Remote { .. } => prepare_chunks(texts, None),
            Self::Uninitialized => unreachable!("planning initialization succeeded"),
        }
    }
}

impl Default for PlanningTokenizer {
    fn default() -> Self {
        Self::Owned(PlanningState::default())
    }
}

impl PlanningTokenizer {
    fn shared(config: &Config, selection: &Selection) -> Self {
        Self::Shared {
            selection: selection.key.clone(),
            client: server::Client::new(&config.directories, selection),
        }
    }

    fn materialize(
        &mut self,
        config: &Config,
        selection: &Selection,
        index: &ReadIndex,
    ) -> Result<(Vec<EmbeddingDocument>, ChunkInputs)> {
        let reusable = index::prepared(config, &selection.key)?;
        plan_records(&index.messages()?, &reusable, |texts| match self {
            Self::Owned(state) => state.prepare(selection, texts),
            Self::Shared {
                selection: key,
                client,
            } => {
                ensure!(key == &selection.key, "planning client selection differs");
                client.prepare(texts)
            }
        })
    }
}

#[cfg(test)]
fn materialize(
    index: &ReadIndex,
    tokenizer: Option<&mut Tokenizer>,
) -> Result<(Vec<EmbeddingDocument>, ChunkInputs)> {
    let records = index.messages()?;
    match tokenizer {
        Some(tokenizer) => DocumentTokenizer::prepare(tokenizer, |tokenizer| {
            materialize_documents(&records, Some(tokenizer))
        }),
        None => materialize_documents(&records, None),
    }
}

struct DocumentTokenizer<'a> {
    tokenizer: &'a mut Tokenizer,
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl DocumentTokenizer<'_> {
    fn prepare<T>(
        tokenizer: &mut Tokenizer,
        prepare: impl FnOnce(&Tokenizer) -> Result<T>,
    ) -> Result<T> {
        // Chunk complete text without padding or truncation. Restore both settings
        // even when document tokenization fails.
        let truncation = tokenizer.get_truncation().cloned();
        let padding = tokenizer.get_padding().cloned();
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("configuring chunk tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        let document = DocumentTokenizer {
            tokenizer,
            truncation,
            padding,
        };
        let result = prepare(document.tokenizer);
        // Restoration is fallible and only runs on normal return, not unwinding.
        document.restore()?;
        result
    }

    fn restore(self) -> Result<()> {
        self.tokenizer.with_padding(self.padding);
        self.tokenizer
            .with_truncation(self.truncation)
            .map_err(|e| anyhow::anyhow!("restoring inference tokenizer: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
fn materialize_documents(
    records: &[ConversationMessage],
    tokenizer: Option<&Tokenizer>,
) -> Result<(Vec<EmbeddingDocument>, ChunkInputs)> {
    plan_records(records, &HashMap::new(), |texts| {
        prepare_chunks(texts, tokenizer)
    })
}

fn plan_records(
    records: &[ConversationMessage],
    reusable: &HashMap<String, PreparedDocument>,
    mut prepare: impl FnMut(&[String]) -> Result<Vec<Vec<ChunkInput>>>,
) -> Result<(Vec<EmbeddingDocument>, ChunkInputs)> {
    let mut inputs = BTreeMap::new();
    let mut plan = Vec::new();
    for batch in documents(records).chunks(BATCH_SIZE) {
        // Cap documents before sending them to inference so a native transcript
        // cannot make a planning request exceed the transport's frame limit.
        let texts: Vec<String> = batch
            .iter()
            .map(|(_, text)| text.chars().take(DOCUMENT_MAX_CHARS).collect())
            .collect();
        // Pairing and result metadata come from this commit. Reuse only the
        // capped composed input: a later reply can change an older user's plan.
        let identities: Vec<_> = texts.iter().map(|text| digest(&[text])).collect();
        let missing: Vec<_> = texts
            .iter()
            .zip(&identities)
            .filter(|(_, input)| !reusable.contains_key(*input))
            .map(|(text, _)| text.clone())
            .collect();
        let prepared = if missing.is_empty() {
            Vec::new()
        } else {
            prepare(&missing)?
        };
        ensure!(
            prepared.len() == missing.len(),
            "planning batch size differs"
        );
        let mut prepared = prepared.into_iter();
        for ((record, _), input) in batch.iter().zip(identities) {
            let prepared = match reusable.get(&input) {
                Some(prepared) => prepared.clone(),
                None => PreparedDocument {
                    input,
                    chunks: prepared
                        .next()
                        .unwrap()
                        .into_iter()
                        .map(|chunk| (digest(&[&chunk.text]), chunk))
                        .collect(),
                },
            };
            let mut keys = Vec::new();
            for (key, chunk) in &prepared.chunks {
                keys.push(key.clone());
                inputs.entry(key.clone()).or_insert_with(|| chunk.clone());
            }
            plan.push(EmbeddingDocument {
                record: (**record).clone(),
                keys,
                prepared: Some(prepared),
            });
        }
    }
    Ok((plan, inputs))
}

enum Materialization {
    Deferred,
    Materialized(ChunkInputs),
}

enum VectorCoverage {
    Complete,
    Missing,
}

struct DocumentPlan<'a> {
    selection: &'a Selection,
    index: &'a ReadIndex,
    version: crate::store::ConversationRevision,
    documents: Vec<EmbeddingDocument>,
    materialization: Materialization,
}

impl DocumentPlan<'_> {
    fn vector_coverage(&self, cache: &Cache) -> VectorCoverage {
        if self
            .documents
            .iter()
            .flat_map(|document| &document.keys)
            .all(|key| cache.vectors.contains_key(key))
        {
            VectorCoverage::Complete
        } else {
            VectorCoverage::Missing
        }
    }

    fn take_inputs(
        &mut self,
        config: &Config,
        tokenizer: &mut PlanningTokenizer,
    ) -> Result<ChunkInputs> {
        // Taking inputs leaves the plan rematerializable from the same snapshot.
        // Deferred inputs say nothing about whether its vectors are complete.
        match std::mem::replace(&mut self.materialization, Materialization::Deferred) {
            Materialization::Materialized(inputs) => Ok(inputs),
            Materialization::Deferred => {
                Ok(tokenizer.materialize(config, self.selection, self.index)?.1)
            }
        }
    }
}

fn document_plan<'a>(
    config: &Config,
    selection: &'a Selection,
    index: &'a ReadIndex,
    tokenizer: &mut PlanningTokenizer,
) -> Result<DocumentPlan<'a>> {
    let version = index.version();
    let (documents, materialization) =
        if let Some(documents) = index::documents(config, &selection.key, &version)? {
            (documents, Materialization::Deferred)
        } else {
            let (documents, inputs) = tokenizer.materialize(config, selection, index)?;
            (documents, Materialization::Materialized(inputs))
        };
    let plan = DocumentPlan {
        selection,
        index,
        version,
        documents,
        materialization,
    };
    Ok(plan)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct SelectionIdentity(String);

impl SelectionIdentity {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct Selection {
    key: SelectionIdentity,
    executor: String,
    model: String,
    directory: Option<PathBuf>,
}

impl Selection {
    fn from_config(config: &Config) -> Result<Self> {
        ensure!(
            config.semantic(),
            "semantic retrieval is disabled; run trwl semantic enable"
        );
        let executor = config
            .string("semantic", "executor")
            .unwrap_or("builtin")
            .to_owned();
        let model = validate_selection(&executor, config.string("semantic", "model"))?.to_owned();
        let mut files_hash = String::new();
        let directory = if executor == "builtin" {
            let directory = std::env::var_os("TRWL_SEMANTIC_MODEL_DIR")
                .map(PathBuf::from)
                .or_else(|| option_env!("TRWL_SEMANTIC_MODEL_DIR").map(PathBuf::from))
                .context("no built-in model supplied; set TRWL_SEMANTIC_MODEL_DIR")?;
            let directory = directory
                .canonicalize()
                .with_context(|| format!("opening model directory {}", directory.display()))?;
            // Model files can change in place. Include their contents in the
            // cache key so vectors from different models are never mixed.
            files_hash = model_hash(config, &directory)?;
            Some(directory)
        } else {
            None
        };
        Ok(Self {
            key: SelectionIdentity(digest(&[INPUT_VERSION, &executor, &model, &files_hash])),
            executor,
            model,
            directory,
        })
    }
}

enum Executor {
    Shared(server::Client),
    Builtin {
        session: BuiltinSession,
        tokenizer: Box<Tokenizer>,
    },
    Http {
        agent: ureq::Agent,
        endpoint: String,
        model: String,
        token: Option<String>,
    },
}

impl Executor {
    fn shared(config: &Config, selection: &Selection) -> Result<Self> {
        if selection.directory.is_some() {
            Ok(Self::Shared(server::Client::new(
                &config.directories,
                selection,
            )))
        } else {
            let agent = ureq::Agent::config_builder()
                .max_redirects(0)
                .proxy(None)
                .timeout_connect(Some(HTTP_CONNECT_TIMEOUT))
                .timeout_send_request(Some(HTTP_IO_TIMEOUT))
                .timeout_send_body(Some(HTTP_IO_TIMEOUT))
                .timeout_recv_response(Some(HTTP_IO_TIMEOUT))
                .timeout_recv_body(Some(HTTP_IO_TIMEOUT))
                .build()
                .into();
            Ok(Self::Http {
                agent,
                endpoint: format!("{}/embeddings", selection.executor.trim_end_matches('/')),
                model: selection.model.clone(),
                token: std::env::var("TRWL_SEMANTIC_EMBEDDING_TOKEN").ok(),
            })
        }
    }

    fn builtin(directory: &Path, environment: &Environment) -> Result<Self> {
        let session = builtin_session(environment, &directory.join("onnx/model.onnx"), None)
            .with_context(|| format!("loading model in {}", directory.display()))?;
        Ok(Self::Builtin {
            session,
            tokenizer: Box::new(inference_tokenizer(directory)?),
        })
    }

    fn accelerated(selection: &Selection) -> Result<Option<(Self, Accelerator)>> {
        let directory = selection
            .directory
            .as_ref()
            .context("worker requires a built-in model")?;
        let environment = load_runtime()?;
        let tokenizer = inference_tokenizer(directory)?;
        // GetAvailableProviders does not initialize CUDA.
        let mut providers = Vec::new();
        if ort::ep::CUDA::is_available()? {
            providers.push((Accelerator::Cuda, ort::ep::CUDA::default().build()));
        }
        if intel_render_device_available() && ort::ep::OpenVINO::is_available()? {
            providers.push((
                Accelerator::OpenvinoGpu,
                ort::ep::OpenVINO::default()
                    .with_device_type("GPU")
                    // Intel's OpenCL compiler can crash in
                    // SPIRV::OCLToSPIRVBase during parallel compilation.
                    .with_arbitrary_config(
                        "load_config",
                        json!({"GPU": {"COMPILATION_NUM_THREADS": "1"}}).to_string(),
                    )
                    .build(),
            ));
        }
        for (device, provider) in providers {
            match builtin_session(
                &environment,
                &directory.join("onnx/model.onnx"),
                Some(provider),
            ) {
                Ok(session) => {
                    return Ok(Some((
                        Self::Builtin {
                            session,
                            tokenizer: Box::new(tokenizer),
                        },
                        device,
                    )));
                }
                Err(error) => {
                    eprintln!(
                        "trwl: {} embeddings unavailable: {error:#}",
                        device.as_str()
                    )
                }
            }
        }
        Ok(None)
    }

    fn embed(&mut self, texts: &[String], prefix: &str) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut vectors = match self {
            Self::Shared(client) => return client.embed(texts, prefix),
            Self::Http {
                agent,
                endpoint,
                model,
                token,
            } => {
                let mut request = agent.post(endpoint.as_str());
                if let Some(token) = token {
                    request = request.header("Authorization", format!("Bearer {token}"));
                }
                let mut response = request
                    .send_json(json!({"model": model, "input": texts, "encoding_format": "float"}))
                    .context("HTTP embedding request failed")?;
                // Embedding dimensions are server-defined; do not impose read_json's size limit.
                let response: Value =
                    serde_json::from_reader(response.body_mut().with_config().reader())
                        .context("invalid embedding response JSON")?;
                http_vectors(&response, texts.len())?
            }
            Self::Builtin { session, tokenizer } => {
                let encoded = tokenizer
                    .encode_batch(
                        texts
                            .iter()
                            .map(|s| format!("{prefix}{s}"))
                            .collect::<Vec<_>>(),
                        true,
                    )
                    .map_err(|e| anyhow::anyhow!("tokenizing input: {e}"))?;
                let width = encoded[0].get_ids().len();
                let ids: Vec<i64> = encoded
                    .iter()
                    .flat_map(|e| e.get_ids().iter().map(|&n| n as i64))
                    .collect();
                let mask: Vec<i64> = encoded
                    .iter()
                    .flat_map(|e| e.get_attention_mask().iter().map(|&n| n as i64))
                    .collect();
                let types: Vec<i64> = encoded
                    .iter()
                    .flat_map(|e| e.get_type_ids().iter().map(|&n| n as i64))
                    .collect();
                session.batch(|session| {
                    let mut inputs = ort::inputs![
                        "input_ids" => Tensor::from_array(([texts.len(), width], ids))?,
                        "attention_mask" => Tensor::from_array(([texts.len(), width], mask.clone()))?,
                    ];
                    if session
                        .inputs()
                        .iter()
                        .any(|i| i.name() == "token_type_ids")
                    {
                        inputs.push((
                            "token_type_ids".into(),
                            Tensor::from_array(([texts.len(), width], types))?.into(),
                        ));
                    }
                    let outputs = session.run(inputs)?;
                    let (shape, hidden) = outputs[0].try_extract_tensor::<f32>()?;
                    ensure!(
                        shape.len() == 3
                            && shape[0] as usize == texts.len()
                            && shape[1] as usize == width,
                        "unexpected E5 output shape {shape:?}"
                    );
                    let dimensions = shape[2] as usize;
                    let mut result = Vec::with_capacity(texts.len());
                    for batch in 0..texts.len() {
                        let mut vector = vec![0.0; dimensions];
                        let mut count = 0;
                        for token in 0..width {
                            if mask[batch * width + token] == 0 {
                                continue;
                            }
                            count += 1;
                            let start = (batch * width + token) * dimensions;
                            for (value, hidden) in
                                vector.iter_mut().zip(&hidden[start..start + dimensions])
                            {
                                *value += hidden;
                            }
                        }
                        ensure!(count > 0, "tokenizer produced an empty attention mask");
                        for value in &mut vector {
                            *value /= count as f32;
                        }
                        result.push(vector);
                    }
                    drop(outputs);
                    Ok(result)
                })?
            }
        };
        let dimensions = vectors[0].len();
        for vector in &mut vectors {
            ensure!(
                vector.len() == dimensions,
                "inconsistent embedding dimensions"
            );
            normalize(vector)?;
        }
        Ok(vectors)
    }
}

fn intel_render_device_available() -> bool {
    let Ok(nodes) = fs::read_dir("/dev/dri") else {
        return false;
    };
    nodes.filter_map(Result::ok).any(|node| {
        node.file_name()
            .to_str()
            .is_some_and(|name| name.starts_with("renderD"))
            && node
                .metadata()
                .is_ok_and(|metadata| metadata.file_type().is_char_device())
            && fs::read_to_string(
                Path::new("/sys/class/drm")
                    .join(node.file_name())
                    .join("device/vendor"),
            )
            .is_ok_and(|vendor| vendor.trim() == "0x8086")
    })
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Accelerator {
    Cuda,
    OpenvinoGpu,
}

impl Accelerator {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::OpenvinoGpu => "openvino-gpu",
        }
    }
}

fn load_runtime() -> Result<Environment> {
    let runtime = std::env::var("ORT_DYLIB_PATH")
        .ok()
        .or_else(|| option_env!("ORT_LIB_PATH").map(|path| format!("{path}/libonnxruntime.so")))
        .context("ONNX runtime library path is unavailable")?;
    Ok(ort::init_from(&runtime)?.build()?)
}

fn inference_tokenizer(directory: &Path) -> Result<Tokenizer> {
    let mut tokenizer = load_tokenizer(directory)?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MODEL_MAX_TOKENS,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("configuring tokenizer: {e}"))?;
    let pad_id = tokenizer
        .token_to_id("<pad>")
        .context("E5 tokenizer has no <pad> token")?;
    tokenizer.with_padding(Some(PaddingParams {
        pad_id,
        pad_token: "<pad>".into(),
        ..Default::default()
    }));
    Ok(tokenizer)
}

struct BuiltinSession {
    session: Session,
    profile_pending: bool,
}

impl BuiltinSession {
    fn batch<T>(&mut self, operation: impl FnOnce(&mut Session) -> Result<T>) -> Result<T> {
        let result = operation(&mut self.session)?;
        if self.profile_pending {
            // Flush the first batch before replying: the GPU worker is
            // killed after the build, so session destruction cannot do it.
            self.session
                .end_profiling()
                .context("flushing ONNX execution profile")?;
            self.profile_pending = false;
        }
        Ok(result)
    }
}

fn builtin_session(
    environment: &Environment,
    model: &Path,
    provider: Option<ExecutionProviderDispatch>,
) -> Result<BuiltinSession> {
    let mut builder = Session::builder(environment)?
        .with_intra_threads(INFERENCE_THREADS)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(1) // Execute graph nodes sequentially; parallelize each node.
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_op_spinning(false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_op_spinning(false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(provider) = provider {
        builder = builder
            .with_execution_providers([provider.error_on_failure()])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let profile_directory = std::env::var_os("TRWL_SEMANTIC_PROFILE_DIR");
    if let Some(directory) = &profile_directory {
        builder = builder
            .with_profiling(Path::new(directory).join(format!("ort-{}", std::process::id())))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    Ok(BuiltinSession {
        session: builder.commit_from_file(model)?,
        profile_pending: profile_directory.is_some(),
    })
}

fn http_vectors(value: &Value, count: usize) -> Result<Vec<Vec<f32>>> {
    let data = value["data"]
        .as_array()
        .context("embedding response has no data array")?;
    ensure!(
        data.len() == count,
        "embedding response count does not match input"
    );
    let mut vectors = vec![None; count];
    for item in data {
        let index = item["index"]
            .as_u64()
            .context("embedding response has no input index")? as usize;
        ensure!(
            index < count && vectors[index].is_none(),
            "duplicate or invalid embedding response index"
        );
        let vector: Vec<f32> = serde_json::from_value(item["embedding"].clone())
            .context("embedding is not a float vector")?;
        vectors[index] = Some(vector);
    }
    vectors
        .into_iter()
        .map(|vector| vector.context("embedding response omitted an input"))
        .collect()
}

fn normalize(vector: &mut [f32]) -> Result<()> {
    ensure!(
        !vector.is_empty() && vector.iter().all(|v| v.is_finite()),
        "empty or nonfinite embedding"
    );
    let norm = vector
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm > 0.0 && norm.is_finite(),
        "zero or nonfinite embedding norm"
    );
    for value in vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

/// Pair each user message with the last nonempty reply before the next user.
fn documents(records: &[ConversationMessage]) -> Vec<(&Record, String)> {
    let mut messages: Vec<_> = records
        .iter()
        .map(|record| (record, record.text.clone()))
        .collect();
    messages.sort_by(|(a, _), (b, _)| {
        a.session_id.cmp(&b.session_id).then_with(|| {
            crate::record::chronological_order(
                (&a.occurred_at, a.sequence, &a.event_id),
                (&b.occurred_at, b.sequence, &b.event_id),
            )
        })
    });
    // OpenCode emits one record per part. Merge visible parts before pairing
    // turns, keeping the first part as the result's event reference.
    let mut merged: Vec<(&ConversationMessage, String)> = Vec::new();
    let mut positions = HashMap::new();
    for (record, text) in messages {
        if let Some(message) = &record.provider_message_id {
            let key = (&record.session_id, &record.source, message);
            if let Some(&position) = positions.get(&key) {
                let (_, combined): &mut (&ConversationMessage, String) = &mut merged[position];
                if !combined.is_empty() && !text.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&text);
                continue;
            }
            positions.insert(key, merged.len());
        }
        merged.push((record, text));
    }
    let mut result = Vec::new();
    let mut turns = TurnAccumulator::default();
    // Walking backwards keeps the last nonempty reply before the next user.
    for (record, text) in merged.iter().rev() {
        if let Some(document) = turns.push(record, text) {
            result.push(document);
        }
    }
    result
}

#[derive(Default)]
struct TurnAccumulator<'record, 'text> {
    session: &'record str,
    reply: Option<&'text str>,
}

impl<'record, 'text> TurnAccumulator<'record, 'text> {
    fn push(
        &mut self,
        record: &'record ConversationMessage,
        text: &'text str,
    ) -> Option<(&'record Record, String)> {
        if record.session_id != self.session {
            self.session = &record.session_id;
            self.reply = None;
        }
        let text = text.trim();
        if let Some(request) = &record.request {
            let reply = self.reply.take();
            // These user-role records are harness notices, not requests.
            if !text.is_empty() && !crate::providers::is_control_message(text) {
                let mut document = format!("user:\n{text}");
                if let Some(reply) = reply {
                    document.push_str(&format!("\n\nassistant:\n{reply}"));
                }
                return Some((request, document));
            }
        } else if !text.is_empty() && self.reply.is_none() {
            self.reply = Some(text);
        }
        None
    }
}

fn chunks(text: &str) -> Vec<&str> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    // Slice at character boundaries, including the final short chunk.
    let offsets: Vec<_> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .take(DOCUMENT_MAX_CHARS + 1)
        .collect();
    let mut result = Vec::new();
    let mut start = 0;
    while start + 1 < offsets.len() {
        let end = (start + CHUNK_CHARS).min(offsets.len() - 1);
        let chunk = &text[offsets[start]..offsets[end]];
        if !chunk.trim().is_empty() {
            result.push(chunk);
        }
        if end == offsets.len() - 1 {
            break;
        }
        start = end - CHUNK_OVERLAP_CHARS;
    }
    result
}

fn document_chunks<'a>(
    text: &'a str,
    tokenizer: Option<&Tokenizer>,
) -> Result<Vec<(&'a str, usize)>> {
    let Some(tokenizer) = tokenizer else {
        // HTTP executors do not expose their tokenizer.
        return Ok(chunks(text)
            .into_iter()
            .map(|text| (text, text.chars().count()))
            .collect());
    };
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let cap = text
        .char_indices()
        .nth(DOCUMENT_MAX_CHARS)
        .map_or(text.len(), |(i, _)| i);
    let text = &text[..cap];
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizing document: {e}"))?;
    let offsets = encoding.get_offsets();
    let prefix = tokenizer
        .encode(DOCUMENT_PREFIX, true)
        .map_err(|e| anyhow::anyhow!("tokenizing prefix: {e}"))?
        .len();
    let budget = MODEL_MAX_TOKENS.saturating_sub(prefix + 2);
    ensure!(budget > 0, "retrieval prefix exceeds model token budget");
    let overlap = budget / 4;
    let mut start = 0;
    let mut result = Vec::new();
    while start < offsets.len() {
        let from = if start == 0 { 0 } else { offsets[start].0 };
        let mut end = (start + budget).min(offsets.len());
        loop {
            let to = if end == offsets.len() {
                text.len()
            } else {
                offsets[end].0
            };
            let chunk = text
                .get(from..to)
                .context("invalid tokenizer byte offsets")?;
            let length = tokenizer
                .encode(format!("{DOCUMENT_PREFIX}{chunk}"), true)
                .map_err(|e| anyhow::anyhow!("tokenizing chunk: {e}"))?
                .len();
            if length <= MODEL_MAX_TOKENS {
                if !chunk.trim().is_empty() {
                    result.push((chunk, length));
                }
                break;
            }
            ensure!(end > start + 1, "one token exceeds model budget");
            end -= 1;
        }
        if end == offsets.len() {
            break;
        }
        start = end.saturating_sub(overlap).max(start + 1);
    }
    Ok(result)
}

fn prepare_chunks(texts: &[String], tokenizer: Option<&Tokenizer>) -> Result<Vec<Vec<ChunkInput>>> {
    // Padding follows token count, which can differ for equal-length text.
    // Keep the length measured with inference's prefix during the budget check.
    // Discard encodings rather than retaining the whole corpus's tokens in memory.
    texts
        .iter()
        .map(|text| {
            Ok(document_chunks(text, tokenizer)?
                .into_iter()
                .map(|(text, tokens)| ChunkInput {
                    text: text.to_owned(),
                    tokens,
                })
                .collect())
        })
        .collect()
}

fn ordered_inputs(inputs: ChunkInputs) -> Vec<(String, ChunkInput)> {
    let mut inputs: Vec<_> = inputs.into_iter().collect();
    inputs.sort_by_key(|(_, input)| input.tokens);
    inputs
}

fn heavy_incremental(inputs: &[(String, ChunkInput)]) -> bool {
    // Sorted batches pay for padding to their longest input. The work budget
    // amortizes worker/model startup; device costs vary, so it is not a
    // hardware-independent latency prediction.
    inputs
        .chunks(BATCH_SIZE)
        .map(|batch| {
            batch
                .iter()
                .map(|(_, input)| input.tokens)
                .max()
                .unwrap_or(0)
                .saturating_mul(batch.len())
        })
        .fold(0usize, usize::saturating_add)
        >= 16_384
}

fn fill(
    config: &Config,
    plan: &mut DocumentPlan<'_>,
    tokenizer: &mut PlanningTokenizer,
    executor: &mut Option<Executor>,
) -> Result<Cache> {
    let _lock = lock(&config.directories.cache, "semantic.lock")?;
    // A query can wait behind a newer fill. Its pinned inputs still need
    // embeddings, but must not replace the newer persistent snapshot.
    let publication =
        if store::semantic_version(&config.directories.data)?.as_ref() == Some(&plan.version) {
            Publication::Current
        } else {
            Publication::Local
        };
    fill_locked(
        config,
        plan,
        tokenizer,
        executor,
        FillMode::Query,
        publication,
    )
}

#[derive(PartialEq, Eq)]
enum Publication {
    Current,
    Local,
}

#[derive(PartialEq, Eq)]
enum FillMode {
    Query,
    Update,
}

enum FillBackend {
    Gpu(worker::Client),
    Cpu,
}

enum FillProgress<'a> {
    Gpu(worker::Pending<&'a [(String, ChunkInput)]>),
    Cpu(worker::Batch<&'a [(String, ChunkInput)]>),
    Finished(FillBackend),
}

impl FillBackend {
    fn submit(self, inputs: &[(String, ChunkInput)]) -> FillProgress<'_> {
        let batch = worker::Batch {
            texts: inputs.iter().map(|(_, input)| input.text.clone()).collect(),
            association: inputs,
        };
        match self {
            Self::Gpu(client) => FillProgress::Gpu(client.submit(batch)),
            Self::Cpu => FillProgress::Cpu(batch),
        }
    }
}

fn fill_locked(
    config: &Config,
    plan: &mut DocumentPlan<'_>,
    tokenizer: &mut PlanningTokenizer,
    executor: &mut Option<Executor>,
    mode: FillMode,
    publication: Publication,
) -> Result<Cache> {
    let selection = plan.selection;
    if publication == Publication::Current {
        index::publish_mapping(config, plan)?;
    }
    let mut cache = index::load(config, &selection.key)?;
    if matches!(plan.vector_coverage(&cache), VectorCoverage::Complete) {
        if publication == Publication::Current {
            index::mark_build_complete(config, plan)?;
        }
        return Ok(cache);
    }
    let inputs = plan.take_inputs(config, tokenizer)?;
    let missing = inputs
        .into_iter()
        .filter(|(key, _)| !cache.vectors.contains_key(key))
        .collect();
    let missing = ordered_inputs(missing);
    let worker = if mode == FillMode::Update
        && selection.directory.is_some()
        && (index::build_history(config, &selection.key)?.needs_full_build()
            || heavy_incremental(&missing))
        && std::env::var_os("TRWL_SEMANTIC_CPU_ONLY").is_none_or(|v| v != "1")
    {
        worker::Client::start(&config.directories, selection)?
    } else {
        None
    };
    let backend = match worker {
        Some(client) => FillBackend::Gpu(client),
        None => FillBackend::Cpu,
    };
    let mut batches = missing.chunks(BATCH_SIZE);
    let mut progress = match batches.next() {
        Some(batch) => backend.submit(batch),
        None => FillProgress::Finished(backend),
    };
    // Keep the ready worker through final publication and completion marking.
    let _backend = loop {
        let (backend, batch, accelerated) = match progress {
            FillProgress::Gpu(operation) => match operation.receive()? {
                worker::BatchOutcome::Completed {
                    client,
                    batch,
                    vectors,
                } => (FillBackend::Gpu(client), batch, Some(vectors)),
                worker::BatchOutcome::CpuRecovery(batch) => {
                    // Recovery has reaped the GPU process. This transition is
                    // irreversible for this fill, including after partial progress.
                    (FillBackend::Cpu, batch, None)
                }
            },
            FillProgress::Cpu(batch) => (FillBackend::Cpu, batch, None),
            FillProgress::Finished(backend) => break backend,
        };
        let vectors = match accelerated {
            Some(vectors) => vectors,
            None => {
                if executor.is_none() {
                    *executor = Some(Executor::shared(config, selection)?);
                }
                executor
                    .as_mut()
                    .unwrap()
                    .embed(&batch.texts, DOCUMENT_PREFIX)?
            }
        };
        let completed = Cache {
            vectors: batch
                .association
                .iter()
                .map(|(key, _)| key.clone())
                .zip(vectors)
                .collect(),
        };
        // Let the GPU compute the next batch while this one commits. No cache
        // transaction is held while submitting or waiting for inference.
        progress = match batches.next() {
            Some(batch) => backend.submit(batch),
            None => FillProgress::Finished(backend),
        };
        if publication == Publication::Current {
            index::publish_batch(config, plan, &completed)?;
        }
        cache.vectors.extend(completed.vectors);
    };
    ensure!(
        matches!(plan.vector_coverage(&cache), VectorCoverage::Complete),
        "semantic fill is missing required embeddings"
    );
    if publication == Publication::Current {
        index::mark_build_complete(config, plan)?;
    }
    Ok(cache)
}

pub fn refresh(config: &Config) -> Result<()> {
    if !config.semantic() {
        return Ok(());
    }
    Planner::default().refresh(config)
}

/// One vocabulary per updater; prepared history remains in the semantic index.
#[derive(Default)]
pub struct Planner {
    tokenizer: PlanningTokenizer,
}

impl Planner {
    pub fn selection_key(config: &Config) -> Result<SelectionIdentity> {
        Ok(Selection::from_config(config)?.key)
    }

    pub fn covered(
        config: &Config,
        selection: &SelectionIdentity,
        version: &store::ConversationRevision,
    ) -> Result<bool> {
        index::covered(config, selection, version)
    }

    pub fn refresh(&mut self, config: &Config) -> Result<()> {
        refresh_inner(config, false, &mut self.tokenizer)
    }
}

pub fn rebuild(config: &Config) -> Result<()> {
    ensure!(
        config.semantic(),
        "semantic retrieval is disabled; run trwl semantic enable"
    );
    refresh_inner(config, true, &mut PlanningTokenizer::default())
}

fn refresh_inner(config: &Config, rebuild: bool, tokenizer: &mut PlanningTokenizer) -> Result<()> {
    let _lock = lock(&config.directories.cache, "semantic.lock")?;
    let Some(index) = ReadIndex::open(&config.directories.data)? else {
        return Ok(());
    };
    let selection = Selection::from_config(config)?;
    // Repeating an interrupted rebuild resumes its committed batches.
    if rebuild {
        index::begin_rebuild(config, &selection.key)?;
    }
    // Chunking and batch ordering share a vocabulary, loaded only on a cache miss.
    let mut plan = document_plan(config, &selection, &index, tokenizer)?;
    fill_locked(
        config,
        &mut plan,
        tokenizer,
        &mut None,
        FillMode::Update,
        Publication::Current,
    )?;
    Ok(())
}

pub fn status(config: &Config) -> Result<SemanticStatus> {
    let active = is_locked(&config.directories.cache, "semantic.lock")?;
    let mut report = SemanticStatus {
        enabled: config.semantic(),
        active,
        executor: config
            .string("semantic", "executor")
            .unwrap_or("builtin")
            .into(),
        model: config
            .string("semantic", "model")
            .unwrap_or(BUILTIN_MODEL)
            .into(),
        state: State::Disabled,
        documents: None,
        chunks: None,
        reason: None,
    };
    if !config.semantic() {
        return Ok(report);
    }
    if let Err(error) = coverage(config, &mut report) {
        report.state = State::Unavailable;
        report.reason = Some(format!("{error:#}"));
    }
    Ok(report)
}

fn coverage(config: &Config, report: &mut SemanticStatus) -> Result<()> {
    let Some(index) = ReadIndex::open(&config.directories.data)? else {
        report.state = State::Uninitialized;
        return Ok(());
    };
    let selection = Selection::from_config(config)?;
    let cache = index::load(config, &selection.key)?;
    let plan = document_plan(
        config,
        &selection,
        &index,
        &mut PlanningTokenizer::default(),
    )?;
    let mut keys = HashSet::new();
    let mut ready_documents = 0;
    let mut searchable_documents = 0;
    for document in &plan.documents {
        let document_keys = &document.keys;
        let ready = document_keys
            .iter()
            .filter(|key| cache.vectors.contains_key(*key))
            .count();
        ready_documents += usize::from(ready == document_keys.len());
        searchable_documents += usize::from(ready > 0);
        keys.extend(document_keys.iter().cloned());
    }
    // Old content and other model selections do not count toward coverage.
    let ready_chunks = keys
        .iter()
        .filter(|key| cache.vectors.contains_key(*key))
        .count();
    let pending = keys.len() - ready_chunks;
    report.state = if pending == 0 {
        State::Ready
    } else if report.active {
        State::CatchingUp
    } else {
        State::Pending
    };
    report.documents = Some(Coverage {
        total: plan.documents.len(),
        ready: ready_documents,
        searchable: Some(searchable_documents),
        pending: plan.documents.len() - ready_documents,
    });
    report.chunks = Some(Coverage {
        total: keys.len(),
        ready: ready_chunks,
        pending,
        searchable: None,
    });
    Ok(())
}

pub struct SearchResult {
    pub hits: Vec<(f32, Record)>,
    pub complete: bool,
}

pub fn search(
    config: &Config,
    index: &ReadIndex,
    terms: &[String],
    fill_missing: bool,
) -> Result<SearchResult> {
    // Without a cache directory, passive search cannot reuse any vectors. Check
    // this before hashing the built-in model files to determine cache identity.
    let empty = index.is_empty()?;
    if !fill_missing
        && !empty
        && config.semantic()
        && !config.directories.cache.join("semantic").try_exists()?
    {
        bail!("semantic index has no embeddings for current history; use --refresh wait");
    }
    let selection = Selection::from_config(config)?;
    let cached = if fill_missing {
        None
    } else {
        Some(index::load(config, &selection.key)?)
    };
    if empty {
        return Ok(SearchResult {
            hits: Vec::new(),
            complete: true,
        });
    }
    ensure!(
        cached
            .as_ref()
            .is_none_or(|cache| !cache.vectors.is_empty()),
        "semantic index has no embeddings for current history; use --refresh wait"
    );
    // Detect cache and query-executor failures before loading transcript payloads.
    // A failed semantic attempt must leave lexical fallback independent of their size.
    let mut executor = Some(Executor::shared(config, &selection)?);
    let queries = executor.as_mut().unwrap().embed(terms, QUERY_PREFIX)?;
    let mut tokenizer = if selection.directory.is_some() {
        PlanningTokenizer::shared(config, &selection)
    } else {
        PlanningTokenizer::default()
    };
    let mut plan = document_plan(config, &selection, index, &mut tokenizer)?;
    let cache = match cached {
        Some(cache) => cache,
        None => fill(config, &mut plan, &mut tokenizer, &mut executor)?,
    };
    let mut available = Vec::new();
    let mut complete = true;
    for document in &plan.documents {
        let record = &document.record;
        let keys = &document.keys;
        if keys.is_empty() {
            continue;
        }
        let vectors: Vec<_> = keys
            .iter()
            .filter_map(|key| cache.vectors.get(key))
            .collect();
        if vectors.len() != keys.len() {
            complete = false;
        }
        if !vectors.is_empty() {
            available.push((record, vectors));
        }
    }
    if available.is_empty() {
        if complete {
            return Ok(SearchResult {
                hits: Vec::new(),
                complete: true,
            });
        }
        bail!("semantic index has no embeddings for current history; use --refresh wait");
    }
    let mut results = Vec::with_capacity(available.len());
    for (record, vectors) in available {
        let mut best = f32::NEG_INFINITY;
        for vector in vectors {
            for query in &queries {
                ensure!(
                    vector.len() == query.len() && vector.iter().all(|x| x.is_finite()),
                    "cached embedding is incompatible with selected model"
                );
                let score = vector.iter().zip(query).map(|(a, b)| a * b).sum::<f32>();
                best = best.max(score);
            }
        }
        results.push((best, record.clone()));
    }
    Ok(SearchResult {
        hits: results,
        complete,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        record::Provider,
        test_support::{codex_message, import, source},
    };

    #[test]
    fn empty_refresh_publishes_and_completes_without_inference() {
        let api = crate::test_support::EmbeddingServer::start(|_| true);
        let dir = tempfile::tempdir().unwrap();
        let mut config =
            Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        config.document = format!(
            "[search]\nsemantic=true\n[semantic]\nexecutor='{}'\nmodel='test'\n",
            api.url
        )
        .parse()
        .unwrap();
        let path = dir.path().join("empty.jsonl");
        fs::write(
            &path,
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"empty\"}}\n",
        )
        .unwrap();
        let main = import(&config, Provider::Codex, &path);
        let selection = Selection::from_config(&config).unwrap();
        let mut planner = Planner::default();
        planner.refresh(&config).unwrap();
        assert!(Planner::covered(&config, &selection.key, &main.version()).unwrap());
        assert!(matches!(
            index::build_history(&config, &selection.key).unwrap(),
            index::BuildHistory::Completed
        ));
        assert!(
            index::documents(&config, &selection.key, &main.version())
                .unwrap()
                .unwrap()
                .is_empty()
        );
        planner.refresh(&config).unwrap();
        assert!(api.requests.lock().unwrap().is_empty());
    }

    #[test]
    fn pinned_search_and_waited_fill_preserve_the_newer_snapshot() {
        use crate::test_support::EmbeddingServer;
        use std::{sync::mpsc, thread};

        for replace_first in [false, true] {
            let api = EmbeddingServer::start(|_| true);
            let dir = tempfile::tempdir().unwrap();
            let mut config =
                Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
            config.document = format!(
                "[search]\nsemantic=true\n[semantic]\nexecutor='{}'\nmodel='test'\n",
                api.url
            )
            .parse()
            .unwrap();
            let path = dir.path().join("history.jsonl");
            let write = |count: usize, replace: bool| {
                let mut text =
                    "{\"type\":\"session_meta\",\"payload\":{\"id\":\"snapshot\"}}\n".to_owned();
                for n in 0..count {
                    text.push_str(
                        &codex_message(
                            "user",
                            &format!(
                                "message {n} {}",
                                if replace && n == 0 {
                                    "edited"
                                } else {
                                    "original"
                                }
                            ),
                        )
                        .to_string(),
                    );
                    text.push('\n');
                }
                fs::write(&path, text).unwrap();
                import(&config, Provider::Codex, &path)
            };
            let old = write(3, false);
            Planner::default().refresh(&config).unwrap();
            let selected = Selection::from_config(&config).unwrap();
            // The newer writer owns the fill lock while the old query waits.
            let lock = lock(&config.directories.cache, "semantic.lock").unwrap();
            let (done, result) = mpsc::channel();
            let (pinned, ready) = mpsc::channel();
            let (begin, started) = mpsc::channel();
            let query_config = config.clone();
            let reader = thread::spawn(move || {
                let old = ReadIndex::open(&query_config.directories.data)
                    .unwrap()
                    .unwrap();
                pinned.send(()).unwrap();
                started.recv().unwrap();
                done.send(search(&query_config, &old, &["query".into()], true))
                    .unwrap();
            });
            ready.recv_timeout(Duration::from_secs(5)).unwrap();
            let current = write(4, replace_first);
            let before = api.requests.lock().unwrap().len();
            begin.send(()).unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while api.requests.lock().unwrap().len() == before {
                assert!(std::time::Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            assert!(matches!(
                result.recv_timeout(Duration::from_millis(50)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            let mut plan = document_plan(
                &config,
                &selected,
                &current,
                &mut PlanningTokenizer::default(),
            )
            .unwrap();
            fill_locked(
                &config,
                &mut plan,
                &mut PlanningTokenizer::default(),
                &mut None,
                FillMode::Update,
                Publication::Current,
            )
            .unwrap();
            let database = config.directories.cache.join("semantic/cache.sqlite3");
            let observer = rusqlite::Connection::open(&database).unwrap();
            // Make every publication stage observable, including warm mapping
            // publication and completion updates that would change nothing.
            observer.execute_batch("UPDATE builds SET complete=0;
                CREATE TRIGGER no_mapping BEFORE INSERT ON selections BEGIN SELECT RAISE(ABORT, 'stale mapping'); END;
                CREATE TRIGGER no_vectors BEFORE INSERT ON vectors BEGIN SELECT RAISE(ABORT, 'stale vectors'); END;
                CREATE TRIGGER no_completion BEFORE UPDATE ON builds BEGIN SELECT RAISE(ABORT, 'stale completion'); END;").unwrap();
            let passive = search(&config, &old, &["query".into()], false).unwrap();
            assert_eq!(passive.complete, !replace_first);
            assert_eq!(passive.hits.len(), if replace_first { 2 } else { 3 });
            assert!(Planner::covered(&config, &selected.key, &current.version()).unwrap());
            assert!(status(&config).unwrap().state == State::Ready);
            drop(lock);
            let waited = result
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap();
            reader.join().unwrap();
            assert!(waited.complete);
            assert_eq!(waited.hits.len(), 3);
            assert!(
                waited
                    .hits
                    .iter()
                    .all(|(_, record)| !record.text.contains("edited"))
            );
            let disk = index::load(&config, &selected.key).unwrap();
            assert_eq!(disk.vectors.len(), 4);
            assert!(Planner::covered(&config, &selected.key, &current.version()).unwrap());
            let complete: bool = observer
                .query_row("SELECT complete FROM builds", [], |r| r.get(0))
                .unwrap();
            assert!(!complete);
            let latest = search(&config, &current, &["query".into()], false).unwrap();
            assert!(latest.complete);
            assert_eq!(latest.hits.len(), 4);
            let requests = api.requests.lock().unwrap();
            let documents: usize = requests
                .iter()
                .flatten()
                .filter(|text| text.starts_with("user:"))
                .count();
            assert_eq!(documents, if replace_first { 6 } else { 4 });
        }
    }

    pub fn write_cache(config: &Config, bytes: &[u8]) {
        let path = config.directories.cache.join("semantic").join(format!(
            "{}.msgpack",
            Selection::from_config(config).unwrap().key.as_str()
        ));
        atomic_write(&path, bytes).unwrap();
    }

    #[test]
    #[ignore = "requires the built-in model and ONNX Runtime with the OpenVINO provider"]
    fn builtin_providers_produce_equivalent_normalized_embeddings() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::edit(
            &crate::directories::Directories::isolated(dir.path()),
            |doc| {
                doc["search"]["semantic"] = toml_edit::value(true);
                Ok(())
            },
        )
        .unwrap();
        let selection = Selection::from_config(&config).unwrap();
        let environment = load_runtime().unwrap();
        let mut executor =
            Executor::builtin(selection.directory.as_ref().unwrap(), &environment).unwrap();
        let model = selection
            .directory
            .as_ref()
            .unwrap()
            .join("onnx/model.onnx");
        let texts = ["database connection pool".into(), "日本語の検索".into()];
        let cpu = executor.embed(&texts, QUERY_PREFIX).unwrap();
        let session = builtin_session(
            &environment,
            &model,
            Some(ort::ep::OpenVINO::default().with_device_type("CPU").build()),
        )
        .unwrap();
        let mut openvino = Executor::Builtin {
            session,
            tokenizer: Box::new(
                inference_tokenizer(selection.directory.as_ref().unwrap()).unwrap(),
            ),
        };
        let vectors = openvino.embed(&texts, QUERY_PREFIX).unwrap();
        assert_eq!(vectors.len(), texts.len());
        for (vector, expected) in vectors.iter().zip(&cpu) {
            assert_eq!(vector.len(), expected.len());
            let norm: f32 = vector.iter().map(|value| value * value).sum();
            assert!((norm - 1.0).abs() < 1e-5);
            let similarity: f32 = vector.iter().zip(expected).map(|(a, b)| a * b).sum();
            assert!(
                similarity > 0.999,
                "provider changed embedding: {similarity}"
            );
        }
        // Failed provider registration must reach the caller, not silently
        // construct a CPU session that the worker would report as accelerated.
        assert!(
            builtin_session(
                &environment,
                &model,
                Some(
                    ort::ep::OpenVINO::default()
                        .with_device_type("GPU.99999")
                        .build()
                )
            )
            .is_err()
        );
    }

    #[test]
    fn http_responses_reject_missing_or_duplicate_inputs() {
        let value = json!({"data": [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 0.0]}]});
        assert_eq!(
            http_vectors(&value, 2).unwrap(),
            vec![vec![3.0, 0.0], vec![0.0, 2.0]]
        );
        assert!(http_vectors(&json!({"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]}), 2).is_err());
        assert!(http_vectors(&value, 1).is_err());
        assert!(http_vectors(&value, 3).is_err());
    }

    #[test]
    fn model_hash_cache_reads_only_files_with_changed_identity() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let model = dir.path().join("model");
        fs::create_dir_all(model.join("onnx")).unwrap();
        for name in [
            "onnx/model.onnx",
            "tokenizer.json",
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ] {
            fs::write(model.join(name), "original".repeat(100)).unwrap();
        }
        HASHED_BYTES.with(|bytes| bytes.set(0));
        let first = model_hash(&config, &model).unwrap();
        assert_eq!(HASHED_BYTES.with(|bytes| bytes.get()), 4000);
        assert_eq!(model_hash(&config, &model).unwrap(), first);
        assert_eq!(HASHED_BYTES.with(|bytes| bytes.get()), 4000);
        // Restore mtime after a same-size edit: ctime must still invalidate it.
        let path = model.join("onnx/model.onnx");
        let modified = fs::metadata(&path).unwrap().modified().unwrap();
        fs::write(&path, "modified".repeat(100)).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        let edited = model_hash(&config, &model).unwrap();
        assert_ne!(first, edited);
        assert_eq!(HASHED_BYTES.with(|bytes| bytes.get()), 4800);
        // Replacing a file with identical bytes invalidates its inode stamp.
        let replacement = model.join("replacement");
        fs::write(&replacement, "modified".repeat(100)).unwrap();
        fs::rename(replacement, &path).unwrap();
        assert_eq!(model_hash(&config, &model).unwrap(), edited);
        assert_eq!(HASHED_BYTES.with(|bytes| bytes.get()), 5600);
    }

    #[test]
    fn builtin_chunks_use_tokens_and_cover_unicode_without_gaps() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [("[UNK]".to_owned(), 0), ("needle".to_owned(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        let text = "needle 日本語 🦀 ".repeat(2000);
        let parts = document_chunks(&text, Some(&tokenizer)).unwrap();
        assert!(
            parts
                .iter()
                .any(|(part, _)| part.chars().count() > CHUNK_CHARS)
        );
        let mut covered = 0;
        for (part, _) in parts {
            let start = part.as_ptr() as usize - text.as_ptr() as usize;
            assert!(start <= covered, "chunk gap at {start}");
            covered = covered.max(start + part.len());
            assert!(
                tokenizer
                    .encode(format!("{DOCUMENT_PREFIX}{part}"), true)
                    .unwrap()
                    .len()
                    <= MODEL_MAX_TOKENS
            );
        }
        assert_eq!(covered, text.len());
    }

    #[test]
    fn materialize_restores_inference_settings() {
        let (_dir, config, path) = crate::test_support::history_file("input.jsonl");
        let text = "needle 日本語 🦀 ".repeat(1000);
        let header = json!({"type":"session_meta", "payload":{"id":"native"}});
        let row = codex_message("user", &text);
        fs::write(&path, format!("{header}\n{row}\n")).unwrap();
        let index = import(&config, Provider::Codex, &path);
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[UNK]".to_owned(), 0)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        let records = index.messages().unwrap();
        let expected = materialize_documents(&records, Some(&tokenizer)).unwrap();
        assert!(expected.0[0].keys.len() > 1);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 8,
                ..Default::default()
            }))
            .unwrap();
        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(16),
            ..Default::default()
        }));
        let settings = tokenizer.to_string(false).unwrap();
        let actual = materialize(&index, Some(&mut tokenizer)).unwrap();
        assert_eq!(actual.0[0].keys, expected.0[0].keys);
        assert_eq!(actual.1, expected.1);
        assert_eq!(tokenizer.to_string(false).unwrap(), settings);

        // A failed tokenization must not leave inference unpadded/untruncated.
        tokenizer.with_model(tokenizers::models::wordlevel::WordLevel::default());
        let settings = tokenizer.to_string(false).unwrap();
        assert!(materialize(&index, Some(&mut tokenizer)).is_err());
        assert_eq!(tokenizer.to_string(false).unwrap(), settings);
    }

    #[test]
    fn planning_reuses_its_vocabulary_and_cached_maps_need_no_tokenizer_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let path = dir.path().join("input.jsonl");
        let header = json!({"type":"session_meta", "payload":{"id":"native"}});
        let row = codex_message("user", "database connection pool");
        fs::write(&path, format!("{header}\n{row}\n")).unwrap();
        let index = import(&config, Provider::Codex, &path);
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[UNK]".to_owned(), 0)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        let tokenizer_path = dir.path().join("tokenizer.json");
        tokenizer.save(&tokenizer_path, false).unwrap();
        let selection = Selection {
            key: SelectionIdentity("test".into()),
            executor: "builtin".into(),
            model: "test".into(),
            directory: Some(dir.path().into()),
        };
        let mut vocabulary = PlanningTokenizer::default();
        let mut plan = document_plan(&config, &selection, &index, &mut vocabulary).unwrap();
        index::publish_mapping(&config, &plan).unwrap();
        fs::remove_file(&tokenizer_path).unwrap();
        let inputs = ordered_inputs(plan.take_inputs(&config, &mut vocabulary).unwrap());
        let documents = &plan.documents;
        assert_eq!(inputs.len(), 1);
        let mut cached = document_plan(
            &config,
            &selection,
            &index,
            &mut PlanningTokenizer::default(),
        )
        .unwrap();
        assert!(matches!(cached.materialization, Materialization::Deferred));
        assert_eq!(cached.documents[0].keys, documents[0].keys);
        let vectors = Cache {
            vectors: [(documents[0].keys[0].clone(), vec![1.0, 0.0])].into(),
        };
        index::publish_batch(&config, &plan, &vectors).unwrap();
        let cached = fill(
            &config,
            &mut cached,
            &mut PlanningTokenizer::default(),
            &mut None,
        )
        .unwrap();
        assert_eq!(cached.vectors, vectors.vectors);

        // A different commit map forces planning, but unchanged composed input
        // is reusable even by a fresh planner with no tokenizer file available.
        let mut metadata = row.clone();
        metadata["timestamp"] = json!("2026-01-01T00:00:00Z");
        fs::write(&path, format!("{header}\n{metadata}\n")).unwrap();
        let other_index = import(&config, Provider::Codex, &path);
        assert_ne!(other_index.version(), index.version());
        let other = document_plan(
            &config,
            &selection,
            &other_index,
            &mut PlanningTokenizer::default(),
        )
        .unwrap();
        index::publish_mapping(&config, &other).unwrap();
        index::publish_batch(&config, &other, &vectors).unwrap();
        let reused = document_plan(
            &config,
            &selection,
            &index,
            &mut PlanningTokenizer::default(),
        )
        .unwrap();
        assert_eq!(reused.documents[0].keys, documents[0].keys);

        // An appended reply changes the earlier user's input, not its identity.
        let reply = codex_message("assistant", "limit the number of clients");
        fs::write(&path, format!("{header}\n{row}\n{reply}\n")).unwrap();
        let next = import(&config, Provider::Codex, &path);
        assert!(
            document_plan(
                &config,
                &selection,
                &next,
                &mut PlanningTokenizer::default()
            )
            .is_err()
        );
        let paired = document_plan(&config, &selection, &next, &mut vocabulary).unwrap();
        assert_ne!(paired.documents[0].keys, documents[0].keys);
        assert_eq!(
            paired.documents[0].record.event_id,
            documents[0].record.event_id
        );

        // Neither a previous selection's plans nor its loaded vocabulary may
        // satisfy a different model/tokenizer identity.
        let changed = Selection {
            key: SelectionIdentity("changed-tokenizer".into()),
            ..selection.clone()
        };
        assert!(document_plan(&config, &changed, &next, &mut vocabulary).is_err());

        // A failed initialization must retry when the tokenizer becomes available.
        tokenizer.save(&tokenizer_path, false).unwrap();
        let retried = document_plan(&config, &changed, &next, &mut vocabulary).unwrap();
        assert_eq!(
            retried.documents[0].prepared.as_ref().unwrap().chunks,
            paired.documents[0].prepared.as_ref().unwrap().chunks,
        );
    }

    #[test]
    fn document_plans_track_relevant_native_edits() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::edit(
            &crate::directories::Directories::isolated(dir.path()),
            |doc| {
                doc["search"]["semantic"] = toml_edit::value(true);
                doc["semantic"]["executor"] = toml_edit::value("http://127.0.0.1:1/v1");
                doc["semantic"]["model"] = toml_edit::value("test");
                Ok(())
            },
        )
        .unwrap();
        let path = dir.path().join("history.jsonl");
        let source = source(Provider::Claude, &path);
        let write = |user: &Value, assistant: Option<&Value>| {
            let mut lines = format!("{user}\n");
            let selected_only = json!({"type":"assistant", "uuid":"selected-only", "sessionId":"native",
                "message":{"role":"assistant", "content":[{"type":"redacted_thinking", "data":"opaque"}]}});
            lines.push_str(&format!("{selected_only}\n"));
            if let Some(assistant) = assistant {
                lines.push_str(&format!("{assistant}\n"));
            }
            fs::write(&path, lines).unwrap();
            assert!(
                crate::store::update(&config, std::slice::from_ref(&source), false)
                    .unwrap()
                    .completed()
            );
            ReadIndex::open(&config.directories.data).unwrap().unwrap()
        };
        let selection = Selection::from_config(&config).unwrap();
        let plan = |index: &ReadIndex| {
            index.record_decodes.set(0);
            let plan = document_plan(
                &config,
                &selection,
                index,
                &mut PlanningTokenizer::default(),
            )
            .unwrap();
            (plan.documents, plan.materialization)
        };
        let mut user = json!({
            "type":"user", "uuid":"11111111-1111-4111-8111-111111111111",
            "sessionId":"native", "timestamp":"2026-01-01T00:00:00.000Z",
            "cwd":"/workspace/first", "message":{"role":"user", "content":[
                {"type":"thinking", "thinking":"long selected prefix ".repeat(30)},
                {"type":"text", "text":"question"}
            ]}
        });
        let mut assistant = json!({
            "type":"assistant", "uuid":"22222222-2222-4222-8222-222222222222",
            "sessionId":"native", "timestamp":"2026-01-01T00:01:00.000Z",
            "cwd":"/workspace/first", "message":{"role":"assistant", "usage":{"input_tokens":10},
                "content":[
                    {"type":"thinking", "thinking":"private reasoning"},
                    {"type":"text", "text":"answer"},
                    {"type":"tool_use", "id":"tool", "name":"read", "input":{"path":"first"}}
                ]}
        });
        let index = write(&user, Some(&assistant));
        let assistant_id = index
            .all()
            .unwrap()
            .into_iter()
            .find(|record| {
                record.provider_event_id == assistant["uuid"].as_str().unwrap()
                    && record.is_conversation_message()
            })
            .unwrap()
            .event_id;
        let (first, inputs) = plan(&index);
        assert!(matches!(inputs, Materialization::Materialized(_)));
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].record.text, "question");
        assert!(first[0].record.source_spans.is_none());
        assert!(first[0].record.structured_content.is_null());
        assert_eq!(
            first[0].keys,
            vec![digest(&["user:\nquestion\n\nassistant:\nanswer"])]
        );
        assert_eq!(index.record_decodes.get(), 2);
        let initial = document_plan(
            &config,
            &selection,
            &index,
            &mut PlanningTokenizer::default(),
        )
        .unwrap();
        index::publish_mapping(&config, &initial).unwrap();

        assistant["message"]["content"][0]["thinking"] = json!("revised private reasoning");
        user["message"]["content"][0]["thinking"] = json!("revised selected user prefix");
        assistant["message"]["content"][2]["input"]["path"] = json!("second");
        assistant["message"]["usage"]["input_tokens"] = json!(20);
        let irrelevant = write(&user, Some(&assistant));
        assert_eq!(irrelevant.version(), index.version());
        let latest = irrelevant.event_reference(&assistant_id).unwrap();
        assert_eq!(latest.len(), 1);
        assert_eq!(latest[0].text, "answer");
        let mut expected = assistant["message"].clone();
        expected["content"].as_array_mut().unwrap().remove(0);
        assert_eq!(latest[0].structured_content, expected);
        let (cached, inputs) = plan(&irrelevant);
        assert!(matches!(inputs, Materialization::Deferred));
        assert_eq!(cached[0].keys, first[0].keys);
        assert_eq!(irrelevant.record_decodes.get(), 0);

        assistant["message"]["content"][1]["text"] = json!("edited answer");
        let visible = write(&user, Some(&assistant));
        assert_ne!(visible.version(), irrelevant.version());
        let (edited, inputs) = plan(&visible);
        assert!(matches!(inputs, Materialization::Materialized(_)));
        assert_eq!(edited.len(), 1);
        assert_eq!(
            edited[0].keys,
            vec![digest(&["user:\nquestion\n\nassistant:\nedited answer"])]
        );
        assert_eq!(visible.record_decodes.get(), 2);

        user["cwd"] = json!("/workspace/second");
        user["timestamp"] = json!("2026-01-01T00:00:30.000Z");
        let metadata = write(&user, Some(&assistant));
        assert_ne!(metadata.version(), visible.version());
        let (updated, inputs) = plan(&metadata);
        assert!(matches!(inputs, Materialization::Materialized(_)));
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].keys, edited[0].keys);
        assert_eq!(updated[0].record.workspace, "/workspace/second");
        assert_eq!(updated[0].record.occurred_at, "2026-01-01T00:00:30.000Z");
        assert_eq!(metadata.record_decodes.get(), 2);

        let deleted = write(&user, None);
        assert_ne!(deleted.version(), metadata.version());
        let (unanswered, inputs) = plan(&deleted);
        assert!(matches!(inputs, Materialization::Materialized(_)));
        assert_eq!(unanswered.len(), 1);
        assert_eq!(unanswered[0].keys, vec![digest(&["user:\nquestion"])]);
        assert_eq!(deleted.record_decodes.get(), 1);

        let (pinned, _) = plan(&irrelevant);
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0].keys, first[0].keys);
        assert_eq!(pinned[0].record.workspace, "/workspace/first");
        assert_eq!(pinned[0].record.occurred_at, "2026-01-01T00:00:00.000Z");
        assert_eq!(irrelevant.record_decodes.get(), 0);
    }

    #[test]
    fn opencode_parts_form_one_anchor_and_one_complete_reply_per_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("native.db");
        let db = crate::test_support::opencode_db(&path);
        db.execute_batch(
            "INSERT INTO message (id,session_id,data) VALUES
            ('user','session','{\"role\":\"user\"}'),
            ('reply','session','{\"role\":\"assistant\"}'),
            ('unanswered','session','{\"role\":\"user\"}');",
        )
        .unwrap();
        for (n, (message, kind, text)) in [
            ("user", "text", "first"),
            ("user", "text", "question"),
            ("reply", "reasoning", "private"),
            ("reply", "text", "first reply"),
            ("reply", "text", "last piece"),
            ("unanswered", "text", "unanswered"),
        ]
        .into_iter()
        .enumerate()
        {
            db.execute("INSERT INTO part (id,message_id,session_id,data,time_created) VALUES (?,?, 'session',?,?)",
                rusqlite::params![format!("part-{n}"), message, json!({"type":kind,"text":text}).to_string(),n as i64]).unwrap();
        }
        let config = Config::load(&crate::directories::Directories::isolated(dir.path())).unwrap();
        let index = import(&config, Provider::Opencode, &path);
        let messages = index.messages().unwrap();
        assert_eq!(messages.len(), 5);
        let docs = documents(&messages);
        assert_eq!(docs.len(), 2);
        let paired = docs
            .iter()
            .find(|(record, _)| record.provider_message_id.as_deref() == Some("user"))
            .unwrap();
        assert_eq!(
            paired.1,
            "user:\nfirst\nquestion\n\nassistant:\nfirst reply\nlast piece"
        );
        assert_eq!(paired.0.provider_event_id, "part-0");
        assert!(docs.iter().any(|(_, text)| text == "user:\nunanswered"));
    }

    #[test]
    fn unicode_chunks_cover_every_character_up_to_the_source_cap() {
        let corpus: String = "日本語 λ résumé 🦀"
            .chars()
            .cycle()
            .take(DOCUMENT_MAX_CHARS + 1)
            .collect();
        for length in [
            0,
            CHUNK_CHARS - 1,
            CHUNK_CHARS,
            CHUNK_CHARS + 1,
            DOCUMENT_MAX_CHARS + 1,
        ] {
            let text: String = corpus.chars().take(length).collect();
            let parts = chunks(&text);
            assert!(parts.iter().all(|part| part.chars().count() <= CHUNK_CHARS));
            let mut reconstructed = parts.first().copied().unwrap_or("").to_owned();
            for pair in parts.windows(2) {
                let previous: Vec<_> = pair[0].chars().collect();
                let overlap: Vec<_> = pair[1].chars().take(CHUNK_OVERLAP_CHARS).collect();
                assert_eq!(&previous[previous.len() - CHUNK_OVERLAP_CHARS..], overlap);
                reconstructed.extend(pair[1].chars().skip(CHUNK_OVERLAP_CHARS));
            }
            assert_eq!(
                reconstructed,
                text.chars().take(DOCUMENT_MAX_CHARS).collect::<String>(),
                "length {length}"
            );
        }
        assert!(chunks(" \n\t").is_empty());
    }

    #[test]
    fn invalid_vectors_fail() {
        for mut v in [vec![], vec![0.0], vec![f32::NAN], vec![f32::INFINITY]] {
            assert!(normalize(&mut v).is_err());
        }
        let mut v = vec![3.0, 4.0];
        normalize(&mut v).unwrap();
        assert_eq!(v, vec![0.6, 0.8]);
    }

    #[test]
    fn user_turns_pair_only_with_their_last_visible_reply() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.jsonl");
        let mut lines = String::new();
        for (session, role, content) in [
            ("first", "user", json!("question")),
            ("first", "assistant", json!("earlier")),
            (
                "first",
                "user",
                json!([{"type":"tool_result", "content":"tool output"}]),
            ),
            (
                "first",
                "assistant",
                json!([
                    {"type":"thinking", "thinking":"private reasoning"},
                    {"type":"text", "text":"last"},
                    {"type":"tool_use", "name":"read", "input":{"path":"file"}}
                ]),
            ),
            (
                "first",
                "assistant",
                json!([{"type":"thinking", "thinking":"more reasoning"}]),
            ),
            ("first", "assistant", json!("")),
            ("first", "user", json!("unanswered")),
            ("first", "user", json!("")),
            ("first", "assistant", json!("after empty user")),
            ("first", "user", json!("before control")),
            ("first", "user", json!("<turn_aborted>")),
            ("first", "assistant", json!("after control")),
            ("second", "assistant", json!("other session")),
        ] {
            let row = json!({"type":role, "sessionId":session,
                "message":{"role":role, "content":content}});
            lines.push_str(&format!("{row}\n"));
        }
        fs::write(&path, lines).unwrap();
        let mut records = Vec::new();
        crate::providers::parse(Provider::Claude, &path, None, |record| {
            if let Some(message) = ConversationMessage::new(record) {
                records.push(message);
            }
            Ok(())
        })
        .unwrap();
        let mut texts: Vec<_> = documents(&records)
            .into_iter()
            .map(|(_, text)| text)
            .collect();
        texts.sort();
        assert_eq!(
            texts,
            [
                "user:\nbefore control",
                "user:\nquestion\n\nassistant:\nlast",
                "user:\nunanswered"
            ]
        );
    }
}
