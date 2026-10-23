# GPU usage by the built-in embedding executor

### Update fills

The first fill of a model selection embeds all history — whether
started by `semantic rebuild`, `setup --wait` or the background
updater, including after a model change or cache eviction. An
interrupted build resumes its completed batches when repeated.
`import --full` reparses history; it does not request a rebuild.

Incremental updates also attempt GPU when the remaining embedding work
reaches 16,384 padded tokens. Distinct uncached chunks are sorted by token
length and grouped into batches of up to 16. Each batch contributes its
chunk count multiplied by its longest chunk's token count, including the
embedding prefix. The threshold amortizes worker startup; it does not
guarantee a speedup on every device.
Updates with no missing chunks do not start a GPU worker.

Queries, including their waited fills of missing embeddings, use CPU.

An eligible fill tries CUDA, then Intel OpenVINO GPU, then CPU; OpenVINO needs
an Intel render device, and the selected device is reported on
stderr. ONNX Runtime can run unsupported operations on CPU even when
a GPU is selected.

One per-user lock at `$XDG_RUNTIME_DIR/trwl/gpu.lock` coordinates GPU
fills between trwl processes across data roots and models; a
contended lock selects CPU without waiting. `TRWL_GPU_LOCK_DIR`
overrides the directory. When `XDG_RUNTIME_DIR` is unavailable, the fallback
is `$TMPDIR/trwl-runtime-<uid>/trwl/gpu.lock`, using `/tmp` when `TMPDIR` is
unset and the effective user ID as `<uid>`.

GPU errors, crashes and timeouts cause the fill to finish on CPU; CPU
failures propagate. `SIGINT`, `SIGTERM` or `SIGHUP` during a GPU fill
cancels it. `TRWL_SEMANTIC_CPU_ONLY=1` disables GPU usage
completely.
