# GPU usage by the built-in embedding executor

### Full builds

The first fill of a model selection embeds all history — whether
started by `semantic rebuild`, `setup --wait` or the background
updater, including after a model change or cache eviction. An
interrupted build resumes its completed batches when repeated.
`import --full` reparses history; it does not request a rebuild.

A build tries CUDA, then Intel OpenVINO GPU, then CPU; OpenVINO needs
an Intel render device, and the selected device is reported on
stderr. ONNX Runtime can run unsupported operations on CPU even when
a GPU is selected.

One per-user lock at `$XDG_RUNTIME_DIR/trwl/gpu.lock` coordinates GPU
builds between trwl processes across data roots and models; a
contended lock selects CPU without waiting. `TRWL_GPU_LOCK_DIR`
overrides the directory. When `XDG_RUNTIME_DIR` is unset, the fallback
is `<temporary-directory>/<uid>/trwl/gpu.lock`, using `TEMPDIR`, then
`TMPDIR`, then `/tmp` as the temporary directory and the effective user
ID as `<uid>`.

GPU errors, crashes and timeouts cause the build to finish on CPU; CPU
failures propagate. `SIGINT`, `SIGTERM` or `SIGHUP` during a GPU build
cancels it. `TRWL_SEMANTIC_CPU_ONLY=1` disables GPU usage
completely.
