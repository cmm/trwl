# Installation

trwl has only been tested on Linux (NixOS and Ubuntu + home-manager).
It probably won't work on a non-Linux system out of the box.

## Contents

- [Nix](#nix)
- [Cargo](#cargo)
- [Agent skill](#agent-skill)

## Nix

The build requires a CUDA-enabled `onnxruntime`. This is not
configurable at the moment. To avoid building CUDA things and
`onnxruntime` locally, enable the [NixOS-CUDA community cache](https://cache.nixos-cuda.org/)
before building trwl.

With Nix's `nix-command` and `flakes` features enabled:

```sh
nix profile install .#trwl
```

To run without installing into your profile:

```sh
nix run .#trwl -- --help
```

The Nix installation includes ONNX Runtime and the built-in embedding model.

For Home Manager or NixOS, add a `trwl` input:

```nix
inputs.trwl.url = "github:cmm/trwl";
```

Add `inputs.trwl.packages.${pkgs.stdenv.hostPlatform.system}.trwl` to
`home.packages` or `environment.systemPackages`.

## Cargo

Install with a Rust toolchain and a C compiler:

```sh
cargo install --path . --locked
```

Add `$CARGO_HOME/bin` to `PATH`, or `~/.cargo/bin` if `CARGO_HOME` is unset.

For built-in embedding executor, install ONNX Runtime 1.24 or newer and the
[multilingual-E5-small model](https://huggingface.co/intfloat/multilingual-e5-small),
then supply their directories when installing trwl:

```sh
ORT_LIB_PATH=/path/to/onnxruntime/lib \
TRWL_SEMANTIC_MODEL_DIR=/path/to/model \
cargo install --path . --locked
```

The model directory contains `onnx/model.onnx`, `tokenizer.json`, `config.json`,
`special_tokens_map.json` and `tokenizer_config.json`. These files and the ONNX
Runtime library are needed at runtime.

At runtime, `ORT_DYLIB_PATH` selects the shared library file and
`TRWL_SEMANTIC_MODEL_DIR` selects the model directory. Lexical search and HTTP
embedding executor do not need local model files or ONNX Runtime.

## Agent skill

[skills/trwl/SKILL.md](skills/trwl/SKILL.md) contains the agent workflow. Copy the
`skills/trwl` directory into the agent's skill directory. For OpenCode, that is
`~/.config/opencode/skills/trwl` (or the corresponding `$XDG_CONFIG_HOME` path).
Restart OpenCode to load the installed skill.

With Home Manager and a flake input:

```nix
xdg.configFile."opencode/skills/trwl".source = "${inputs.trwl}/skills/trwl";
```
