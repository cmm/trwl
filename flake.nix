{
  description = "Search coding agent history";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    embedding-model = {
      url = "git+https://huggingface.co/intfloat/multilingual-e5-small?ref=main";
      flake = false;
    };
  };

  outputs =
    { nixpkgs, embedding-model, ... }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      packageFor =
        system:
        (pkgsFor system).callPackage ./default.nix {
          trwlEmbeddingModelSource = embedding-model;
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          trwl = packageFor system;
        in
        {
          inherit trwl;
          default = trwl;
        }
      );

      apps = forAllSystems (
        system:
        let
          trwl = packageFor system;
          app = {
            type = "app";
            program = nixpkgs.lib.getExe trwl;
            inherit (trwl) meta;
          };
        in
        {
          trwl = app;
          default = app;
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          trwl = (packageFor system).override { enableGpu = false; };
          python = pkgs.python3.withPackages (p: [
            p.numpy
            p.onnxruntime
            p.tokenizers
          ]);
        in
        {
          regression =
            pkgs.runCommand "trwl-regression"
              {
                nativeBuildInputs = [
                  python
                  pkgs.git
                  pkgs.time
                ];
              }
              ''
                # The harness writes artifacts beside tests and selects CPU inference.
                cp -r ${./tests} tests
                if ! python3 tests/e2e.py \
                  --binary ${pkgs.lib.getExe trwl} \
                  --model-dir ${trwl.model}; then
                  cat .cache/trwl-e2e-*/report.json
                  exit 1
                fi
                mkdir -p "$out"
                cp .cache/trwl-e2e-*/report.json "$out/report.json"
              '';
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          trwl = packageFor system;
        in
        {
          default =
            (pkgs.mkShell {
              inputsFrom = [ trwl ];
              packages = [
                pkgs.rustfmt
                pkgs.clippy
              ];
              inherit (trwl) ORT_LIB_PATH TRWL_SEMANTIC_MODEL_DIR;
            }).overrideAttrs
              {
                # mkShell's documentation lookup builds dependencies during evaluation.
                shellHook = "";
              };
        }
      );

      formatter = forAllSystems (system: (pkgsFor system).nixfmt);
    };
}
