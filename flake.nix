{
  description = "Search coding agent history";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
