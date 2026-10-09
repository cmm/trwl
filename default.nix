{
  pkgsCuda,
  lib,
  rustPlatform,
  versionCheckHook,
  python3,
  git,
  time,
  intel-compute-runtime,
  stdenv,
  callPackage,
  trwlEmbeddingModelSource,
  makeWrapper,
}:
let
  model = callPackage ./model.nix { source = trwlEmbeddingModelSource; };
  cudaRuntime = pkgsCuda.onnxruntime;
  intelRuntimeSupported = lib.meta.availableOn stdenv.hostPlatform intel-compute-runtime;
in
rustPlatform.buildRustPackage {
  pname = "trwl";
  version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package.version;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./Cargo.toml
      ./Cargo.lock
      ./LICENSE
      ./skills
      ./src
      ./tests
    ];
  };
  cargoLock.lockFile = ./Cargo.lock;

  # Ignored tests need the built-in model and ONNX Runtime, which the
  # environment below provides.
  checkFlags = [ "--include-ignored" ];

  doInstallCheck = true;
  nativeBuildInputs = [ makeWrapper ];
  nativeInstallCheckInputs = [
    versionCheckHook
    python3
    git
    time
  ];
  buildInputs = [ cudaRuntime ];
  env.ORT_LIB_PATH = "${cudaRuntime}/lib";
  env.TRWL_SEMANTIC_MODEL_DIR = model;

  postInstall = lib.optionalString intelRuntimeSupported ''
    wrapProgram "$out/bin/trwl" \
      --set-default OCL_ICD_VENDORS "${intel-compute-runtime}/etc/OpenCL/vendors"
  '';

  postInstallCheck = ''
    patchShebangs tests
    tests/e2e.py --binary "$out/bin/trwl" \
      --case discovery --case retrieval --case selection --case workspaces --case updates \
      --case refresh-restart --case hybrid-grouping --case semantic-cache
    tests/test-perf-compare.py
    tests/test-perf-runner.py
    tests/test-realm-corpus.py
  '';

  passthru = {
    inherit model;
    inherit (model) modelMetadata;
    onnxruntime = cudaRuntime;
  };

  meta = {
    description = "Search coding agent history";
    license = lib.licenses.asl20;
    sourceProvenance = [ lib.sourceTypes.fromSource ];
    mainProgram = "trwl";
    platforms = lib.platforms.linux;
  };
}
