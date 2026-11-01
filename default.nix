{
  pkgsCuda,
  onnxruntime,
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
  enableGpu ? true,
}:
let
  model = callPackage ./model.nix { source = trwlEmbeddingModelSource; };
  runtime = if enableGpu then pkgsCuda.onnxruntime else onnxruntime;
  intelRuntimeSupported = lib.meta.availableOn stdenv.hostPlatform intel-compute-runtime;
  meta = {
    description = "Search coding agent history";
    license = lib.licenses.asl20;
    sourceProvenance = [ lib.sourceTypes.fromSource ];
    mainProgram = "trwl";
    platforms = lib.platforms.linux;
  };
  # Runtime selection belongs to the wrappers so CPU and GPU packages share
  # the Rust build. Direct execution of the binary falls back to CPU inference.
  unwrapped = rustPlatform.buildRustPackage {
    pname = "trwl-unwrapped";
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

    # The ignored provider-equivalence test needs OpenVINO, which the ONNX
    # Runtime package in stable nixpkgs does not include.
    buildInputs = [ onnxruntime ];
    env.ORT_LIB_PATH = "${onnxruntime}/lib";
    env.TRWL_SEMANTIC_MODEL_DIR = model;
    inherit meta;
  };
in
stdenv.mkDerivation {
  pname = "trwl";
  inherit (unwrapped) version;
  dontUnpack = true;
  doInstallCheck = true;
  nativeBuildInputs = [ makeWrapper ];
  nativeInstallCheckInputs = [
    versionCheckHook
    python3
    git
    time
  ];
  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin"
    makeWrapper ${lib.getExe unwrapped} "$out/bin/trwl" \
      --set-default ORT_DYLIB_PATH "${runtime}/lib/libonnxruntime.so" \
      ${lib.optionalString (enableGpu && intelRuntimeSupported) ''
        --set-default OCL_ICD_VENDORS "${intel-compute-runtime}/etc/OpenCL/vendors"
      ''}
    runHook postInstall
  '';

  postInstallCheck = ''
    cp -r ${./tests} tests
    chmod -R u+w tests
    patchShebangs tests
    tests/e2e.py --binary "$out/bin/trwl" \
      --case discovery --case retrieval --case opencode-v2 --case selection --case workspaces --case updates \
      --case refresh-restart --case hybrid-grouping --case semantic-cache
    tests/test-perf-compare.py
    tests/test-perf-runner.py
    tests/test-realm-corpus.py
  '';

  passthru = {
    inherit model unwrapped;
    inherit (model) modelMetadata;
    onnxruntime = runtime;
  };

  inherit meta;
}
