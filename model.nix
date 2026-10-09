{
  lib,
  fetchurl,
  runCommand,
  source,
}:
let
  lfsFile =
    path:
    let
      pointer = builtins.match "version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?" (
        builtins.readFile "${source}/${path}"
      );
      sha256 = builtins.elemAt pointer 0;
      size = builtins.fromJSON (builtins.elemAt pointer 1);
    in
    assert lib.assertMsg (pointer != null) "Invalid Git LFS pointer: ${path}";
    {
      inherit path sha256 size;
      src = fetchurl {
        url = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/${source.rev}/${path}";
        inherit sha256;
      };
    };

  ordinaryFile =
    path:
    let
      src = "${source}/${path}";
    in
    {
      inherit path src;
      sha256 = builtins.hashFile "sha256" src;
      size = builtins.stringLength (builtins.readFile src);
    };

  files = [
    (lfsFile "onnx/model.onnx")
    (lfsFile "tokenizer.json")
    (ordinaryFile "config.json")
    (ordinaryFile "special_tokens_map.json")
    (ordinaryFile "tokenizer_config.json")
  ];

  modelMetadata = builtins.toJSON {
    source_revision = source.rev;
    files = map (file: { inherit (file) path sha256 size; }) files;
  };
in
runCommand "trwl-embedding-model" { passthru = { inherit modelMetadata; }; } ''
  mkdir -p "$out/onnx"
  ${lib.concatMapStringsSep "\n" (
    file: "cp ${lib.escapeShellArg file.src} \"$out\"/${lib.escapeShellArg file.path}"
  ) files}
''
