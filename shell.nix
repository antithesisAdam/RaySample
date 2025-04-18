{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python310.withPackages (ps: with ps; [
    numpy
    ray
    gymnasium       # for gymnasium core
    shimmy          # for `shimmy.atari_env`
    ale-py          # for ALE/Pong ROM support
  ]);
in

pkgs.mkShell {
  # give us python + all of the above installed
  buildInputs = [
    pythonEnv
    pkgs.zlib       # C‐library for numpy/zlib support
  ];

  shellHook = ''
    # so numpy C‐exts can find zlib
    export LD_LIBRARY_PATH="${pkgs.zlib.lib}/lib:$LD_LIBRARY_PATH"

    echo "🐍  Ready!"
    echo "    export RAY_ADDRESS=127.0.0.1:6379"
    echo "    python py-pong-ray.py"
  '';
}
