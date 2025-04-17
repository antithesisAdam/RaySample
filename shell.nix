{ pkgs ? import <nixpkgs> {} }:

let
  py = pkgs.python310;
  pypkgs = pkgs.python310Packages;
in

pkgs.mkShell {
  buildInputs = [
    # pre‑built Python packages
    pypkgs.numpy
    pypkgs.ray
    pypkgs.gymnasium
    pypkgs.shimmy
    pypkgs.ale_py

    # system libs for numpy C‑extensions
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  # so that numpy can find zlib & libstdc++
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "🐍  Ready! Just:"
    echo "    pip install --upgrade pip"
    echo "    pip install gymnasium[accept-rom-license] shimmy==1.2.0"
    echo "    export RAY_ADDRESS=127.0.0.1:6379"
    echo "    python py-pong-ray.py"
  '';
}
