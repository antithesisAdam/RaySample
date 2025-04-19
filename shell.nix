{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python310Full;   # full CPython with ensurepip/venv

in

pkgs.mkShell {
  buildInputs = [

    python
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  # so that numpy can find zlib & libstdc++
  shellHook = ''

    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

    # bootstrap your venv if it doesn't exist
    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    source .venv/bin/activate

    # install everything from requirements.txt in one go
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    echo "✅ venv ready – just:"

    echo "    export RAY_ADDRESS=127.0.0.1:6379"
    echo "    python py-pong-ray.py"
  '';
}