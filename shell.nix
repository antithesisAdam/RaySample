{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python310;
  libPaths = pkgs.lib.concatStringsSep ":" [
    "${pkgs.stdenv.cc.cc.lib}/lib"
    "${pkgs.zlib}/lib"
  ];
in
pkgs.mkShell {
  buildInputs = [
    python
    python.pkgs.pip
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${libPaths}:$LD_LIBRARY_PATH
    echo "🐍 Creating venv if not exists..."
    if [ ! -d "./venv" ]; then
      ${python.interpreter} -m venv venv
      source ./venv/bin/activate
      pip install --upgrade pip
      pip install numpy==1.26.4 torch ray gymnasium
    else
      source ./venv/bin/activate
    fi

    echo "✅ venv ready. Run: python py-pong.py"
  '';
}
