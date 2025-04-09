{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.virtualenv
    pkgs.gcc13
    pkgs.which
    pkgs.libffi
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=/nix/store/41vbjqryicfhkl5a81qkrc4sfjnif62s-gcc-13.3.0-lib/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$(pwd)/.venv/lib/python3.12/site-packages:$PYTHONPATH
    source .venv/bin/activate || true
  '';
}
