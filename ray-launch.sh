#!/usr/bin/env bash
set -e

GCC_LIB="/nix/store/mhd0rk497xm0xnip7262xdw9bylvzh99-gcc-13.3.0-lib/lib"
LD_SO="/nix/store/wn7v2vhyyyi6clcyn0s9ixvl7d4d87ic-glibc-2.40-36/lib/ld-linux-x86-64.so.2"
PYTHON_BIN="$(pwd)/.venv/bin/python"

exec "$LD_SO" \
  --library-path "$GCC_LIB" \
  "$PYTHON_BIN" -m ray.scripts.scripts start --head --dashboard-host=0.0.0.0 "$@"
