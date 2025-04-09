#!/usr/bin/env bash

# Path to your local gcc libstdc++
export LD_LIBRARY_PATH="/nix/store/mhd0rk497xm0xnip7262xdw9bylvzh99-gcc-13.3.0-lib/lib:$LD_LIBRARY_PATH"

# Set PYTHONPATH to ensure ray inside .venv can see its modules
export PYTHONPATH="$(pwd)/.venv/lib/python3.12/site-packages:$PYTHONPATH"

# Use ray from your virtualenv
exec "$(pwd)/.venv/bin/ray" "$@"
