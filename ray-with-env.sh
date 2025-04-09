#!/usr/bin/env bash

# Point to working libstdc++
export LD_LIBRARY_PATH="/nix/store/mhd0rk497xm0xnip7262xdw9bylvzh99-gcc-13.3.0-lib/lib:$LD_LIBRARY_PATH"

# Ensure your venv site-packages are visible to the python call
VENV_SITE_PACKAGES="$(find .venv/lib -type d -name 'site-packages' | head -n 1)"
export PYTHONPATH="$VENV_SITE_PACKAGES:$PYTHONPATH"

# Use Ray's internal CLI entry point
exec .venv/bin/python -m ray.scripts.scripts "$@"
