#!/usr/bin/env bash
# entrypoint.sh
set -euo pipefail

# Decide role based on RAY_MODE (default to head)
if [ "${RAY_MODE:-head}" = "head" ]; then
  exec ray start --head \
       --port=6379 \
       --dashboard-host=0.0.0.0 \
       --dashboard-port=8265 \
       --disable-usage-stats \
       --block

elif [ "${RAY_MODE}" = "worker" ]; then
  # RAY_ADDRESS should be like "head-ip:6379"
  exec ray start --address="${RAY_ADDRESS:-127.0.0.1:6379}" \
       --disable-usage-stats \
       --block

else
  # Fallback: run any provided command
  exec "$@"
fi
 