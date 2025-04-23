#!/usr/bin/env bash
set -euo pipefail

echo "⏳ waiting for full cluster…"
# give the worker up to ~20s to join
until python3 <<'EOF'
import ray, time, sys
ray.init(address='auto')
for _ in range(20):
    if len(ray.nodes()) > 1:
        sys.exit(0)
    time.sleep(1)
sys.exit(1)
EOF
do
  echo "⏳ cluster not ready yet…"
  sleep 2
done

echo "✅ cluster ready—starting training"
exec python3 py-pong.py
