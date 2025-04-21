#!/usr/bin/env bash
set -euo pipefail

# 1) Tear down any old cluster
podman rm -f ray-head ray-worker 2>/dev/null || true

# 2) Start the head node (host networking, so 127.0.0.1:6379 & 127.0.0.1:8265 are bound on your machine)
podman run -d --name ray-head \
  --network host \
  --user "$(id -u):$(id -g)" \
  rayproject/ray:latest \
  ray start --head \
            --port=6379 \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8265 \
            --disable-usage-stats \
            --block

# 3) Start a worker that joins the head
podman run -d --name ray-worker \
  --network host \
  --user "$(id -u):$(id -g)" \
  rayproject/ray:latest \
  ray start --address=127.0.0.1:6379 \
            --disable-usage-stats \
            --block

# 4) Give the dashboard a couple seconds to come online
echo "Waiting for Ray dashboard to wake up…"
sleep 5

# 5) From an ephemeral CLI container, install antithesis (your smoke‑test helper)
#    and submit the job pointing at the host’s dashboard on 8265
podman run --rm \
  --network host \
  -v "$(pwd)":/app:Z \
  -w /app \
  rayproject/ray:latest \
  bash -lc '
    pip install antithesis &&
    export RAY_ADDRESS=http://127.0.0.1:8265 &&
    ray job submit \
      --working-dir /app \
      --runtime-env-json "{\"pip\":[\"antithesis\"],\"excludes\":[\"*.git\",\"result\",\"docs\"]}" \
      -- python py-pong.py
  '
