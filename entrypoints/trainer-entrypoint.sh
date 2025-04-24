#!/usr/bin/env bash
set -euo pipefail

echo "⏳ waiting for head at head:6379…"
until bash -c "echo > /dev/tcp/head/6379" 2>/dev/null; do
  sleep 2
  echo "⏳ still waiting for head…"
done
echo "✅ head is up!"

echo "⏳ waiting for dashboard at head:8265…"
until bash -c "echo > /dev/tcp/head/8265" 2>/dev/null; do
  sleep 2
  echo "⏳ still waiting for dashboard…"
done
echo "✅ dashboard is up!"

#!/usr/bin/env bash
set -euo pipefail

echo "⏳ submitting py-pong.py as a Ray Job…"
ray job submit \
  --address http://head:8265 \
  --working-dir /app \
  --runtime-env-json '{
    "excludes": [
      ".git",
      "result/bin",
      "result/libexec"
    ]
  }' \
  -- python py-pong.py

echo "✅ Ray Job finished."

# docker-compose build
# docker-compose up -d
# docker-compose down

# docker-compose logs -f trainer


# docker images
# The program 'docker' is not in your PATH. It is provided by several packages.
# You can make it available in an ephemeral shell by typing one of the following:
#   nix-shell -p docker
#   nix-shell -p docker-client
#   nix-shell -p docker_25
#   nix-shell -p docker_26
#   nix-shell -p docker_28
#   nix-shell -p nvidia-container-toolkit

#fix this