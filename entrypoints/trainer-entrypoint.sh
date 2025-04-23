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

# docker-compose logs -f trainer
