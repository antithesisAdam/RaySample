#!/usr/bin/env bash
set -euo pipefail

echo "⏳ waiting for head at head:6379…"
until echo > /dev/tcp/head/6379 2>/dev/null; do
  sleep 2
  echo "⏳ still waiting for head…"
done

echo "✅ head answered—joining cluster"
exec ray start --address=head:6379 --disable-usage-stats --block