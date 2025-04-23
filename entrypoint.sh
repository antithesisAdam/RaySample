#!/usr/bin/env bash
set -euo pipefail

ROLE="$1"
shift

case "$ROLE" in
  head)
    echo "🔨 Starting Ray head node…"
    exec ray start \
      --head \
      --port=6379 \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265 \
      --disable-usage-stats \
      --block
    ;;
  worker)
    echo "⏳ Waiting for head at head:6379…"
    # loop until the head is listening
    until bash -c "echo > /dev/tcp/head/6379"; do
      sleep 1
    done
    echo "✅ Head is up—joining cluster"
    exec ray start \
      --address=head:6379 \
      --disable-usage-stats \
      --block
    ;;
  trainer)
    echo "🚂 Running trainer script…"
    exec python py-pong.py "$@"
    ;;
  *)
    echo "❌ Unknown role: $ROLE" >&2
    exit 1
    ;;
esac
