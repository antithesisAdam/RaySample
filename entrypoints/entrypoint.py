#!/usr/bin/env python3
"""
Entrypoint to check Ray cluster health and signal Antithesis setup complete.
"""

import ray
import time
from antithesis.lifecycle import setup_complete

SLEEP = 5
MAX_RETRIES = 12

def check_ray_cluster():
    try:
        info = ray.nodes()
        healthy = all(node["Alive"] for node in info)
        return healthy
    except Exception as e:
        print(f"[Entrypoint] Error checking Ray nodes: {e}")
        return False

if __name__ == "__main__":
    print("[Entrypoint] Starting Ray health check...")
    ray.init(address="auto")

    retries = 0
    while retries < MAX_RETRIES:
        if check_ray_cluster():
            print("[Entrypoint] Ray cluster is healthy!")
            break
        else:
            print(f"[Entrypoint] Cluster not healthy yet, retrying in {SLEEP} seconds...")
            time.sleep(SLEEP)
            retries += 1
    else:
        print("[Entrypoint] Failed to confirm Ray cluster health after retries.")
        exit(1)

    print("[Entrypoint] Emitting Antithesis setup_complete...")
    setup_complete({"message": "Ray cluster is healthy and ready"})

    print("[Entrypoint] Setup complete. Sleeping forever to keep container alive...")
    time.sleep(31536000)  # 1 year