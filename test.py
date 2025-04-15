import ray
import time
import socket

ray.init(address="auto")  # or your Ray address

@ray.remote
def heavy_task(seconds):
    """Simulate a longer task."""
    time.sleep(seconds)
    return socket.gethostname()

# Launch more tasks than a single node can handle quickly
futures = [heavy_task.remote(3) for _ in range(100)]
results = ray.get(futures)
print(results)
