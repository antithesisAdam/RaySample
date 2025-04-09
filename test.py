import ray

# Define a runtime environment that excludes large, unnecessary files/directories.
runtime_env = {
    "excludes": [
        "venv/",  # Exclude the entire virtual environment if not needed.
        # Alternatively, exclude specific large paths:
        "venv/lib64/python3.10/site-packages/nvidia/",
        "venv/lib64/python3.10/site-packages/torch/",
        "venv/lib64/python3.10/site-packages/triton/",
        "venv/lib/python3.10/site-packages/nvidia/",
        "venv/lib/python3.10/site-packages/torch/",
        "venv/lib/python3.10/site-packages/triton/",
        "result/"  # Exclude output directories if not needed.
    ]
}

# Connect explicitly to the head node with the runtime_env.
ray.init(address="127.0.0.1:6379", runtime_env=runtime_env)

@ray.remote
def get_hostname():
    import socket
    return socket.gethostname()

futures = [get_hostname.remote() for _ in range(4)]
print("Node hostnames:", ray.get(futures))
