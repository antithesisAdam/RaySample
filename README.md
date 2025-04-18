# Ray Pong + Antithesis PoC

This project is a proof-of-concept (PoC) built on the "Learning to Play Pong" Ray Core example. It demonstrates how Antithesis can be integrated into a distributed reinforcement learning training pipeline (using Ray and Gymnasium) to introduce chaos, detect bugs, and reproduce failures deterministically.

## Get Ray Cluster Running

### Podman Container Setup

Create a temporary directory for Ray's session files:

```bash
mkdir -p ~/ray_temp
Start the Ray head node container:

bash
Copy
podman run -d --replace --name ray-head --network=host \
  -v ~/ray_temp:/tmp/ray \
  --user $(id -u):$(id -g) \
  -e RAY_NODE_IP_ADDRESS=127.0.0.1 \
  -e RAY_PUBLIC_IP=127.0.0.1 \
  rayproject/ray:latest \
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats --block --temp-dir=/tmp/ray
Similarly, start the worker container:

bash
Copy
podman run -d --replace --name ray-worker-1 --network=host \
  -v ~/ray_temp:/tmp/ray \
  --user $(id -u):$(id -g) \
  rayproject/ray:latest \
  ray start --address=127.0.0.1:6379 --disable-usage-stats --block --temp-dir=/tmp/ray
Access the Dashboard
Open your web browser and navigate to:

bash
Copy
http://localhost:8265
The dashboard will display the cluster status, nodes, and resource usage.

Submit a Job
Before submitting, create a runtime_env.json file in your working directory that excludes unnecessary large files. For example, create a file named runtime_env.json with the following content:

json
Copy
{
  "excludes": [
    "venv/",
    "result/",
    "venv/lib64/python3.10/site-packages/nvidia/",
    "venv/lib64/python3.10/site-packages/torch/",
    "venv/lib64/python3.10/site-packages/triton/"
  ]
}
Submit your job by running:

bash
Copy
RAY_ADDRESS="http://127.0.0.1:8265" ray job submit --working-dir . --runtime-env runtime_env.json -- python test.py
The command will output a job ID (e.g., raysubmit_...) and provide instructions to view logs and check the status.

Troubleshooting
View Running Containers:

bash
Copy
podman ps
Check Logs of the Head Node:

bash
Copy
podman logs ray-head
Check Cluster Status with Ray CLI:

bash
Copy
ray status
Confirming Multi-Node Usage
To verify that tasks are distributed across nodes, you can run a script that prints the hostnames of the nodes executing the tasks. For example:

python
Copy
import ray
import socket
import time

ray.init(address="127.0.0.1:6379")

@ray.remote
def get_hostname():
    time.sleep(3)  # Delay to help visualize scheduling
    return socket.gethostname()

# Launch multiple tasks to force distribution (e.g., 100 tasks)
futures = [get_hostname.remote() for _ in range(100)]
results = ray.get(futures)
print("Node hostnames:", results)
If tasks are distributed across multiple nodes, you should see different hostnames in the output.

Additional Considerations
Resource Constraints:
Specify resource requirements in your task definitions to encourage distribution. For example:

python
Copy
@ray.remote(num_cpus=1)
def my_task():
    pass
Excluding Files:
Adjust the runtime_env "excludes" list as needed to avoid packaging large, unnecessary files.

Development Workflow:
Consider using a minimal working directory for job submission if your repository contains many large files that are not required at runtime.

This README provides a comprehensive guide to setting up your Ray cluster with Podman, submitting jobs with a custom runtime environment to exclude large files, and verifying that distributed tasks are running across multiple nodes.

yaml
Copy

---

### To Create the File

1. Open your text editor.
2. Paste the content from the code block above.
3. Save the file as **README.md** in your project directory.
4. (Optional) Add it to your Git repository:
   ```bash
   git add README.md
   git commit -m "Add README file"
   git push


   Command to get pods to work

   podman rm -f ray-head ray-worker-1 2>/dev/null

podman run -d --name ray-head \
  -p 6379:6379 -p 8265:8265 \
  --user $(id -u):$(id -g) \
  -e RAY_NODE_IP_ADDRESS=127.0.0.1 \
  rayproject/ray:latest \
  ray start --head --port=6379 \
            --dashboard-host=0.0.0.0 --dashboard-port=8265 \
            --disable-usage-stats --block          # no --temp-dir

# worker
podman run -d --name ray-worker-1 \
  --network host \
  --user $(id -u):$(id -g) \
  rayproject/ray:latest \
  ray start --address=127.0.0.1:6379 \
            --disable-usage-stats --block



#virtual deactivate

deactivate
-----------------------------

Submit a job:

ray job submit   --address http://127.0.0.1:8265   --working-dir ~/ray_job   --runtime-env-json '{
      "pip": [
        "gymnasium[accept-rom-license]==0.29.1",
        "shimmy[atari]==0.2.1",
        "ale-py==0.8.1"
      ]
    }'   -- python py-pong.py


    odman rm -f ray-head ray-worker-1 2>/dev/null

podman run -d --name ray-head \
  -p 6379:6379 -p 8265:8265 \
  --user $(id -u):$(id -g) \
  -e RAY_NODE_IP_ADDRESS=127.0.0.1 \
  rayproject/ray:latest \
  ray start --head --port=6379 \
            --dashboard-host=0.0.0.0 --dashboard-port=8265 \
            --disable-usage-stats --block          # no --temp-dir

# worker
podman run -d --name ray-worker-1 \
  --network host \
  --user $(id -u):$(id -g) \
  rayproject/ray:latest \
  ray start --address=127.0.0.1:6379 \
            --disable-usage-stats --block



ray job stop 06000000 --address http://127.0.0.1:8265

