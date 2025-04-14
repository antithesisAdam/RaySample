# Ray Pong + Antithesis PoC

This project is a proof-of-concept (PoC) built on the “Learning to Play Pong” Ray Core example. It demonstrates how Antithesis can be integrated into a distributed reinforcement learning training pipeline (using Ray and Gymnasium) to introduce chaos, detect bugs, and reproduce failures deterministically.

#Get Ray Cluster running

Podman Container Setup:

```mkdir -p ~/ray_temp
podman run -d --replace --name ray-head --network=host \
  -v ~/ray_temp:/tmp/ray \
  --user $(id -u):$(id -g) \
  -e RAY_NODE_IP_ADDRESS=127.0.0.1 \
  -e RAY_PUBLIC_IP=127.0.0.1 \
  rayproject/ray:latest \
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats --block --temp-dir=/tmp/ray

Similarly, run worker nodes:

```bash
podman run -d --replace --name ray-worker-1 --network=host \
  -v ~/ray_temp:/tmp/ray \
  --user $(id -u):$(id -g) \
  rayproject/ray:latest \
  ray start --address=127.0.0.1:6379 --disable-usage-stats --block --temp-dir=/tmp/ray

Access dashboard:

``` http://localhost:8265


Submit a job:

```RAY_ADDRESS="http://127.0.0.1:8265" ray job submit --working-dir . --runtime-env runtime_env.json -- python test.py


Troubleshooting:

View Running Containers:

``` podman ps

Check Logs:

```podman logs ray-head

Use Ray CLI:
```ray status
