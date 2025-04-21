# test.py

import ray

def main():
    # 1️⃣ Connect to your head node
    ray.init(address="192.168.68.106:6379")

    @ray.remote
    def ping():
        return "pong"

    # 2️⃣ Fire off the remote task and print the result
    result = ray.get(ping.remote())
    print(f"Ping remote returned: {result}")

    # 3️⃣ Print out cluster resources
    resources = ray.cluster_resources()
    print("Cluster resources:", resources)

if __name__ == "__main__":
    main()
