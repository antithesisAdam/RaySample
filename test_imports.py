import numpy as np
import gymnasium as gym
import ale_py

print("numpy version:", np.__version__)
print("gymnasium version:", gym.__version__)
print("ale-py version:", ale_py.__version__)

# Optionally, list all registered environment IDs:
print("Registered Gymnasium env IDs:")
print(list(gym.envs.registry.keys()))
