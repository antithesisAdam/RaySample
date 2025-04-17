# py-pong-ray.py
import os
import pickle

import numpy as np
import ray
import gymnasium as gym
from gymnasium.envs.registration import register
from shimmy.atari_env import AtariEnv

# ─── 1) Ray init with runtime_env ──────────────────────────────────────────────
ray.init(
    address=os.environ.get("RAY_ADDRESS", None),
    runtime_env={
        # ship your local code
        "working_dir": ".",
        "excludes": ["*.git*", "__pycache__"],
        # exact dependencies
        "pip": [
            "gymnasium",
            "ale-py==0.7.5",
            "shimmy==1.2.0",
            "gymnasium[accept-rom-license]",
        ],
        # allow Shimmy to find ROMs
        "env_vars": {"PYTHONWARNINGS": "default::ImportWarning:ale_py.roms"},
    },
)

# ─── 2) Hyperparameters ────────────────────────────────────────────────────────
H, batch_size = 200, 10
learning_rate, gamma, decay_rate = 1e-4, 0.99, 0.99
resume, render = False, False
D = 80 * 80

# ─── 3) Model init / load ─────────────────────────────────────────────────────
if resume and os.path.exists("save.p"):
    model = pickle.load(open("save.p", "rb"))
else:
    model = {
        "W1": np.random.randn(H, D) / np.sqrt(D),
        "W2": np.random.randn(H) / np.sqrt(H),
    }

# ─── 4) Utility functions ─────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(frame):
    I = frame[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()

def discount_rewards(r):
    discounted = np.zeros_like(r)
    running = 0
    for t in reversed(range(len(r))):
        if r[t] != 0:
            running = 0
        running = running * gamma + r[t]
        discounted[t] = running
    return discounted

def policy_forward(x, model):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0
    logp = np.dot(model["W2"], h)
    return sigmoid(logp), h

def policy_backward(eph, epdlogp, epx, model):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model["W2"])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {"W1": dW1, "W2": dW2}

# ─── 5) Remote rollout ─────────────────────────────────────────────────────────
@ray.remote
def rollout(weights, seed=None):
    # rehydrate model
    model = {k: v.copy() for k, v in weights.items()}

    # re‑register inside this worker
    register(
        id="ALE/Pong-v5",
        entry_point="shimmy.atari_env:AtariEnv",
        kwargs={"game": "pong", "obs_type": "rgb"},
        max_episode_steps=10000,
    )
    env = gym.make("ALE/Pong-v5")

    # optional seeding
    if seed is not None:
        obs, _ = env.reset(seed=seed)
        np.random.seed(seed)
    else:
        obs, _ = env.reset()

    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    reward_sum = 0

    while True:
        x = prepro(obs)
        x = x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = x

        aprob, h = policy_forward(x, model)
        action = 2 if np.random.rand() < aprob else 3

        xs.append(x); hs.append(h)
        dlogps.append((1 if action == 2 else 0) - aprob)

        obs, reward, terminated, truncated, _ = env.step(action)
        drs.append(reward)
        reward_sum += reward

        done = terminated or truncated
        if done:
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)

            # normalize & compute gradient
            discounted = discount_rewards(epr)
            discounted -= discounted.mean()
            discounted /= (discounted.std() + 1e-8)
            epdlogp *= discounted

            grad = policy_backward(eph, epdlogp, epx, model)
            return grad, reward_sum

# ─── 6) Training loop ──────────────────────────────────────────────────────────
grad_buffer    = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache  = {k: np.zeros_like(v) for k, v in model.items()}

for it in range(1, 1001):
    # fire off a batch of parallel rollouts
    futures = [rollout.remote(model, seed=np.random.randint(1e6))
               for _ in range(batch_size)]
    results = ray.get(futures)

    rewards = []
    for grad, rsum in results:
        rewards.append(rsum)
        for k in model:
            grad_buffer[k] += grad[k]

    # update params
    for k in model:
        g = grad_buffer[k] / batch_size
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k].fill(0)

    print(f"Iteration {it} | avg reward: {np.mean(rewards):.2f}")

    if it % 50 == 0:
        pickle.dump(model, open("save.p", "wb"))

ray.shutdown()
