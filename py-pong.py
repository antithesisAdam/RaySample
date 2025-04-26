#!/usr/bin/env python3
"""
Distributed Pong with Ray:
  - @ray.remote train_episode() runs one full episode on any available worker.
  - Driver schedules BATCH_SIZE of them in parallel, aggregates grads, updates the model.
"""

import os
import pickle
import ray
import gymnasium as gym
import numpy as np
import json
from gymnasium.envs.registration import register
from antithesis.assertions import always as assert_always
#from antithesis.runtime import Runtime  # <-- NEW LINE (import Runtime)

# 1. Ray init: auto connects to head+workers in your cluster
ray.init(address="auto")

# 2. (Shimmy) register the Gym Pong env
register(
    id="ALE/Pong-v5",
    entry_point="shimmy.atari_env:AtariEnv",
    kwargs={"game": "pong", "obs_type": "rgb"},
    max_episode_steps=10000,
)

# -------------------------------------------------------------------
# 3. All your existing preprocessing, network, discount, etc.
# -------------------------------------------------------------------
H = 200            # hidden neurons
gamma = 0.99       # reward discount
decay_rate = 0.99  # RMSProp decay
D = 80 * 80        # input dims (80x80 after your crop/downsample)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    I = I[35:195]            # crop
    I = I[::2, ::2, 0]       # downsample + take red channel
    I[I == 144] = 0
    I[I == 109] = 0
    I[I !=   0] = 1
    return I.astype(np.float32).ravel()

def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(r.size)):
        if r[t] != 0:  # game boundary (pong)
            running_add = 0.0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x, model):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0  # ReLU
    logp = np.dot(model["W2"], h)
    return sigmoid(logp), h

def policy_backward(eph, epdlogp, epx, model):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model["W2"])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {"W1": dW1, "W2": dW2}

# -------------------------------------------------------------------
# 4. Remote Episode Roll-out + Gradient Computation
# -------------------------------------------------------------------
@ray.remote
def train_episode(remote_weights: dict):
    """Run a single episode of Pong, compute gradients & total reward."""
    env = gym.make("ALE/Pong-v5")
    observation, _ = env.reset()
    prev_x = None

    # storage for this episode
    xs, hs, dlogps, drs = [], [], [], []
    total_reward = 0.0

    while True:
        x = prepro(observation)
        x = x - prev_x if prev_x is not None else np.zeros_like(x)
        prev_x = x

        # forward pass
        aprob, h = policy_forward(x, remote_weights)
        action = 2 if np.random.uniform() < aprob else 3

        xs.append(x); hs.append(h)
        y = 1 if action == 2 else 0
        dlogps.append(y - aprob)

        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        drs.append(reward)

        if done:
            # stack episode data
            epx     = np.vstack(xs)
            eph     = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr     = np.vstack(drs)
            # compute discounted, normalized reward
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            if np.std(discounted_epr) != 0:
                discounted_epr /= np.std(discounted_epr)
            epdlogp *= discounted_epr

            # compute gradients
            grads = policy_backward(eph, epdlogp, epx, remote_weights)
            return grads, total_reward

# -------------------------------------------------------------------
# 5. Driver: initialize model & run distributed training loop
# -------------------------------------------------------------------
if __name__ == "__main__":
    # hyperparams
    BATCH_SIZE    = 10    # number of episodes per update
    LEARNING_RATE = 1e-4
    RESUME        = False

    # Antithesis setup complete signal
    #Runtime.signal_setup_complete()  # <-- NEW LINE (signal setup)

    # 5.1 initialize or resume model
    if RESUME and os.path.exists("save.p"):
        model = pickle.load(open("save.p", "rb"))
    else:
        model = {
            "W1": np.random.randn(H, D) / np.sqrt(D),
            "W2": np.random.randn(H)    / np.sqrt(H),
        }

    # buffers for RMSProp
    grad_buffer    = {k: np.zeros_like(v) for k,v in model.items()}
    rmsprop_cache  = {k: np.zeros_like(v) for k,v in model.items()}

    episode_number = 0
    running_reward = None

    print("=== Starting distributed training ===")
    while True:
        # dispatch BATCH_SIZE episodes in parallel
        futures = [train_episode.remote(model) for _ in range(BATCH_SIZE)]
        results = ray.get(futures)

        # aggregate grads & rewards
        batch_reward = 0.0
        for grads, ep_reward in results:
            batch_reward += ep_reward
            for k in model:
                grad_buffer[k] += grads[k]

        episode_number += BATCH_SIZE
        batch_reward /= BATCH_SIZE

        # RMSProp update
        for k,v in model.items():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1-decay_rate) * g**2
            model[k] += LEARNING_RATE * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)

        # running reward smoothing
        running_reward = batch_reward if running_reward is None else running_reward*0.99 + batch_reward*0.01
        print(f"After {episode_number} episodes: batch avg reward {batch_reward:.2f}, running mean {running_reward:.2f}")

        # assertion: weights remain finite
        assert_always(
            "W1 finite",
            bool(np.isfinite(model["W1"]).all()),
            "W1 went NaN!"
        )

        # periodically save
        if episode_number % 100 == 0:
            pickle.dump(model, open("save.p", "wb"))
            print("Model snapshot saved.")
