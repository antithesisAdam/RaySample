""" Trains an agent with (stochastic) Policy Gradients on Pong using OpenAI Gymnasium and Ray. """

import ray
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from shimmy.atari_env import AtariEnv
ray.init(address="auto")  
# ✅ Manually register ALE/Pong-v5 (through Shimmy)
register(
    id="ALE/Pong-v5",
    entry_point="shimmy.atari_env:AtariEnv",
    kwargs={"game": "pong", "obs_type": "rgb"},
    max_episode_steps=10000,
)

# ✅ Create environment
env = gym.make("ALE/Pong-v5")
print("✅ Pong environment loaded!")

# Hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # episodes per parameter update
learning_rate = 1e-4
gamma = 0.99  # reward discount factor
decay_rate = 0.99  # RMSProp decay
resume = False  # toggle for loading saved model
render = False

D = 80 * 80  # input dimensionality: 80x80 grid

# Initialize model
if resume:
    import pickle
    model = pickle.load(open("save.p", "rb"))
else:
    model = {
        "W1": np.random.randn(H, D) / np.sqrt(D),
        "W2": np.random.randn(H) / np.sqrt(H),
    }

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    """ Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(float).ravel()

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0
    logp = np.dot(model["W2"], h)
    p = sigmoid(logp)
    return p, h

def policy_backward(eph, epdlogp, epx):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model["W2"])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {"W1": dW1, "W2": dW2}

# Main training loop
observation, _ = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render:
        env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    xs.append(x)
    hs.append(h)
    y = 1 if action == 2 else 0
    dlogps.append(y - aprob)

    observation, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    reward_sum += reward
    drs.append(reward)

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp, epx)
        for k in model:
            grad_buffer[k] += grad[k]

        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f"Episode {episode_number} — Reward: {reward_sum:.2f}, Running Mean: {running_reward:.2f}")
        if episode_number % 100 == 0:
            import pickle
            pickle.dump(model, open("save.p", "wb"))
        reward_sum = 0
        observation, _ = env.reset()
        prev_x = None

    if reward != 0:
        print(f"ep {episode_number}: game finished, reward: {reward:.2f}" + ("" if reward == -1 else " !!!!!!!!"))
