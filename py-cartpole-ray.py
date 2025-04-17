# py-cartpole-ray.py
import os
import pickle
import numpy as np
import ray
import gymnasium as gym

# 1) Ray init (point at your local cluster)
ray.init(address=os.environ.get("RAY_ADDRESS", None))

# 2) Hyperparameters
H = 32              # hidden units
batch_size = 16     # parallel rollouts per update
learning_rate = 1e-2
gamma = 0.99
decay_rate = 0.99
resume = False

# 3) Model init / load
if resume and os.path.exists("save_cartpole.p"):
    model = pickle.load(open("save_cartpole.p", "rb"))
else:
    # simple two‑layer policy network
    D = gym.make("CartPole-v1").observation_space.shape[0]
    model = {
        "W1": np.random.randn(H, D) / np.sqrt(D),
        "W2": np.random.randn(2, H) / np.sqrt(H),
    }

# 4) Helpers
def discount_rewards(r):
    out = np.zeros_like(r)
    running = 0
    for t in reversed(range(len(r))):
        running = running * gamma + r[t]
        out[t] = running
    return out

def policy_forward(x, model):
    h = np.dot(model["W1"], x)
    h = np.maximum(h, 0)            # ReLU
    logits = np.dot(model["W2"], h)
    # softmax
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs, h

def policy_backward(eph, epdlogp, epx, model):
    dW2 = epdlogp.T @ eph
    dh = (epdlogp @ model["W2"]) * (eph > 0)
    dW1 = dh.T @ epx
    return {"W1": dW1, "W2": dW2}

# 5) Remote rollout
@ray.remote
def rollout(weights, seed=None):
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    xs, hs, dlogps, drs = [], [], [], []
    done = False
    reward_sum = 0

    while not done:
        x = obs.astype(float)
        probs, h = policy_forward(x, weights)
        a = np.random.choice(2, p=probs)
        xs.append(x); hs.append(h)
        dlogps.append((np.eye(2)[a] - probs))
        obs, r, done, _, _ = env.step(a)
        drs.append(r)
        reward_sum += r

    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    discounted = discount_rewards(np.array(drs))
    discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
    epdlogp *= discounted[:, None]

    grad = policy_backward(eph, epdlogp, epx, weights)
    return grad, reward_sum

# 6) Training loop
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

for it in range(1, 501):
    futures = [rollout.remote(model, seed=np.random.randint(1e6))
               for _ in range(batch_size)]
    results = ray.get(futures)

    rewards = []
    for grad, r in results:
        rewards.append(r)
        for k in model:
            grad_buffer[k] += grad[k]

    # update
    for k in model:
        g = grad_buffer[k] / batch_size
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * (g ** 2)
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k].fill(0)

    print(f"Iteration {it:03d} | avg reward: {np.mean(rewards):.2f}")

    if it % 50 == 0:
        pickle.dump(model, open("save_cartpole.p", "wb"))
