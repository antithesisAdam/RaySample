import ray
import gymnasium as gym
import numpy as np
import pickle
import multiprocessing

# --- Config ---
H = 200
batch_size = 5
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
D = 80 * 80  # 6400
num_workers = multiprocessing.cpu_count()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(float).ravel()

def policy_forward(x, model):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h

def run_single_episode(model):
    env = gym.make("ALE/Pong-v5", render_mode=None)
    observation, _ = env.reset()
    prev_x = None

    xs, hs, dlogps, drs = [], [], [], []
    reward_sum = 0

    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        aprob, h = policy_forward(x, model)
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
            break

    return xs, hs, dlogps, drs, reward_sum

@ray.remote
def run_episodes(model_weights, n=4):
    model = model_weights.copy()
    return [run_single_episode(model) for _ in range(n)]

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_backward(eph, epdlogp, epx, model):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

if __name__ == "__main__":
    ray.shutdown()
    #ray.init(ignore_reinit_error=True, include_dashboard=True, log_to_driver=True, logging_level="debug")
    ray.init(address="auto")


    model = {
        'W1': np.random.randn(H, D) / np.sqrt(D),
        'W2': np.random.randn(H) / np.sqrt(H)
    }
    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}
    running_reward = None
    episode_number = 0

    while True:
        futures = [run_episodes.remote(model, n=2) for _ in range(num_workers // 2)]
        results = ray.get(futures)

        for batch in results:
            for xs, hs, dlogps, drs, reward_sum in batch:
                episode_number += 1
                epx = np.vstack(xs)
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)

                discounted_epr = discount_rewards(epr)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                epdlogp *= discounted_epr
                grad = policy_backward(eph, epdlogp, epx, model)

                for k in model:
                    grad_buffer[k] += grad[k]

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print(f"Episode {episode_number} — Reward: {reward_sum:.2f}, Running Mean: {running_reward:.2f}")

        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        if episode_number % 100 == 0:
            pickle.dump(model, open('save_ray_parallel.p', 'wb'))
