# Ray Pong + Antithesis PoC

This project is a proof-of-concept (PoC) built on the “Learning to Play Pong” Ray Core example. It demonstrates how Antithesis can be integrated into a distributed reinforcement learning training pipeline (using Ray and Gymnasium) to introduce chaos, detect bugs, and reproduce failures deterministically.

## Why This is a Good Fit

- **Distributed Computation with Ray:**  
  The example uses Ray Actors and Tasks to distribute the computation, showcasing one of Ray’s core strengths. Antithesis is designed to test such distributed systems under chaotic or fault-injected conditions.

- **Stateful and Property-based Testing:**  
  The Pong example has stateful elements (such as model weights, agent rewards, and environment state) that allow you to define meaningful properties. Examples include:  
  - “The agent's score never decreases during evaluation.”  
  - “The model weights should converge (or not diverge wildly).”

- **Deterministic Training:**  
  With careful setup (e.g., setting random seeds via `np.random.seed()`, `random.seed()`, and seeding the Gymnasium environment), the training can be made sufficiently deterministic. This is crucial for replaying failures exactly (bit-for-bit).

## Challenges

- **Non-deterministic Gym Environments:**  
  Gym environments (e.g., Pong via Gymnasium) aren’t deterministic by default unless explicitly seeded and controlled.

- **External Dependencies & Rendering:**  
  Real-time rendering and external dependencies can complicate virtualization and simulation inside the deterministic sandbox provided by Antithesis.

- **Integration Overhead:**  
  Instrumenting the Ray Pong code with Antithesis SDK assertions is necessary to define invariant properties (for example, checking that the total reward exceeds a threshold or that weights remain finite). This extra instrumentation can add development overhead.

## Suggested Implementation Plan

### Step 1: Fork & Prepare the Project
- **Fork/Clone the Ray Pong Example:**  
  Ensure that the Ray Pong example (using Ray, Gymnasium, and other dependencies) is working correctly on your local system.  
- **Containerize the Project:**  
  Create a Dockerfile to containerize the environment so that the PoC can run reproducibly under Antithesis.

### Step 2: Make the System Deterministic
- **Set Fixed Seeds:**  
  Update your training script to call `np.random.seed()`, `random.seed()`, and initialize the Gymnasium environment with a fixed seed (e.g., `gym.make("Pong-v0", seed=42)`).
- **Log Metrics:**  
  Log important metrics such as total reward and loss so you can track progress and verify reproducibility.

### Step 3: Define Invariants (Properties)
- **Antithesis SDK Assertions:**  
  Integrate the Antithesis SDK into your codebase. For example:
  ```python
  from antithesis import assert_always, assert_eventually
  
  # Example property: The agent's score should never decrease.
  assert_always(agent.total_reward >= previous_total_reward)
  
  # Example property: Ensure model weights remain finite.
  assert_always(np.isfinite(agent.policy_network.weights).all())
