"""
REINFORCE Hyperparameter Optimizer for LunarLander-v3
=====================================================
Systematic 5-phase hyperparameter tuning framework for the REINFORCE agent.

Usage:
    python hyperparameter_optimizer.py --phase 1          # Run Phase 1 (learning rate)
    python hyperparameter_optimizer.py --phase 2          # Run Phase 2 (architecture)
    python hyperparameter_optimizer.py --phase 3          # Run Phase 3 (gamma)
    python hyperparameter_optimizer.py --phase 4          # Run Phase 4 (training freq)
    python hyperparameter_optimizer.py --phase 5          # Run Phase 5 (final validation)
    python hyperparameter_optimizer.py --all              # Run all phases sequentially
    python hyperparameter_optimizer.py --optimal          # Train with optimal config only
"""

import argparse
import collections
import json
import os
import random
import time
from copy import deepcopy
from shutil import rmtree

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
ENV_NAME = "LunarLander-v3"
_env_probe = gym.make(ENV_NAME)
NUM_ACTIONS = _env_probe.action_space.n
OBS_SHAPE = _env_probe.observation_space.shape
_env_probe.close()

# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------
def make_policy_network(num_actions: int, layers: list) -> hk.Transformed:
    """Factory for a simple MLP policy network."""
    def policy_network(obs):
        network = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP(layers + [num_actions])
        ])
        return network(obs)
    return hk.without_apply_rng(hk.transform(policy_network))


# ---------------------------------------------------------------------------
# Core RL functions
# ---------------------------------------------------------------------------
def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns for each timestep."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def sample_action(random_key, logits):
    """Sample an action from the policy logits."""
    return jax.random.categorical(random_key, logits)


def compute_weighted_log_prob(action_prob, episode_return):
    """Compute weighted log probability for policy gradient."""
    return jnp.log(action_prob + 1e-8) * episode_return


def policy_gradient_loss(action, logits, returns):
    """Compute the policy gradient loss for a single transition."""
    action_probs = jax.nn.softmax(logits)
    log_probs = jnp.log(action_probs + 1e-8)
    one_hot_action = jax.nn.one_hot(action, logits.shape[-1])
    return -jnp.sum(one_hot_action * log_probs * returns)


def batched_policy_gradient_loss(params, obs_batch, action_batch, returns_batch, network):
    """Compute mean policy gradient loss over a batch."""
    logits_batch = jax.vmap(network.apply, in_axes=(None, 0))(params, obs_batch)
    losses = jax.vmap(policy_gradient_loss)(action_batch, logits_batch, returns_batch)
    return jnp.mean(losses)


# ---------------------------------------------------------------------------
# Memory buffer
# ---------------------------------------------------------------------------
EpisodeReturnsMemory = collections.namedtuple(
    "EpisodeReturnsMemory", ["obs", "action", "returns"]
)
Transition = collections.namedtuple(
    "Transition", ["obs", "action", "reward", "next_obs", "done"]
)


class EpisodeReturnsBuffer:
    """Replay buffer storing (obs, action, discounted_return) tuples."""

    def __init__(self, num_transitions_to_store=4096, batch_size=512, gamma=0.99):
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory_buffer = collections.deque(maxlen=num_transitions_to_store)
        self.current_episode_transition_buffer = []

    def push(self, transition):
        self.current_episode_transition_buffer.append(transition)
        if transition.done:
            episode_rewards = [t.reward for t in self.current_episode_transition_buffer]
            G = compute_returns(episode_rewards, gamma=self.gamma)
            for i, t in enumerate(self.current_episode_transition_buffer):
                self.memory_buffer.append(EpisodeReturnsMemory(t.obs, t.action, G[i]))
            self.current_episode_transition_buffer = []

    def is_ready(self):
        return len(self.memory_buffer) >= self.batch_size

    def sample(self):
        sample = random.sample(self.memory_buffer, self.batch_size)
        obs_batch, action_batch, returns_batch = zip(*sample)
        return EpisodeReturnsMemory(
            np.stack(obs_batch).astype("float32"),
            np.asarray(action_batch).astype("int32"),
            np.asarray(returns_batch).astype("float32"),
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
REINFORCELearnState = collections.namedtuple("LearnerState", ["optim_state"])


def build_agent(config: dict):
    """Build policy network, optimizer and learn function from config."""
    network = make_policy_network(NUM_ACTIONS, config["layers"])
    rng_init = jax.random.PRNGKey(GLOBAL_SEED)
    dummy_obs = np.ones(OBS_SHAPE, "float32")
    params = network.init(rng_init, dummy_obs)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.get("grad_clip", 0.5)),
        optax.adam(config["lr"]),
    )
    optim_state = optimizer.init(params)
    learn_state = REINFORCELearnState(optim_state)

    def _loss_fn(p, obs, act, ret):
        return batched_policy_gradient_loss(p, obs, act, ret, network)

    def learn(key, params, learner_state, memory):
        grads = jax.grad(_loss_fn)(params, memory.obs, memory.action, memory.returns)
        updates, new_optim_state = optimizer.update(grads, learner_state.optim_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, REINFORCELearnState(new_optim_state)

    def choose_action(key, params, actor_state, obs, evaluation=False):
        logits = network.apply(params, obs)
        if evaluation:
            action = jnp.argmax(logits)
        else:
            action = sample_action(key, logits)
        return action, actor_state

    learn_jit = jax.jit(learn)
    choose_action_jit = jax.jit(choose_action)

    return params, learn_state, learn_jit, choose_action_jit


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_training(config: dict, video_subdir: str = "", verbose: bool = True):
    """
    Run a full training loop with the given config.

    Returns
    -------
    episode_returns : list[float]
    evaluator_returns : list[float]
    """
    params, learn_state, learn_jit, choose_action_jit = build_agent(config)

    memory = EpisodeReturnsBuffer(
        num_transitions_to_store=config["buffer_size"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
    )

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    eval_env = gym.make(ENV_NAME, render_mode="rgb_array")

    if video_subdir:
        video_dir = f"./video/{video_subdir}"
        try:
            rmtree(video_dir)
        except FileNotFoundError:
            pass
        evaluator_period = config.get("evaluator_period", 100)
        evaluation_episodes = config.get("evaluation_episodes", 8)
        env = RecordVideo(
            env, f"{video_dir}/train",
            episode_trigger=lambda x: (x % evaluator_period) == 0
        )
        eval_env = RecordVideo(
            eval_env, f"{video_dir}/eval",
            episode_trigger=lambda x: (x % evaluation_episodes) == 0
        )

    evaluator_period = config.get("evaluator_period", 100)
    evaluation_episodes = config.get("evaluation_episodes", 8)
    num_episodes = config["num_episodes"]
    learn_steps = config.get("learn_steps_per_episode", 4)

    rng = hk.PRNGSequence(jax.random.PRNGKey(config.get("seed", GLOBAL_SEED)))
    seed = config.get("seed", GLOBAL_SEED)
    random.seed(seed)

    episode_returns = []
    evaluator_returns = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed)
        episode_return = 0.0
        done = False

        while not done:
            action, _ = choose_action_jit(next(rng), params, None, np.array(obs))
            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            memory.push(Transition(obs, action, reward, next_obs, done))
            episode_return += reward
            obs = next_obs

        episode_returns.append(episode_return)

        if memory.is_ready():
            for _ in range(learn_steps):
                batch = memory.sample()
                params, learn_state = learn_jit(next(rng), params, learn_state, batch)

        if episode % evaluator_period == 0:
            eval_return = 0.0
            for _ in range(evaluation_episodes):
                obs_e, _ = eval_env.reset(seed=seed)
                done_e = False
                while not done_e:
                    action_e, _ = choose_action_jit(
                        next(rng), params, None, np.array(obs_e), evaluation=True
                    )
                    obs_e, r_e, term_e, trunc_e, _ = eval_env.step(int(action_e))
                    eval_return += r_e
                    done_e = term_e or trunc_e
            eval_return /= evaluation_episodes
            evaluator_returns.append(eval_return)

            if verbose:
                avg_last20 = np.mean(episode_returns[-20:])
                print(
                    f"Episode: {episode}\t"
                    f"Episode Return: {episode_return:.1f}\t"
                    f"Average Episode Return: {avg_last20:.1f}\t"
                    f"Evaluator Episode Return: {eval_return:.1f}"
                )

    env.close()
    eval_env.close()
    return episode_returns, evaluator_returns


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

# Base config (updated during phases)
BASE_CONFIG = {
    "layers": [256, 128],
    "lr": 3e-4,
    "gamma": 0.99,
    "buffer_size": 4096,
    "batch_size": 512,
    "learn_steps_per_episode": 4,
    "num_episodes": 3000,
    "evaluator_period": 100,
    "evaluation_episodes": 8,
    "grad_clip": 0.5,
    "seed": GLOBAL_SEED,
}

PHASE_CONFIGS = {
    1: {
        "name": "Learning Rate Optimization",
        "param": "lr",
        "values": [5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
        "fixed": {"layers": [128, 128], "gamma": 0.99, "buffer_size": 2048,
                  "batch_size": 256, "learn_steps_per_episode": 2, "num_episodes": 3000},
    },
    2: {
        "name": "Network Architecture Tuning",
        "param": "layers",
        "values": [[64, 64], [128, 128], [256, 256], [128, 64], [256, 128]],
        "fixed": {"lr": 3e-4, "gamma": 0.99, "buffer_size": 2048,
                  "batch_size": 256, "learn_steps_per_episode": 2, "num_episodes": 3000},
    },
    3: {
        "name": "Discount Factor Optimization",
        "param": "gamma",
        "values": [0.95, 0.97, 0.99, 0.995],
        "fixed": {"lr": 3e-4, "layers": [256, 128], "buffer_size": 2048,
                  "batch_size": 256, "learn_steps_per_episode": 2, "num_episodes": 3000},
    },
    4: {
        "name": "Training Frequency and Memory Tuning",
        "param": "combo",
        "combos": [
            {"learn_steps_per_episode": 1, "buffer_size": 1024, "batch_size": 256},
            {"learn_steps_per_episode": 2, "buffer_size": 2048, "batch_size": 256},
            {"learn_steps_per_episode": 4, "buffer_size": 4096, "batch_size": 512},
            {"learn_steps_per_episode": 4, "buffer_size": 4096, "batch_size": 1024},
        ],
        "fixed": {"lr": 3e-4, "layers": [256, 128], "gamma": 0.99, "num_episodes": 3000},
    },
    5: {
        "name": "Final Validation",
        "fixed": {
            "lr": 3e-4, "layers": [256, 128], "gamma": 0.99,
            "buffer_size": 4096, "batch_size": 512, "learn_steps_per_episode": 4,
            "num_episodes": 5000,
        },
        "seeds": [42, 123, 456, 789, 1024],
    },
}


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _merge(fixed: dict, override: dict) -> dict:
    cfg = deepcopy(BASE_CONFIG)
    cfg.update(fixed)
    cfg.update(override)
    return cfg


def run_phase(phase_num: int, results_dir: str = "tuning_results"):
    """Run a single tuning phase and save results."""
    os.makedirs(results_dir, exist_ok=True)
    phase = PHASE_CONFIGS[phase_num]
    print(f"\n{'='*60}")
    print(f"  Phase {phase_num}: {phase['name']}")
    print(f"{'='*60}")

    phase_results = {}

    if phase_num in (1, 2, 3):
        param = phase["param"]
        for val in phase["values"]:
            label = str(val).replace(" ", "")
            print(f"\n--- {param} = {label} ---")
            config = _merge(phase["fixed"], {param: val})
            ep_returns, eval_returns = run_training(config)
            phase_results[label] = {
                "config": config,
                "episode_returns": ep_returns,
                "evaluator_returns": eval_returns,
                "final_eval_mean": float(np.mean(eval_returns[-5:])) if eval_returns else None,
            }

    elif phase_num == 4:
        for combo in phase["combos"]:
            label = f"ls{combo['learn_steps_per_episode']}_buf{combo['buffer_size']}_bs{combo['batch_size']}"
            print(f"\n--- {label} ---")
            config = _merge(phase["fixed"], combo)
            ep_returns, eval_returns = run_training(config)
            phase_results[label] = {
                "config": config,
                "episode_returns": ep_returns,
                "evaluator_returns": eval_returns,
                "final_eval_mean": float(np.mean(eval_returns[-5:])) if eval_returns else None,
            }

    elif phase_num == 5:
        for seed in phase["seeds"]:
            label = f"seed{seed}"
            print(f"\n--- seed = {seed} ---")
            config = _merge(phase["fixed"], {"seed": seed})
            ep_returns, eval_returns = run_training(config, video_subdir=f"final_seed{seed}")
            phase_results[label] = {
                "config": config,
                "episode_returns": ep_returns,
                "evaluator_returns": eval_returns,
                "final_eval_mean": float(np.mean(eval_returns[-5:])) if eval_returns else None,
            }

    # Save results
    out_file = os.path.join(results_dir, f"phase{phase_num}_results.json")
    serializable = {}
    for k, v in phase_results.items():
        serializable[k] = {
            "config": {ck: (cv if not isinstance(cv, np.ndarray) else cv.tolist())
                       for ck, cv in v["config"].items()},
            "episode_returns": [float(r) for r in v["episode_returns"]],
            "evaluator_returns": [float(r) for r in v["evaluator_returns"]],
            "final_eval_mean": v["final_eval_mean"],
        }
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nPhase {phase_num} results saved to {out_file}")

    # Print summary
    print(f"\n--- Phase {phase_num} Summary ---")
    for label, res in phase_results.items():
        print(f"  {label:40s}  final_eval_mean={res['final_eval_mean']:.1f}")

    return phase_results


def run_optimal_training():
    """Train once with the confirmed optimal configuration."""
    print(f"\n{'='*60}")
    print("  Training with OPTIMAL configuration")
    print(f"{'='*60}\n")
    config = deepcopy(BASE_CONFIG)
    config["num_episodes"] = 5000
    episode_returns, evaluator_returns = run_training(config, video_subdir="optimal")

    # Save results
    out = {
        "config": {k: (v if not isinstance(v, list) else v) for k, v in config.items()},
        "episode_returns": [float(r) for r in episode_returns],
        "evaluator_returns": [float(r) for r in evaluator_returns],
        "final_eval_mean": float(np.mean(evaluator_returns[-10:])) if evaluator_returns else None,
    }
    with open("optimal_training_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(episode_returns, alpha=0.4, color="steelblue", label="Episode Return")
    smoothed = np.convolve(episode_returns, np.ones(50) / 50, mode="valid")
    axes[0].plot(range(49, len(episode_returns)), smoothed, color="navy", linewidth=2,
                 label="50-ep moving average")
    axes[0].axhline(200, color="green", linestyle="--", label="Target (200)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Training Returns — Optimal Config")
    axes[0].legend()

    eval_ep = [i * config["evaluator_period"] for i in range(len(evaluator_returns))]
    axes[1].plot(eval_ep, evaluator_returns, color="darkorange", marker="o", linewidth=2)
    axes[1].axhline(200, color="green", linestyle="--", label="Target (200)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Evaluator Return")
    axes[1].set_title("Evaluator Returns — Optimal Config")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("optimal_training_plot.png", dpi=150)
    print("Plot saved to optimal_training_plot.png")
    plt.close()

    print(f"\nFinal evaluator mean (last 10): {out['final_eval_mean']:.1f}")
    return episode_returns, evaluator_returns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="REINFORCE Hyperparameter Optimizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                       help="Run a specific tuning phase")
    group.add_argument("--all", action="store_true",
                       help="Run all 5 phases sequentially")
    group.add_argument("--optimal", action="store_true",
                       help="Train once with the optimal configuration")
    args = parser.parse_args()

    if args.phase:
        run_phase(args.phase)
    elif args.all:
        for p in range(1, 6):
            run_phase(p)
    elif args.optimal:
        run_optimal_training()


if __name__ == "__main__":
    main()
