"""
Comparison Plots for REINFORCE Hyperparameter Tuning
=====================================================
Generates comparison visualisations from the phase result JSON files
produced by hyperparameter_optimizer.py.

Usage:
    python comparison_plots.py --phase 1          # Plot Phase 1 results
    python comparison_plots.py --phase 2          # Plot Phase 2 results
    python comparison_plots.py --phase 3          # Plot Phase 3 results
    python comparison_plots.py --phase 4          # Plot Phase 4 results
    python comparison_plots.py --phase 5          # Plot Phase 5 results
    python comparison_plots.py --all              # Plot all available phases
    python comparison_plots.py --summary          # Plot best-config comparison summary
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "tuning_results"
PLOTS_DIR = "plots"
SMOOTHING_WINDOW = 50


def smooth(data, window=SMOOTHING_WINDOW):
    """Apply a moving-average smoothing."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_phase(phase_num: int) -> dict:
    path = os.path.join(RESULTS_DIR, f"phase{phase_num}_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}\nRun hyperparameter_optimizer.py --phase {phase_num} first.")
    with open(path) as f:
        return json.load(f)


def plot_phase(phase_num: int, results: dict, save_dir: str = PLOTS_DIR):
    """Create comparison plots for a single phase."""
    os.makedirs(save_dir, exist_ok=True)
    labels = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    # ---- Figure 1: Training returns ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Phase {phase_num} — Comparison", fontsize=14, fontweight="bold")

    for label, color in zip(labels, colors):
        ep_ret = results[label]["episode_returns"]
        eval_ret = results[label]["evaluator_returns"]

        # Training returns (smoothed)
        sm = smooth(ep_ret)
        axes[0].plot(range(SMOOTHING_WINDOW - 1, len(ep_ret)), sm,
                     label=label, color=color, linewidth=1.8)

        # Evaluator returns
        eval_period = results[label]["config"].get("evaluator_period", 100)
        eval_ep = [i * eval_period for i in range(len(eval_ret))]
        axes[1].plot(eval_ep, eval_ret, label=label, color=color,
                     linewidth=1.8, marker="o", markersize=3)

    for ax in axes:
        ax.axhline(200, color="green", linestyle="--", linewidth=1.2, label="Target (200)")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Episode Return (smoothed)")
    axes[0].set_title("Training Returns")
    axes[1].set_ylabel("Evaluator Return")
    axes[1].set_title("Evaluator Returns")

    plt.tight_layout()
    out = os.path.join(save_dir, f"phase{phase_num}_training_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ---- Figure 2: Final performance bar chart ----------------------------
    final_means = [results[lbl]["final_eval_mean"] for lbl in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, final_means, color=colors[:len(labels)], edgecolor="black", linewidth=0.6)
    ax.axhline(200, color="green", linestyle="--", linewidth=1.4, label="Target (200)")
    ax.set_ylabel("Final Evaluator Return (avg last 5 evals)")
    ax.set_title(f"Phase {phase_num} — Final Performance Comparison")
    ax.legend()
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, final_means):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out2 = os.path.join(save_dir, f"phase{phase_num}_final_performance.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved: {out2}")


def plot_summary(save_dir: str = PLOTS_DIR):
    """
    Plot a summary chart showing the best result from each phase.
    Falls back gracefully when phase results are not yet available.
    """
    os.makedirs(save_dir, exist_ok=True)

    phase_best = {}
    for p in range(1, 6):
        path = os.path.join(RESULTS_DIR, f"phase{p}_results.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        best_label = max(data, key=lambda k: data[k]["final_eval_mean"] or -1e9)
        phase_best[f"Phase {p}\n({best_label})"] = data[best_label]["final_eval_mean"]

    if not phase_best:
        print("No phase results found. Run at least one phase first.")
        return

    phases = list(phase_best.keys())
    vals = list(phase_best.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(phases)))

    fig, ax = plt.subplots(figsize=(max(8, len(phases) * 2), 6))
    bars = ax.bar(phases, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(200, color="green", linestyle="--", linewidth=1.4, label="Target (200)")
    ax.set_ylabel("Best Final Evaluator Return")
    ax.set_title("Hyperparameter Tuning Progress — Best Result per Phase")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, vals):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out = os.path.join(save_dir, "tuning_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_phase5_seeds(save_dir: str = PLOTS_DIR):
    """Special plot for Phase 5 showing cross-seed consistency."""
    try:
        results = load_phase(5)
    except FileNotFoundError as e:
        print(e)
        return

    os.makedirs(save_dir, exist_ok=True)
    labels = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 5 — Final Validation (multiple seeds)", fontsize=14, fontweight="bold")

    all_smoothed = []
    for label, color in zip(labels, colors):
        ep_ret = results[label]["episode_returns"]
        sm = smooth(ep_ret)
        all_smoothed.append(sm)
        axes[0].plot(range(SMOOTHING_WINDOW - 1, len(ep_ret)), sm,
                     label=label, color=color, linewidth=1.5, alpha=0.8)

        eval_ret = results[label]["evaluator_returns"]
        eval_period = results[label]["config"].get("evaluator_period", 100)
        eval_ep = [i * eval_period for i in range(len(eval_ret))]
        axes[1].plot(eval_ep, eval_ret, label=label, color=color,
                     linewidth=1.5, marker="o", markersize=3)

    # Shaded mean ± std band
    min_len = min(len(s) for s in all_smoothed)
    arr = np.array([s[:min_len] for s in all_smoothed])
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0)
    x = np.arange(SMOOTHING_WINDOW - 1, SMOOTHING_WINDOW - 1 + min_len)
    axes[0].fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.2, color="black", label="Mean ± Std")

    for ax in axes:
        ax.axhline(200, color="green", linestyle="--", linewidth=1.2, label="Target (200)")
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Episode Return (smoothed)")
    axes[0].set_title("Training Returns Across Seeds")
    axes[1].set_ylabel("Evaluator Return")
    axes[1].set_title("Evaluator Returns Across Seeds")

    plt.tight_layout()
    out = os.path.join(save_dir, "phase5_seed_consistency.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot REINFORCE hyperparameter tuning results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                       help="Plot results for a specific phase")
    group.add_argument("--all", action="store_true",
                       help="Plot all available phases")
    group.add_argument("--summary", action="store_true",
                       help="Plot cross-phase summary")
    args = parser.parse_args()

    if args.phase:
        results = load_phase(args.phase)
        plot_phase(args.phase, results)
        if args.phase == 5:
            plot_phase5_seeds()
    elif args.all:
        for p in range(1, 6):
            try:
                results = load_phase(p)
                plot_phase(p, results)
            except FileNotFoundError as e:
                print(f"Skipping Phase {p}: {e}")
        plot_phase5_seeds()
        plot_summary()
    elif args.summary:
        plot_summary()


if __name__ == "__main__":
    main()
