"""
Generate plots and summary statistics from experiment results (v2).
Handles multi-seed replication, log-prob gap metric, and statistical tests.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import config


def load_results():
    path = os.path.join(config.RESULTS_DIR, "all_results.jsonl")
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def group_by(results, keys):
    """Group results by a combination of keys, returning {key_tuple: [results]}."""
    groups = {}
    for r in results:
        k = tuple(r[k] for k in keys)
        groups.setdefault(k, []).append(r)
    return groups


def plot_heatmap(results, output_dir):
    """Heatmap: X=contamination ratio, Y=generation, color=mean novelty_accuracy."""
    fixed = [r for r in results if r["arm"] == "fixed"]
    ratios = sorted(set(r["ratio"] for r in fixed))
    gens = sorted(set(r["generation"] for r in fixed))

    mean_matrix = np.full((len(gens), len(ratios)), np.nan)
    std_matrix = np.full((len(gens), len(ratios)), np.nan)

    groups = group_by(fixed, ["ratio", "generation"])
    for (ratio, gen), group in groups.items():
        gi = gens.index(gen)
        ri = ratios.index(ratio)
        accs = [r["novelty_accuracy"] for r in group]
        mean_matrix[gi, ri] = np.mean(accs)
        std_matrix[gi, ri] = np.std(accs) if len(accs) > 1 else 0

    # Build annotation strings with mean +/- std
    annot = np.empty_like(mean_matrix, dtype=object)
    for i in range(mean_matrix.shape[0]):
        for j in range(mean_matrix.shape[1]):
            if np.isnan(mean_matrix[i, j]):
                annot[i, j] = ""
            elif std_matrix[i, j] > 0:
                annot[i, j] = f"{mean_matrix[i, j]:.3f}\n+/-{std_matrix[i, j]:.3f}"
            else:
                annot[i, j] = f"{mean_matrix[i, j]:.3f}"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mean_matrix,
        xticklabels=[f"{r:.0%}" for r in ratios],
        yticklabels=[f"Gen {g}" for g in gens],
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Contamination Ratio")
    ax.set_ylabel("Generation")
    ax.set_title("Novelty Accuracy by Contamination Ratio and Generation (mean +/- std)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmap_novelty_accuracy.png"), dpi=150)
    plt.close(fig)
    print("Saved heatmap_novelty_accuracy.png")


def plot_logprob_gap(results, output_dir):
    """Line plot: logprob_gap over generations for each ratio, with error bands."""
    fixed = [r for r in results if r["arm"] == "fixed"]
    ratios = sorted(set(r["ratio"] for r in fixed))
    gens = sorted(set(r["generation"] for r in fixed))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))

    for ratio, color in zip(ratios, colors):
        means, stds = [], []
        for gen in gens:
            group = [r for r in fixed if r["ratio"] == ratio and r["generation"] == gen]
            gaps = [r["logprob_gap"] for r in group]
            means.append(np.mean(gaps))
            stds.append(np.std(gaps) if len(gaps) > 1 else 0)
        means, stds = np.array(means), np.array(stds)
        ax.plot(gens, means, "o-", color=color, label=f"{ratio:.0%}", linewidth=2)
        if np.any(stds > 0):
            ax.fill_between(gens, means - stds, means + stds, color=color, alpha=0.15)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Log-Prob Gap (standard - novel)")
    ax.set_title("Log-Prob Gap Over Generations\n(widening = novel constructions becoming relatively harder)")
    ax.legend(title="Contamination")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "logprob_gap.png"), dpi=150)
    plt.close(fig)
    print("Saved logprob_gap.png")


def plot_perplexity_vs_gap(results, output_dir):
    """Scatter: val_perplexity vs logprob_gap — does the gap widen even when ppl is stable?"""
    fig, ax = plt.subplots(figsize=(8, 6))

    fixed = [r for r in results if r["arm"] == "fixed"]
    compound = [r for r in results if r["arm"] == "compound"]

    if fixed:
        ratios = [r["ratio"] for r in fixed]
        sc = ax.scatter(
            [r["val_perplexity"] for r in fixed],
            [r["logprob_gap"] for r in fixed],
            c=ratios, cmap="viridis", s=80, edgecolors="black", linewidth=0.5,
            label="Fixed-ratio", zorder=3,
        )
        plt.colorbar(sc, ax=ax, label="Contamination Ratio")

    if compound:
        ax.scatter(
            [r["val_perplexity"] for r in compound],
            [r["logprob_gap"] for r in compound],
            marker="^", c="red", s=100, edgecolors="black", linewidth=0.5,
            label="Compounding", zorder=4,
        )

    ax.set_xlabel("Validation Perplexity")
    ax.set_ylabel("Log-Prob Gap")
    ax.set_title("Perplexity vs. Log-Prob Gap\n(upper-left quadrant = gap widens without ppl change)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "perplexity_vs_gap.png"), dpi=150)
    plt.close(fig)
    print("Saved perplexity_vs_gap.png")


def plot_compounding_line(results, output_dir):
    """Line plot: compounding arm metrics over generations."""
    compound = [r for r in results if r["arm"] == "compound"]
    if not compound:
        print("No compounding arm results found, skipping.")
        return

    gens = sorted(set(r["generation"] for r in compound))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, label in zip(axes,
        ["novelty_accuracy", "logprob_gap", "val_perplexity"],
        ["Novelty Accuracy", "Log-Prob Gap", "Val Perplexity"]):

        means, stds = [], []
        for gen in gens:
            group = [r for r in compound if r["generation"] == gen]
            vals = [r[metric] for r in group]
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        means, stds = np.array(means), np.array(stds)
        ax.plot(gens, means, "o-", linewidth=2)
        if np.any(stds > 0):
            ax.fill_between(gens, means - stds, means + stds, alpha=0.2)
        ax.set_xlabel("Generation")
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle(f"Compounding Arm ({config.COMPOUNDING_RATIO:.0%} per generation)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "compounding_arm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved compounding_arm.png")


def plot_diversity_over_generations(results, output_dir):
    """Line plot: Distinct-N over generations for each ratio."""
    fixed = [r for r in results if r["arm"] == "fixed"]
    ratios = sorted(set(r["ratio"] for r in fixed))
    gens = sorted(set(r["generation"] for r in fixed))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, n_label, key in zip(axes,
        ["Distinct-1", "Distinct-2", "Distinct-3"],
        ["distinct_1", "distinct_2", "distinct_3"]):
        for ratio in ratios:
            means = []
            for gen in gens:
                group = [r for r in fixed if r["ratio"] == ratio and r["generation"] == gen]
                means.append(np.mean([r[key] for r in group]))
            ax.plot(gens, means, "o-", label=f"{ratio:.0%}")
        ax.set_xlabel("Generation")
        ax.set_ylabel(n_label)
        ax.set_title(n_label)
        ax.legend(title="Ratio", fontsize=8)

    fig.suptitle("Generation Diversity (Distinct-N) Over Generations", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "diversity_over_generations.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved diversity_over_generations.png")


def print_summary_table(results):
    """Print a summary table with means across seeds."""
    print("\n" + "=" * 110)
    print(f"{'Arm':<12} {'Ratio':>6} {'Gen':>4} {'n':>3} {'NoveltyAcc':>11} {'ValPPL':>9} {'LogProbGap':>11} {'D-1':>6} {'D-2':>6} {'D-3':>6}")
    print("-" * 110)

    groups = group_by(results, ["arm", "ratio", "generation"])
    for key in sorted(groups.keys()):
        arm, ratio, gen = key
        group = groups[key]
        n = len(group)
        acc = np.mean([r["novelty_accuracy"] for r in group])
        ppl = np.mean([r["val_perplexity"] for r in group])
        gap = np.mean([r["logprob_gap"] for r in group])
        d1 = np.mean([r["distinct_1"] for r in group])
        d2 = np.mean([r["distinct_2"] for r in group])
        d3 = np.mean([r["distinct_3"] for r in group])

        acc_std = np.std([r["novelty_accuracy"] for r in group]) if n > 1 else 0
        gap_std = np.std([r["logprob_gap"] for r in group]) if n > 1 else 0

        acc_str = f"{acc:.4f}" + (f"+/-{acc_std:.3f}" if acc_std > 0 else "")
        gap_str = f"{gap:.4f}" + (f"+/-{gap_std:.3f}" if gap_std > 0 else "")

        print(
            f"{arm:<12} {ratio:>6.0%} {gen:>4d} {n:>3d} "
            f"{acc_str:>11} {ppl:>9.2f} {gap_str:>11} "
            f"{d1:>6.4f} {d2:>6.4f} {d3:>6.4f}"
        )
    print("=" * 110)


def run_statistical_tests(results):
    """Run t-tests comparing each condition to the 0% control."""
    fixed = [r for r in results if r["arm"] == "fixed"]
    ratios = sorted(set(r["ratio"] for r in fixed))
    gens = sorted(set(r["generation"] for r in fixed))

    if 0.0 not in ratios:
        print("No control arm (0.0) found, skipping tests.")
        return

    print("\n--- Statistical Tests (vs. 0% control) ---")
    print(f"{'Metric':<15} {'Ratio':>6} {'Gen':>4} {'t-stat':>8} {'p-value':>8} {'sig':>4}")
    print("-" * 55)

    for metric in ["logprob_gap", "novelty_accuracy"]:
        for gen in gens:
            control = [r[metric] for r in fixed if r["ratio"] == 0.0 and r["generation"] == gen]
            for ratio in ratios:
                if ratio == 0.0:
                    continue
                treatment = [r[metric] for r in fixed if r["ratio"] == ratio and r["generation"] == gen]
                if len(control) < 2 or len(treatment) < 2:
                    continue
                t, p = stats.ttest_ind(control, treatment)
                sig = "*" if p < 0.05 else ""
                print(f"{metric:<15} {ratio:>6.0%} {gen:>4d} {t:>8.3f} {p:>8.4f} {sig:>4}")


def main():
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    results = load_results()
    print(f"Loaded {len(results)} result entries.")

    print_summary_table(results)
    plot_heatmap(results, config.PLOT_DIR)
    plot_logprob_gap(results, config.PLOT_DIR)
    plot_perplexity_vs_gap(results, config.PLOT_DIR)
    plot_compounding_line(results, config.PLOT_DIR)
    plot_diversity_over_generations(results, config.PLOT_DIR)
    run_statistical_tests(results)

    print(f"\nAll plots saved to: {config.PLOT_DIR}")


if __name__ == "__main__":
    main()
