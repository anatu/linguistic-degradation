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


def load_results(version="v2"):
    if version == "v4":
        path = config.RESULTS_V4_PATH
    elif version == "v3":
        path = config.RESULTS_V3_PATH
    else:
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


def plot_logprob_gap_critical(results, output_dir):
    """Line plot: critical-span logprob_gap over generations for each ratio."""
    fixed = [r for r in results if r["arm"] == "fixed"]
    ratios = sorted(set(r["ratio"] for r in fixed))
    gens = sorted(set(r["generation"] for r in fixed))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))

    for ratio, color in zip(ratios, colors):
        means, stds = [], []
        for gen in gens:
            group = [r for r in fixed if r["ratio"] == ratio and r["generation"] == gen]
            gaps = [r["logprob_gap_critical"] for r in group]
            means.append(np.mean(gaps))
            stds.append(np.std(gaps) if len(gaps) > 1 else 0)
        means, stds = np.array(means), np.array(stds)
        ax.plot(gens, means, "o-", color=color, label=f"{ratio:.0%}", linewidth=2)
        if np.any(stds > 0):
            ax.fill_between(gens, means - stds, means + stds, color=color, alpha=0.15)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Log-Prob Gap — Critical Span (standard - novel)")
    ax.set_title("Critical-Span Log-Prob Gap Over Generations\n(scoring only construction-defining tokens)")
    ax.legend(title="Contamination")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "logprob_gap_critical.png"), dpi=150)
    plt.close(fig)
    print("Saved logprob_gap_critical.png")


def plot_gap_comparison(results, output_dir):
    """Side-by-side: v2 full-sentence gap vs v3 critical-span gap."""
    fixed = [r for r in results if r["arm"] == "fixed"]
    ratios = sorted(set(r["ratio"] for r in fixed))
    gens = sorted(set(r["generation"] for r in fixed))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))

    for ratio, color in zip(ratios, colors):
        means_v2, means_v3 = [], []
        for gen in gens:
            group = [r for r in fixed if r["ratio"] == ratio and r["generation"] == gen]
            means_v2.append(np.mean([r["logprob_gap"] for r in group]))
            means_v3.append(np.mean([r["logprob_gap_critical"] for r in group]))
        ax1.plot(gens, means_v2, "o-", color=color, label=f"{ratio:.0%}", linewidth=2)
        ax2.plot(gens, means_v3, "o-", color=color, label=f"{ratio:.0%}", linewidth=2)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Log-Prob Gap")
    ax1.set_title("V2: Full-Sentence Gap")
    ax1.legend(title="Contamination")

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Log-Prob Gap (Critical Span)")
    ax2.set_title("V3: Critical-Span Gap")
    ax2.legend(title="Contamination")

    fig.suptitle("Log-Prob Gap: Full Sentence vs Critical Span", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gap_comparison_v2_v3.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved gap_comparison_v2_v3.png")


def plot_decomposed_critical(results, output_dir):
    """Line plot: decomposed standard vs novel (critical) log-probs."""
    fixed = [r for r in results if r["arm"] == "fixed"]
    ratios = sorted(set(r["ratio"] for r in fixed))
    gens = sorted(set(r["generation"] for r in fixed))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))

    for ratio, color in zip(ratios, colors):
        means_std, means_novel = [], []
        for gen in gens:
            group = [r for r in fixed if r["ratio"] == ratio and r["generation"] == gen]
            means_std.append(np.mean([r["logprob_standard"] for r in group]))
            means_novel.append(np.mean([r["logprob_novel_critical"] for r in group]))
        ax1.plot(gens, means_std, "o-", color=color, label=f"{ratio:.0%}", linewidth=2)
        ax2.plot(gens, means_novel, "o-", color=color, label=f"{ratio:.0%}", linewidth=2)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Log-Prob")
    ax1.set_title("Standard Text Log-Prob")
    ax1.legend(title="Contamination")

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Log-Prob")
    ax2.set_title("Novel Text Log-Prob (Critical Span Only)")
    ax2.legend(title="Contamination")

    fig.suptitle("Decomposed Log-Probs: Standard vs Novel (Critical Span)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "decomposed_critical.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved decomposed_critical.png")


def plot_v4_lr_comparison(results, output_dir):
    """V4-specific: compare primary vs aggressive LR on the 100% arm."""
    has_config = any("v4_config" in r for r in results)
    if not has_config:
        return

    primary = [r for r in results if r.get("v4_config") == "primary" and r["ratio"] == 1.0]
    aggressive = [r for r in results if r.get("v4_config") == "aggressive_lr" and r["ratio"] == 1.0]
    if not primary or not aggressive:
        return

    gens = sorted(set(r["generation"] for r in primary))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Novel LP (critical span)
    ax = axes[0]
    for data, label, color, marker in [(primary, "LR=5e-5", "tab:blue", "o"), (aggressive, "LR=1e-4", "tab:red", "s")]:
        vals = [np.mean([r["logprob_novel_critical"] for r in data if r["generation"] == g]) for g in gens]
        ax.plot(gens, vals, f"{marker}-", color=color, label=label, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Log-Prob")
    ax.set_title("Novel Text Log-Prob\n(Critical Span)")
    ax.legend()

    # Panel 2: Critical-span gap
    ax = axes[1]
    for data, label, color, marker in [(primary, "LR=5e-5", "tab:blue", "o"), (aggressive, "LR=1e-4", "tab:red", "s")]:
        vals = [np.mean([r["logprob_gap_critical"] for r in data if r["generation"] == g]) for g in gens]
        ax.plot(gens, vals, f"{marker}-", color=color, label=label, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Log-Prob Gap (Critical Span)")
    ax.set_title("Critical-Span Gap\n(higher = worse at novel text)")
    ax.legend()

    # Panel 3: Standard LP
    ax = axes[2]
    for data, label, color, marker in [(primary, "LR=5e-5", "tab:blue", "o"), (aggressive, "LR=1e-4", "tab:red", "s")]:
        vals = [np.mean([r["logprob_standard"] for r in data if r["generation"] == g]) for g in gens]
        ax.plot(gens, vals, f"{marker}-", color=color, label=label, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Log-Prob")
    ax.set_title("Standard Text Log-Prob")
    ax.legend()

    fig.suptitle("V4: Primary vs Aggressive LR at 100% Contamination", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "v4_lr_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved v4_lr_comparison.png")


def plot_v4_decomposed_by_config(results, output_dir):
    """V4-specific: decomposed log-probs with each config shown separately."""
    has_config = any("v4_config" in r for r in results)
    if not has_config:
        return

    configs = sorted(set(r.get("v4_config", "unknown") for r in results))

    fig, axes = plt.subplots(len(configs), 2, figsize=(14, 6 * len(configs)), squeeze=False)

    for row, cfg in enumerate(configs):
        cfg_results = [r for r in results if r.get("v4_config") == cfg]
        ratios = sorted(set(r["ratio"] for r in cfg_results))
        gens = sorted(set(r["generation"] for r in cfg_results))
        colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))

        lr = cfg_results[0].get("v4_learning_rate", "?")

        for ratio, color in zip(ratios, colors):
            means_std, means_novel = [], []
            for gen in gens:
                group = [r for r in cfg_results if r["ratio"] == ratio and r["generation"] == gen]
                means_std.append(np.mean([r["logprob_standard"] for r in group]))
                means_novel.append(np.mean([r["logprob_novel_critical"] for r in group]))
            axes[row, 0].plot(gens, means_std, "o-", color=color, label=f"{ratio:.0%}", linewidth=2)
            axes[row, 1].plot(gens, means_novel, "o-", color=color, label=f"{ratio:.0%}", linewidth=2)

        axes[row, 0].set_xlabel("Generation")
        axes[row, 0].set_ylabel("Mean Log-Prob")
        axes[row, 0].set_title(f"Standard Text — {cfg} (LR={lr})")
        axes[row, 0].legend(title="Contamination")

        axes[row, 1].set_xlabel("Generation")
        axes[row, 1].set_ylabel("Mean Log-Prob")
        axes[row, 1].set_title(f"Novel Text (Critical Span) — {cfg} (LR={lr})")
        axes[row, 1].legend(title="Contamination")

    fig.suptitle("V4: Decomposed Log-Probs by Config", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "v4_decomposed_by_config.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved v4_decomposed_by_config.png")


def print_summary_table(results):
    """Print a summary table with means across seeds."""
    has_v3 = any("logprob_gap_critical" in r for r in results)
    if has_v3:
        print("\n" + "=" * 125)
        print(f"{'Arm':<12} {'Ratio':>6} {'Gen':>4} {'n':>3} {'NoveltyAcc':>11} {'ValPPL':>9} {'GapFull':>11} {'GapCrit':>11} {'D-1':>6} {'D-2':>6} {'D-3':>6}")
        print("-" * 125)
    else:
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

        if has_v3:
            gap_crit = np.mean([r.get("logprob_gap_critical", float("nan")) for r in group])
            gap_crit_str = f"{gap_crit:.4f}"
            print(
                f"{arm:<12} {ratio:>6.0%} {gen:>4d} {n:>3d} "
                f"{acc_str:>11} {ppl:>9.2f} {gap_str:>11} {gap_crit_str:>11} "
                f"{d1:>6.4f} {d2:>6.4f} {d3:>6.4f}"
            )
        else:
            print(
                f"{arm:<12} {ratio:>6.0%} {gen:>4d} {n:>3d} "
                f"{acc_str:>11} {ppl:>9.2f} {gap_str:>11} "
                f"{d1:>6.4f} {d2:>6.4f} {d3:>6.4f}"
            )
    print("=" * (125 if has_v3 else 110))


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


def plot_training_curves(output_dir):
    """Plot train_loss and val_loss vs step for key conditions.

    Loads training logs from results/logs/ and shows convergence behavior
    across contamination levels for both shallow (V2) and deep (V4) runs.
    """
    log_dir = config.LOG_DIR

    # Define conditions to plot: (label, glob pattern prefix, color, linestyle)
    v2_conditions = [
        ("V2 0% seed42",   "fixed_0.00_gen1_seed42",   "tab:blue",   "-"),
        ("V2 50% seed42",  "fixed_0.50_gen1_seed42",   "tab:orange", "-"),
        ("V2 100% seed42", "fixed_1.00_gen1_seed42",   "tab:red",    "-"),
    ]
    v4_conditions = [
        ("V4 0% seed42",     "v4_primary_fixed_0.00_gen1_seed42",       "tab:blue",   "-"),
        ("V4 50% seed42",    "v4_primary_fixed_0.50_gen1_seed42",       "tab:orange", "-"),
        ("V4 100% seed42",   "v4_primary_fixed_1.00_gen1_seed42",       "tab:red",    "-"),
        ("V4 100% aggr s42", "v4_aggressive_lr_fixed_1.00_gen1_seed42", "tab:red",    "--"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, conditions, title in [
        (axes[0], v2_conditions, "Shallow Fine-Tuning (500 steps, Gen 1)"),
        (axes[1], v4_conditions, "Deep Fine-Tuning (5,000 steps, Gen 1)"),
    ]:
        for label, prefix, color, ls in conditions:
            log_path = os.path.join(log_dir, f"{prefix}.json")
            if not os.path.exists(log_path):
                continue
            with open(log_path) as f:
                log = json.load(f)
            steps_data = log.get("steps", [])
            if not steps_data:
                continue
            steps = [s["step"] for s in steps_data]
            val_losses = [s["val_loss"] for s in steps_data]
            ax.plot(steps, val_losses, color=color, linestyle=ls, linewidth=1.5, label=label)

        ax.set_xlabel("Step")
        ax.set_ylabel("Validation Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Training Curves: Validation Loss by Contamination Level", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved training_curves.png")


def run_per_item_analysis(output_dir):
    """Per-item benchmark sensitivity analysis.

    For each of the 51 benchmark items, compute the mean log-prob gap change
    from baseline (0%, gen 0) to contaminated (100%, gen 1) under deep training.
    Rank items by sensitivity and save a summary table + plot.
    """
    # Load V4 results
    v4_path = config.RESULTS_V4_PATH
    if not os.path.exists(v4_path):
        print("No V4 results found, skipping per-item analysis.")
        return

    results = load_results(version="v4")

    # We need per-item scores — load benchmark and re-evaluate key checkpoints
    benchmark_path = config.BENCHMARK_PATH
    examples = []
    with open(benchmark_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Get category_gaps from results (per-category, not per-item)
    # For per-item we need to run the eval — but we can approximate with category-level
    # data from the existing results, or actually run per-item eval.

    # Use existing category_gaps_critical data to show per-category sensitivity
    primary_100_gen1 = [r for r in results
                        if r.get("v4_config") == "primary"
                        and r["ratio"] == 1.0 and r["generation"] == 1]
    primary_0_gen0 = [r for r in results
                      if r.get("v4_config") == "primary"
                      and r["ratio"] == 0.0 and r["generation"] == 0]

    if not primary_100_gen1 or not primary_0_gen0:
        print("Missing baseline or contaminated results for per-item analysis.")
        return

    # Category-level analysis from existing data
    categories = sorted(set(ex["category"] for ex in examples))
    cat_counts = {}
    for ex in examples:
        cat_counts[ex["category"]] = cat_counts.get(ex["category"], 0) + 1

    print("\n--- Per-Category Sensitivity Analysis (Deep Training, Critical-Span Gap) ---")
    print(f"{'Category':<25} {'N':>3} {'Gap@0%g0':>10} {'Gap@100%g1':>11} {'Delta':>8} {'Rel.Change':>10}")
    print("-" * 70)

    cat_data = []
    for cat in categories:
        baseline_gaps = [r["category_gaps_critical"].get(cat, float("nan")) for r in primary_0_gen0]
        contaminated_gaps = [r["category_gaps_critical"].get(cat, float("nan")) for r in primary_100_gen1]
        base_mean = np.nanmean(baseline_gaps)
        cont_mean = np.nanmean(contaminated_gaps)
        delta = cont_mean - base_mean
        rel = delta / abs(base_mean) * 100 if base_mean != 0 else float("nan")
        cat_data.append((cat, cat_counts.get(cat, 0), base_mean, cont_mean, delta, rel))
        print(f"{cat:<25} {cat_counts.get(cat, 0):>3} {base_mean:>10.4f} {cont_mean:>11.4f} {delta:>8.4f} {rel:>9.1f}%")

    # Sort by delta (most sensitive first)
    cat_data.sort(key=lambda x: -x[4])
    print("\nRanked by sensitivity (most affected first):")
    for i, (cat, n, base, cont, delta, rel) in enumerate(cat_data, 1):
        print(f"  {i}. {cat} (n={n}): +{delta:.3f} ({rel:+.1f}%)")

    # Plot per-category sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = [d[0].replace("_", " ").title() for d in cat_data]
    deltas = [d[4] for d in cat_data]
    baselines = [d[2] for d in cat_data]
    contaminated = [d[3] for d in cat_data]

    x = np.arange(len(cats))
    width = 0.35
    ax.bar(x - width/2, baselines, width, label="Baseline (0%, Gen 0)", color="tab:blue", alpha=0.8)
    ax.bar(x + width/2, contaminated, width, label="Contaminated (100%, Gen 1)", color="tab:red", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Critical-Span Log-Prob Gap")
    ax.set_title("Per-Category Sensitivity to Contamination\n(Deep Training, 100% Contamination, Gen 1)")
    ax.legend()

    # Add delta annotations
    for i, (b, c, d) in enumerate(zip(baselines, contaminated, deltas)):
        ax.annotate(f"+{d:.2f}", (i + width/2, c), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8, color="tab:red")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_category_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("Saved per_category_sensitivity.png")

    # Also do per-category across generations for the full-sentence gap
    fig, ax = plt.subplots(figsize=(10, 6))
    gens = [0, 1, 2]
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

    for cat, color in zip(categories, colors):
        means = []
        for gen in gens:
            group = [r for r in results
                     if r.get("v4_config") == "primary"
                     and r["ratio"] == 1.0 and r["generation"] == gen]
            gaps = [r["category_gaps_critical"].get(cat, float("nan")) for r in group]
            means.append(np.nanmean(gaps))
        ax.plot(gens, means, "o-", color=color,
                label=cat.replace("_", " ").title(), linewidth=2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Critical-Span Log-Prob Gap")
    ax.set_title("Per-Category Gap Trajectory at 100% Contamination\n(Deep Training, Primary LR)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_category_trajectory.png"), dpi=150)
    plt.close(fig)
    print("Saved per_category_trajectory.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", action="store_true", help="Use v3 results with critical-span metrics")
    parser.add_argument("--v4", action="store_true", help="Use v4 results (extended fine-tuning)")
    args = parser.parse_args()

    os.makedirs(config.PLOT_DIR, exist_ok=True)

    if args.v4 and os.path.exists(config.RESULTS_V4_PATH):
        results = load_results(version="v4")
        print(f"Loaded {len(results)} v4 result entries.")
    elif args.v3 and os.path.exists(config.RESULTS_V3_PATH):
        results = load_results(version="v3")
        print(f"Loaded {len(results)} v3 result entries.")
    else:
        results = load_results(version="v2")
        print(f"Loaded {len(results)} result entries.")

    print_summary_table(results)
    plot_heatmap(results, config.PLOT_DIR)
    plot_logprob_gap(results, config.PLOT_DIR)
    plot_perplexity_vs_gap(results, config.PLOT_DIR)
    plot_compounding_line(results, config.PLOT_DIR)
    plot_diversity_over_generations(results, config.PLOT_DIR)
    run_statistical_tests(results)

    # V3-specific plots
    has_v3 = any("logprob_gap_critical" in r for r in results)
    if has_v3:
        plot_logprob_gap_critical(results, config.PLOT_DIR)
        plot_gap_comparison(results, config.PLOT_DIR)
        plot_decomposed_critical(results, config.PLOT_DIR)

    # V4-specific plots
    has_v4 = any("v4_config" in r for r in results)
    if has_v4:
        plot_v4_lr_comparison(results, config.PLOT_DIR)
        plot_v4_decomposed_by_config(results, config.PLOT_DIR)

    # Training curves (works for both V2 and V4 logs)
    plot_training_curves(config.PLOT_DIR)

    # Per-item/category analysis (requires V4 results)
    if args.v4:
        run_per_item_analysis(config.PLOT_DIR)

    print(f"\nAll plots saved to: {config.PLOT_DIR}")


if __name__ == "__main__":
    main()
