from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml


OPTIMIZER_COLORS = {
    "ADAMW":   "#1f77b4",
    "ADAM":    "#1f77b4",
    "RMSPROP": "#ff7f0e",
    "LION":    "#2ca02c",
    "SGD":     "#d62728",
}


def load_history(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "history.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def infer_optimizer_name(run_dir: Path) -> str:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        name = cfg.get("optimizer", {}).get("name", "")
        if name:
            return name.upper()
    return run_dir.name.upper()


def color_for(name: str) -> str:
    return OPTIMIZER_COLORS.get(name.upper(), "#7f7f7f")


def _series(records: List[Dict[str, Any]], split: str, key: str
            ) -> Tuple[List[int], List[float]]:
    epochs, values = [], []
    for r in records:
        sub = r.get(split, {})
        v = sub.get(key)
        if v is None:
            continue
        epochs.append(sub.get("epoch"))
        values.append(float(v))
    return epochs, values


def _line_plot(runs: Dict[str, List[Dict[str, Any]]], split: str, key: str,
               title: str, ylabel: str, note: str, out_path: Path,
               transform: Optional[Callable[[float], float]] = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for name, records in runs.items():
        epochs, values = _series(records, split, key)
        if not epochs:
            continue
        if transform is not None:
            values = [transform(v) for v in values]
        ax.plot(epochs, values, marker="o", linewidth=2,
                color=color_for(name), label=name)
        plotted = True

    ax.set_title(f"{title}\n({note})", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def _bar_plot(values: Dict[str, float], title: str, ylabel: str, note: str,
              out_path: Path, fmt: str = "%.4f",
              ascending: bool = True) -> None:
    if not values:
        return
    items = sorted(values.items(),
                   key=lambda kv: kv[1],
                   reverse=not ascending)
    names = [k for k, _ in items]
    nums = [v for _, v in items]
    colors = [color_for(n) for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, nums, color=colors)
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
    ax.set_title(f"{title}\n({note})", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def _best(records: List[Dict[str, Any]], split: str, key: str,
          maximize: bool) -> Optional[float]:
    vals = [r.get(split, {}).get(key) for r in records]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals) if maximize else min(vals)


def _final_train(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    for r in reversed(records):
        v = r.get("train", {}).get(key)
        if v is not None:
            return float(v)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DPO optimizer comparison.")
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Explicit list of run directories.")
    parser.add_argument("--output-dir", default="outputs/dpo_comparison_plots",
                        help="Where to write plots and summary.json.")
    parser.add_argument("--report-dir", default="docs/report_figures_dpo",
                        help="Optional report-ready copy directory.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: Dict[str, List[Dict[str, Any]]] = {}
    for run_dir in args.run_dirs:
        rd = Path(run_dir)
        records = load_history(rd)
        if not records:
            print(f"[skip] no history at {rd}")
            continue
        name = infer_optimizer_name(rd)
        runs[name] = records
        print(f"Loaded {len(records)} epochs from {rd.name} ({name})")

    if not runs:
        print("No runs to plot.")
        return

    # ───────── Validation curves ─────────
    _line_plot(runs, "val", "mean_dpo_loss",
               "Validation DPO Loss", "DPO Loss", "lower is better",
               out_dir / "fig01_val_dpo_loss.png")
    _line_plot(runs, "val", "preference_accuracy",
               "Validation Preference Accuracy", "Preference Accuracy",
               "higher is better",
               out_dir / "fig02_val_preference_accuracy.png")
    _line_plot(runs, "val", "reward_margin",
               "Validation Reward Margin", "Reward Margin (β · Δ)",
               "higher is better",
               out_dir / "fig03_val_reward_margin.png")
    _line_plot(runs, "val", "logprob_gap",
               "Validation Log-prob Gap", "logπ(chosen) − logπ(rejected)",
               "higher is better",
               out_dir / "fig04_val_logprob_gap.png")
    _line_plot(runs, "val", "tokens_per_sec",
               "Validation Throughput", "Tokens / sec",
               "higher is better",
               out_dir / "fig05_val_tokens_per_sec.png")

    # ───────── Training curves ─────────
    _line_plot(runs, "train", "mean_dpo_loss",
               "Training DPO Loss", "DPO Loss", "lower is better",
               out_dir / "fig06_train_dpo_loss.png")
    _line_plot(runs, "train", "preference_accuracy",
               "Training Preference Accuracy", "Preference Accuracy",
               "higher is better",
               out_dir / "fig07_train_preference_accuracy.png")
    _line_plot(runs, "train", "reward_margin",
               "Training Reward Margin", "Reward Margin (β · Δ)",
               "higher is better",
               out_dir / "fig08_train_reward_margin.png")
    _line_plot(runs, "train", "tokens_per_sec",
               "Training Throughput", "Tokens / sec",
               "higher is better",
               out_dir / "fig09_train_tokens_per_sec.png")

    # ───────── Per-step timing breakdown (train) ─────────
    _line_plot(runs, "train", "mean_step_time_sec",
               "Mean Step Time", "seconds / step", "lower is better",
               out_dir / "fig10_train_step_time.png")
    _line_plot(runs, "train", "mean_forward_time_sec",
               "Mean Forward Time", "seconds / step", "lower is better",
               out_dir / "fig11_train_forward_time.png")
    _line_plot(runs, "train", "mean_backward_time_sec",
               "Mean Backward Time", "seconds / step", "lower is better",
               out_dir / "fig12_train_backward_time.png")
    _line_plot(runs, "train", "mean_optimizer_time_sec",
               "Mean Optimizer Step Time", "seconds / step",
               "lower is better",
               out_dir / "fig13_train_optimizer_time.png")

    # ───────── Memory curves ─────────
    mb = lambda v: v / (1024 ** 2)
    _line_plot(runs, "train", "optimizer_state_bytes",
               "Optimizer State Memory", "MB", "lower is better",
               out_dir / "fig14_optimizer_state_bytes_curve.png",
               transform=mb)
    _line_plot(runs, "train", "gpu_peak_allocated_bytes",
               "Peak GPU Memory", "MB", "lower is better",
               out_dir / "fig15_gpu_peak_memory_curve.png",
               transform=mb)

    # ───────── Best-epoch summary bars ─────────
    best_loss = {n: _best(rs, "val", "mean_dpo_loss", maximize=False)
                 for n, rs in runs.items()}
    _bar_plot({k: v for k, v in best_loss.items() if v is not None},
              "Best Validation DPO Loss", "DPO Loss", "lower is better",
              out_dir / "fig16_best_val_dpo_loss_bar.png",
              fmt="%.4f", ascending=True)

    best_acc = {n: _best(rs, "val", "preference_accuracy", maximize=True)
                for n, rs in runs.items()}
    _bar_plot({k: v for k, v in best_acc.items() if v is not None},
              "Best Validation Preference Accuracy", "Preference Accuracy",
              "higher is better",
              out_dir / "fig17_best_val_preference_accuracy_bar.png",
              fmt="%.4f", ascending=False)

    best_margin = {n: _best(rs, "val", "reward_margin", maximize=True)
                   for n, rs in runs.items()}
    _bar_plot({k: v for k, v in best_margin.items() if v is not None},
              "Best Validation Reward Margin", "Reward Margin (β · Δ)",
              "higher is better",
              out_dir / "fig18_best_val_reward_margin_bar.png",
              fmt="%.4f", ascending=False)

    final_opt_time = {n: _final_train(rs, "mean_optimizer_time_sec")
                      for n, rs in runs.items()}
    _bar_plot({k: v * 1000 for k, v in final_opt_time.items() if v is not None},
              "Final-Epoch Optimizer Step Time", "ms / step",
              "lower is better",
              out_dir / "fig19_final_optimizer_time_bar.png",
              fmt="%.3f", ascending=True)

    final_state_mb = {n: _final_train(rs, "optimizer_state_bytes")
                      for n, rs in runs.items()}
    _bar_plot({k: v / (1024 ** 2) for k, v in final_state_mb.items() if v is not None},
              "Optimizer State Memory (final epoch)", "MB",
              "lower is better",
              out_dir / "fig20_final_optimizer_state_bar.png",
              fmt="%.2f", ascending=True)

    # ───────── summary.json for the report ─────────
    summary = {}
    for name, records in runs.items():
        summary[name] = {
            "epochs": len(records),
            "best_val_dpo_loss":          best_loss.get(name),
            "best_val_preference_accuracy": best_acc.get(name),
            "best_val_reward_margin":     best_margin.get(name),
            "final_train_optimizer_time_ms":
                (final_opt_time.get(name) * 1000) if final_opt_time.get(name) is not None else None,
            "final_optimizer_state_mb":
                (final_state_mb.get(name) / (1024 ** 2)) if final_state_mb.get(name) is not None else None,
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: summary.json")

    # ───────── Mirror to report_figures_dpo/ ─────────
    if args.report_dir:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted(out_dir.glob("fig*.png")):
            shutil.copy2(png, report_dir / png.name)
        shutil.copy2(out_dir / "summary.json", report_dir / "summary.json")
        print(f"Mirrored figures to {report_dir}/")

    print(f"\nDone. {len(list(out_dir.glob('fig*.png')))} figures in {out_dir}/")


if __name__ == "__main__":
    main()
