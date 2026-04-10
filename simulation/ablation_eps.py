"""
Ablation: effect of stopping threshold (eps) on token efficiency vs. quality.

This script is synchronized with the report and with the JSON artifacts stored in
`../results/ablation_eps.json`.

Reference setup used in the report:
    - seed = 42
    - n_episodes = 300 total (100 per task type)
    - n_steps = 20
    - eps in {0.02, 0.05, 0.10, 0.15, 0.20}

Usage:
    python ablation_eps.py
    python ablation_eps.py --seed 42 --n_episodes 300 --n_steps 20
"""

import argparse
import json
import os
import sys

# Allow importing from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from simulate_adaptive_oat import run_experiment, compute_summary


EPS_VALUES = [0.02, 0.05, 0.10, 0.15, 0.20]


def run_ablation(n_episodes: int = 300, n_steps: int = 20, seed: int = 42) -> dict:
    ablation_results = {}
    for eps in EPS_VALUES:
        results = run_experiment(
            n_episodes=n_episodes,
            n_steps=n_steps,
            eps=eps,
            seed=seed,
        )
        summary = compute_summary(results)
        ablation_results[eps] = summary
    return ablation_results


def print_ablation_table(ablation: dict):
    print("\n" + "=" * 70)
    print("  Ablation Study: Stopping Threshold (eps) — All Task Types")
    print("=" * 70)
    print(f"  {'eps':>6}  {'Tokens':>7}  {'Save%':>6}  {'Δerr%':>7}  "
          f"{'AdaptErr':>9}  {'OAT8Err':>9}")
    print("  " + "-" * 60)
    for eps in EPS_VALUES:
        s = ablation[eps]["all"]
        print(
            f"  {eps:>6.2f}  "
            f"{s['adaptive_tokens_mean']:>7.2f}  "
            f"{s['token_saving_pct']:>5.1f}%  "
            f"{s['quality_delta_vs_oat8_pct']:>+6.1f}%  "
            f"{s['adaptive_error']:>9.4f}  "
            f"{s['oat8_error']:>9.4f}"
        )
    print("=" * 70)
    print()

    print("  Per-task breakdown at eps=0.05 (recommended):")
    print("  " + "-" * 60)
    s015 = ablation[0.05]
    for task in ["simple", "complex", "mixed"]:
        if task not in s015:
            continue
        s = s015[task]
        print(
            f"  {task:<8}  tokens={s['adaptive_tokens_mean']:.1f}/8  "
            f"save={s['token_saving_pct']:.1f}%  "
            f"Δerr={s['quality_delta_vs_oat8_pct']:+.1f}%"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Ablation study over eps values")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="../results")
    args = parser.parse_args()

    print(f"\n  Running ablation study: seed={args.seed}, "
          f"episodes={args.n_episodes}, eps={EPS_VALUES}")

    ablation = run_ablation(
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    print_ablation_table(ablation)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "ablation_eps.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in ablation.items()}, f, indent=2)
    print(f"  Ablation results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
