"""
Simulation: Adaptive Action Token Refinement for OAT
=====================================================

Simulates the hypothesis that adaptive early stopping during OAT's
prefix-based action decoding can reduce token usage on simple actions
while preserving quality on complex ones.

Key model assumptions (grounded in OAT paper):
  - VQ tokenizer produces coarse-to-fine tokens; each additional token
    refines the reconstruction (geometric-like error decay).
  - For simple actions (low complexity / small L2 norm), the residual
    information per token is smaller → faster convergence.
  - Stopping criterion: err_k < eps * ||a_initial||
    (absolute threshold scaled to the action magnitude).

No external dependencies — runs on Python 3.8+.

Usage:
    python simulate_adaptive_oat.py
    python simulate_adaptive_oat.py --seed 42 --n_episodes 500 --eps 0.05
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
from dataclasses import dataclass, asdict
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Action trajectory generation
# ---------------------------------------------------------------------------

@dataclass
class ActionStep:
    step_id: int
    complexity: float        # 0=trivial, 1=highly complex
    true_action: List[float] # 7-DoF robot arm action vector


def generate_trajectory(
    n_steps: int = 20,
    task_type: str = "simple",
    rng: random.Random = None,
) -> List[ActionStep]:
    """
    Generate a synthetic robot action trajectory.

    task_type:
      'simple'  - hold position / small corrective motions
      'complex' - pick-and-place, rotation, reaching
      'mixed'   - alternating between low and high-activity phases
    """
    if rng is None:
        rng = random.Random()

    steps = []
    for i in range(n_steps):
        if task_type == "simple":
            complexity = rng.uniform(0.05, 0.25)
        elif task_type == "complex":
            complexity = rng.uniform(0.60, 0.95)
        else:
            # sinusoidal activity profile: low in middle, peaks at transitions
            phase = math.sin(math.pi * i / n_steps)
            complexity = 0.10 + 0.80 * abs(phase)

        action = [rng.gauss(0.0, 0.08 + complexity * 0.85) for _ in range(7)]
        steps.append(ActionStep(step_id=i, complexity=complexity, true_action=action))
    return steps


# ---------------------------------------------------------------------------
# VQ encoder model
# ---------------------------------------------------------------------------

def vq_encode(
    action: List[float],
    n_tokens: int,
    rng: random.Random,
) -> List[float]:
    """
    Simulate coarse-to-fine VQ encoding (as in OAT).

    Returns reconstruction errors at each prefix depth 1..n_tokens.

    Model:
      - Simple actions (low L2 norm): each token captures a larger fraction
        of the remaining variance → fast decay.
      - Complex actions (high L2 norm): slower convergence per token.
    """
    action_norm = _l2_norm(action)
    # Complexity inferred from action magnitude (normalised to ~[0,1])
    inferred_complexity = min(1.0, action_norm / (math.sqrt(7) * 0.70))

    # Initial error (after first token): scales with complexity
    initial_fraction = 0.35 + 0.30 * inferred_complexity + rng.uniform(-0.05, 0.05)
    remaining_error = action_norm * max(0.05, initial_fraction)

    # Decay per token: faster for simple actions
    # simple (complexity~0)  → decay ~ 0.30–0.45 (large reduction per step)
    # complex (complexity~1) → decay ~ 0.62–0.75 (small reduction per step)
    base_decay = 0.30 + 0.40 * inferred_complexity

    errors = []
    for _ in range(n_tokens):
        decay = max(0.15, base_decay + rng.gauss(0, 0.04))
        remaining_error *= decay
        errors.append(remaining_error)

    return errors


def _l2_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


# ---------------------------------------------------------------------------
# Fixed OAT baseline
# ---------------------------------------------------------------------------

def fixed_oat(
    traj: List[ActionStep],
    n_tokens: int,
    rng: random.Random,
) -> Tuple[float, float, int]:
    """Returns (mean_reconstruction_error, mean_complexity, n_tokens)."""
    recon_errors = []
    for step in traj:
        errors = vq_encode(step.true_action, n_tokens, rng)
        recon_errors.append(errors[-1])
    return (
        statistics.mean(recon_errors),
        statistics.mean(s.complexity for s in traj),
        n_tokens,
    )


# ---------------------------------------------------------------------------
# Adaptive OAT
# ---------------------------------------------------------------------------

def adaptive_oat(
    traj: List[ActionStep],
    max_tokens: int,
    eps: float,
    min_tokens: int = 1,
    rng: random.Random = None,
) -> Tuple[float, float, float]:
    """
    Adaptive token refinement with early stopping.

    Stopping criterion:
        stop when err_k < eps * action_norm
    (absolute threshold scaled to the magnitude of the action, so simple
    low-magnitude actions converge earlier than complex high-magnitude ones.)

    Returns (mean_reconstruction_error, mean_complexity, mean_tokens_used).
    """
    if rng is None:
        rng = random.Random()

    recon_errors = []
    tokens_used = []

    for step in traj:
        action_norm = _l2_norm(step.true_action)
        threshold = eps * action_norm

        errors = vq_encode(step.true_action, max_tokens, rng)

        chosen_k = max_tokens
        for k in range(min_tokens, max_tokens + 1):
            if errors[k - 1] < threshold:
                chosen_k = k
                break

        recon_errors.append(errors[chosen_k - 1])
        tokens_used.append(chosen_k)

    return (
        statistics.mean(recon_errors),
        statistics.mean(s.complexity for s in traj),
        statistics.mean(tokens_used),
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode: int
    task_type: str
    mean_complexity: float
    oat1_error: float
    oat2_error: float
    oat4_error: float
    oat8_error: float
    adaptive_error: float
    adaptive_tokens: float
    adaptive_eps: float


def run_experiment(
    n_episodes: int = 300,
    n_steps: int = 20,
    eps: float = 0.05,
    seed: int = 42,
) -> List[EpisodeResult]:
    rng = random.Random(seed)
    task_types = ["simple", "complex", "mixed"]
    results = []

    for ep in range(n_episodes):
        task = task_types[ep % len(task_types)]
        ep_seed = rng.randint(0, 2 ** 31)
        ep_rng = random.Random(ep_seed)
        traj = generate_trajectory(n_steps=n_steps, task_type=task, rng=ep_rng)

        def _rng():
            return random.Random(ep_rng.randint(0, 2 ** 31))

        oat1_err, _, _ = fixed_oat(traj, 1, _rng())
        oat2_err, _, _ = fixed_oat(traj, 2, _rng())
        oat4_err, _, _ = fixed_oat(traj, 4, _rng())
        oat8_err, _, _ = fixed_oat(traj, 8, _rng())
        ad_err, ad_cplx, ad_tok = adaptive_oat(traj, max_tokens=8, eps=eps, rng=_rng())

        results.append(EpisodeResult(
            episode=ep,
            task_type=task,
            mean_complexity=ad_cplx,
            oat1_error=oat1_err,
            oat2_error=oat2_err,
            oat4_error=oat4_err,
            oat8_error=oat8_err,
            adaptive_error=ad_err,
            adaptive_tokens=ad_tok,
            adaptive_eps=eps,
        ))

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_summary(results: List[EpisodeResult]) -> dict:
    summary = {}
    for task in ["simple", "complex", "mixed", "all"]:
        subset = results if task == "all" else [r for r in results if r.task_type == task]
        if not subset:
            continue

        def _mean(attr):
            return statistics.mean(getattr(r, attr) for r in subset)

        def _std(attr):
            vals = [getattr(r, attr) for r in subset]
            return statistics.stdev(vals) if len(vals) > 1 else 0.0

        avg_tok = _mean("adaptive_tokens")
        avg_ad_err = _mean("adaptive_error")
        avg_oat8_err = _mean("oat8_error")

        token_saving_pct = (8.0 - avg_tok) / 8.0 * 100.0
        quality_delta_pct = (avg_ad_err - avg_oat8_err) / (avg_oat8_err + 1e-12) * 100.0

        summary[task] = {
            "n_episodes": len(subset),
            "mean_complexity": round(_mean("mean_complexity"), 3),
            "oat1_error": round(_mean("oat1_error"), 5),
            "oat2_error": round(_mean("oat2_error"), 5),
            "oat4_error": round(_mean("oat4_error"), 5),
            "oat8_error": round(avg_oat8_err, 5),
            "adaptive_error": round(avg_ad_err, 5),
            "adaptive_error_std": round(_std("adaptive_error"), 5),
            "adaptive_tokens_mean": round(avg_tok, 2),
            "adaptive_tokens_std": round(_std("adaptive_tokens"), 2),
            "token_saving_pct": round(token_saving_pct, 1),
            "quality_delta_vs_oat8_pct": round(quality_delta_pct, 1),
        }
    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 76)
    print("  Adaptive Action Token Refinement — Simulation Results")
    print("=" * 76)
    hdr = (f"  {'Task':<9} {'Cmplx':>6} {'OAT1':>8} {'OAT2':>8} "
           f"{'OAT4':>8} {'OAT8':>8} {'Adapt':>8} {'Tok':>5} {'Save%':>6} {'Δerr%':>7}")
    print(hdr)
    print("  " + "-" * 72)
    for task in ["simple", "complex", "mixed", "all"]:
        if task not in summary:
            continue
        s = summary[task]
        print(
            f"  {task:<9} "
            f"{s['mean_complexity']:>6.2f} "
            f"{s['oat1_error']:>8.5f} "
            f"{s['oat2_error']:>8.5f} "
            f"{s['oat4_error']:>8.5f} "
            f"{s['oat8_error']:>8.5f} "
            f"{s['adaptive_error']:>8.5f} "
            f"{s['adaptive_tokens_mean']:>5.1f} "
            f"{s['token_saving_pct']:>5.1f}% "
            f"{s['quality_delta_vs_oat8_pct']:>+6.1f}%"
        )
    print("=" * 76)
    print("  Tok = mean tokens used | Save% = savings vs OAT8")
    print("  Δerr% = adaptive vs OAT8 reconstruction error (+ = slightly worse)\n")


def ascii_chart(summary: dict):
    print("  Token savings by task type (eps=0.05, max=8 tokens)")
    print("  " + "-" * 55)
    for task in ["simple", "complex", "mixed"]:
        if task not in summary:
            continue
        s = summary[task]
        pct = s["token_saving_pct"]
        bar = "█" * int(pct / 2.5)
        tok = s["adaptive_tokens_mean"]
        print(f"  {task:<8} |{bar:<20}| {pct:.1f}%  "
              f"({tok:.1f}/8 tokens used)")
    print()


def save_results(
    results: List[EpisodeResult],
    summary: dict,
    out_dir: str,
    seed: int = 42,
    n_steps: int = 20,
    eps: float = 0.05,
):
    import datetime
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "episodes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            row = {k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in asdict(r).items()}
            writer.writerow(row)

    task_types = ["simple", "complex", "mixed", "all"]
    ordered_summary = {t: summary[t] for t in task_types if t in summary}
    ordered_summary["meta"] = {
        "seed": seed,
        "n_episodes_per_task": len([r for r in results if r.task_type == "simple"]),
        "n_steps": n_steps,
        "eps": eps,
        "max_tokens": 8,
        "min_tokens": 1,
        "run_date": datetime.date.today().isoformat(),
    }

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ordered_summary, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Stopping threshold: stop when err < eps * action_norm")
    parser.add_argument("--out_dir", type=str, default="../results")
    args = parser.parse_args()

    print(f"\n  seed={args.seed}  episodes={args.n_episodes}  "
          f"steps={args.n_steps}  eps={args.eps}")

    results = run_experiment(
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        eps=args.eps,
        seed=args.seed,
    )
    summary = compute_summary(results)
    print_summary(summary)
    ascii_chart(summary)
    save_results(results, summary, args.out_dir,
                 seed=args.seed, n_steps=args.n_steps, eps=args.eps)
    print("  Done.\n")


if __name__ == "__main__":
    main()
