# scripts/plot_prompt_acc.py
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=str, required=True, help="Path to rollouts.jsonl")
    ap.add_argument("--prompt_id", type=str, required=True, help="Hashed prompt id stored in each record")
    ap.add_argument("--num_generations", type=int, required=True, help="K (e.g. 16)")
    ap.add_argument("--out", type=str, default=None, help="If set, save plot here")
    ap.add_argument("--allow_partial", action="store_true",
                    help="Allow steps that have != K rollouts (e.g., truncated runs).")
    args = ap.parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        raise FileNotFoundError(rollouts_path)

    # collect correct flags per step for this prompt_id
    by_step = defaultdict(list)
    for row in iter_jsonl(rollouts_path):
        if row.get("prompt_id") != args.prompt_id:
            continue
        step = int(row["step"])
        by_step[step].append(1 if row.get("correct", False) else 0)

    if not by_step:
        raise RuntimeError(f"No rows found for prompt_id={args.prompt_id}")

    steps = sorted(by_step.keys())
    acc = []

    for s in steps:
        xs = by_step[s]
        if (not args.allow_partial) and (len(xs) != args.num_generations):
            raise RuntimeError(
                f"Step {s} has {len(xs)} rollouts for prompt_id={args.prompt_id}, "
                f"expected {args.num_generations}. Use --allow_partial to ignore."
            )
        acc.append(sum(xs) / len(xs))

    # plot
    plt.figure()
    plt.plot(steps, acc, marker="o", linestyle="-")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("optimizer step (global_step)")
    plt.ylabel(f"accuracy (avg@{args.num_generations})")
    plt.title(f"Accuracy vs time (prompt_id={args.prompt_id})")
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to: {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()