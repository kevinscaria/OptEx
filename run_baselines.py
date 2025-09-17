import argparse
import json
import random

import numpy as np

from optex.src.agent import BaseAgent
from optex.src.buffer import BaseExemplarBuffer
from optex.src.exemplars import (
    NoExemplars, RandomExemplars, SemanticExemplars, RecencyExemplars, SuccessOnlyExemplars
)

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except Exception:
    _HAS_ST = False


def make_selector(name: str, k: int, embedder):
    name = name.lower()
    if name == "none":
        return NoExemplars(k, embedder)
    if name == "random":
        return RandomExemplars(k, embedder)
    if name == "semantic":
        return SemanticExemplars(k, embedder)
    if name == "recency":
        return RecencyExemplars(k, embedder)
    if name == "success":
        return SuccessOnlyExemplars(k, embedder)
    raise ValueError(f"Unknown selector: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="none",
                        choices=["none", "random", "semantic", "recency", "success"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--k", type=int, default=4, help="# exemplars in context")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="runs/baseline_results.jsonl")
    parser.add_argument("--embedder", type=str, default="all-mpnet-base-v2")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Optional semantic embedder
    embedder = None
    if args.baseline == "semantic":
        if not _HAS_ST:
            print("[warn] sentence-transformers not installed; falling back to random.")
        else:
            embedder = SentenceTransformer(args.embedder)

    buffer = BaseExemplarBuffer(max_size=2000)
    selector = make_selector(args.baseline, args.k, embedder)
    agent = BaseAgent(selector, k=args.k)

    # TODO: Replace with real benchmark query generator (e.g., WebShop tasks)
    def sample_instruction(i: int) -> str:
        # trivially mock; replace it with dataset sampling
        goals = [
            "Find a black office chair under $200 with wheels",
            "Buy a budget gaming mouse",
            "Locate a 4K monitor above 27 inches",
            "Get the cheapest noise-cancelling headphones",
            "Find a vegan cookbook under $25"
        ]
        return goals[i % len(goals)]

    # Run episodes
    stats = {"success": 0, "total": 0}
    records = []
    for i in range(args.episodes):
        instr = sample_instruction(i)
        ep = agent.run_episode(buffer, instr)

        # If using semantic baseline, precompute embedding for future retrieval
        if embedder is not None and ep.embedding is None:
            ep.embedding = embedder.encode([ep.instruction])[0]

        buffer.add(ep)
        stats["success"] += int(ep.reward > 0.5)
        stats["total"] += 1

        rec = {
            "i": i,
            "instruction": ep.instruction,
            "plan": ep.plan,
            "actions": ep.actions,
            "reward": ep.reward,
            "timestamp": ep.timestamp,
            "baseline": args.baseline,
            "k": args.k
        }
        records.append(rec)
        if (i + 1) % 50 == 0:
            sr = stats["success"] / stats["total"]
            print(f"[{args.baseline}] episode={i + 1} success_rate={sr:.3f}")

    # Save JSONL
    import os, pathlib
    pathlib.Path(os.path.dirname(args.save) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Done. Saved to {args.save}")
    print(f"Final success rate = {stats['success'] / stats['total']:.3f}")


if __name__ == "__main__":
    main()
