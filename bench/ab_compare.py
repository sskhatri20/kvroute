"""Step 10 — compare the two arms captured by ab_run.py.

Reports each run's mean with a 95% CI (normal approximation:
mean +/- 1.96 * stdev / sqrt(n)) and whether the two arms' CIs overlap
across all repeated runs, per the guide's acceptance check.
"""
import json
import math
from pathlib import Path

RESULTS = Path("bench/results")


def ci95(run: dict) -> tuple[float, float]:
    margin = 1.96 * run["stdev"] / math.sqrt(run["n"]) if run["n"] > 1 else 0.0
    return run["mean"] - margin, run["mean"] + margin


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def main() -> None:
    a_path = RESULTS / "ab-round_robin.json"
    b_path = RESULTS / "ab-prefix_aware.json"
    if not a_path.exists() or not b_path.exists():
        raise SystemExit("run bench/ab_run.py for both strategies first")

    arm_a = json.loads(a_path.read_text())
    arm_b = json.loads(b_path.read_text())

    any_overlap = False
    for i, (run_a, run_b) in enumerate(zip(arm_a["runs"], arm_b["runs"])):
        ci_a, ci_b = ci95(run_a), ci95(run_b)
        overlap = overlaps(ci_a, ci_b)
        any_overlap = any_overlap or overlap
        print(
            f"run {i}: round_robin={run_a['mean']*1000:.1f}ms {ci_a[0]*1000:.1f}-{ci_a[1]*1000:.1f}  "
            f"prefix_aware={run_b['mean']*1000:.1f}ms {ci_b[0]*1000:.1f}-{ci_b[1]*1000:.1f}  "
            f"overlap={overlap}"
        )

    if any_overlap:
        print("\nNo consistent TTFT delta (CIs overlap on at least one run). See Step 10's kill criterion.")
    else:
        print("\nNon-overlapping CIs across all runs: measurable TTFT delta.")


if __name__ == "__main__":
    main()
