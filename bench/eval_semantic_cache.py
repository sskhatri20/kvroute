"""Step 8 acceptance check: threshold-vs-hit-rate and threshold-vs-false-hit curves.

Usage:
    python bench/eval_semantic_cache.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings import cosine_similarity, embed  # noqa: E402

EVAL_SET = Path(__file__).parent / "semantic_eval.json"
OUT = Path(__file__).parent / "results" / "semantic-eval.json"
THRESHOLDS = [round(0.5 + 0.02 * i, 2) for i in range(26)]


def main() -> None:
    pairs = json.loads(EVAL_SET.read_text())
    scored = [
        {"should_hit": pair["should_hit"], "similarity": cosine_similarity(embed(pair["a"]), embed(pair["b"]))}
        for pair in pairs
    ]

    positives = [p for p in scored if p["should_hit"]]
    negatives = [p for p in scored if not p["should_hit"]]

    curve = []
    for threshold in THRESHOLDS:
        hit_rate = sum(p["similarity"] >= threshold for p in positives) / len(positives)
        false_hit_rate = sum(p["similarity"] >= threshold for p in negatives) / len(negatives)
        curve.append({"threshold": threshold, "hit_rate": hit_rate, "false_hit_rate": false_hit_rate})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_pairs": len(pairs), "curve": curve}, indent=2))

    for point in curve:
        print(f"threshold={point['threshold']:.2f}  hit_rate={point['hit_rate']:.2f}  "
              f"false_hit_rate={point['false_hit_rate']:.2f}")


if __name__ == "__main__":
    main()
