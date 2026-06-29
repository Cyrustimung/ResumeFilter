"""
Core Ranking Pipeline: Score candidates and produce ranked output.
"""

import json
import csv
import time
from pathlib import Path

from backend.config import TOP_K
from backend.pipeline.feature_extraction import extract_features
from backend.scoring.fusion import compute_final_score
from backend.scoring.reasoning import generate_reasoning


def load_candidates(filepath: str) -> list[dict]:
    """Load candidates from .json or .jsonl file."""
    path = Path(filepath)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    # JSONL
    candidates: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return candidates


def rank_candidates(candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Score and rank a list of candidate dicts.
    Returns top_k results sorted by score descending.
    """
    results: list[dict] = []
    for cand in candidates:
        features = extract_features(cand)
        result = compute_final_score(features)
        result["_raw_candidate"] = cand
        results.append(result)

    # Sort: highest score first (full precision), tie-break by candidate_id ascending
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # Take top K
    top = results[:top_k]

    # Assign ranks and generate reasoning with full audit
    output: list[dict] = []
    for rank, result in enumerate(top, start=1):
        reasoning = generate_reasoning(result)
        audit = result.get("audit", {})

        output.append({
            "candidate_id": result["candidate_id"],
            "rank": rank,
            "score": result["final_score"],
            "reasoning": reasoning,
            "integrity_gate": result["integrity_gate"],
            "evidence_score": result["evidence_score"],
            "reliability_score": result["reliability_score"],
            "credibility_score": result["credibility_score"],
            "sub_scores": result["sub_scores"],
            "profile": result["features"]["profile"],
            "audit": {
                "verdict": audit.get("verdict", ""),
                "score_breakdown": audit.get("score_breakdown", {}),
                "positive_signals": audit.get("positive_signals", []),
                "negative_signals": audit.get("negative_signals", []),
                "evidence_trail": audit.get("evidence_trail", []),
            },
            "raw_candidate": result.get("_raw_candidate", {}),
        })

    return output


def rank_to_csv(candidates: list[dict], output_path: str, top_k: int = TOP_K) -> list[dict]:
    """Full pipeline: candidates → ranked CSV file."""
    start = time.time()
    print(f"Scoring {len(candidates)} candidates...")

    ranked = rank_candidates(candidates, top_k)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in ranked:
            writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.6f}", r["reasoning"]])

    elapsed = time.time() - start
    print(f"Done! {len(ranked)} candidates ranked in {elapsed:.1f}s -> {output_path}")
    return ranked
