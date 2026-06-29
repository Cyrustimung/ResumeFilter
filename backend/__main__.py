"""CLI Entry Point: python -m backend --candidates X --out Y"""

import argparse
import sys
import time
import csv
from pathlib import Path

from backend.config import SAMPLE_CANDIDATES_PATH


def main():
    parser = argparse.ArgumentParser(description="Canditly - Intelligent Candidate Ranking Engine")
    parser.add_argument("--candidates", help="Path to candidates .json/.jsonl file")
    parser.add_argument("--out", default="./Team_Circle.csv", help="Output CSV path")
    parser.add_argument("--provided", action="store_true", help="Use bundled sample dataset")
    parser.add_argument("--top", type=int, default=100, help="Top N candidates to output")
    args = parser.parse_args()

    print()
    print("=" * 50)
    print("  Canditly - Intelligent Candidate Ranking")
    print("  Team Circle | Redrob Hackathon 2026")
    print("=" * 50)
    print()

    # Determine input
    if args.provided:
        path = SAMPLE_CANDIDATES_PATH
    elif args.candidates:
        path = args.candidates
    else:
        print("  Usage: python -m backend --candidates <file>")
        sys.exit(1)

    if not Path(path).exists():
        print(f"  [ERROR] File not found: {path}")
        sys.exit(1)

    # Load
    sys.stdout.write("  [*] Loading candidates...")
    sys.stdout.flush()
    from backend.ranker import load_candidates
    t0 = time.time()
    candidates = load_candidates(path)
    total = len(candidates)
    print(f" {total:,} loaded ({time.time()-t0:.1f}s)")

    # Import engine
    sys.stdout.write("  [*] Initializing scoring...")
    sys.stdout.flush()
    from backend.pipeline.feature_extraction import extract_features
    from backend.scoring.fusion import compute_final_score
    from backend.scoring.reasoning import generate_reasoning
    print(" ready")
    print()

    # Score
    top_k = min(args.top, total)
    print(f"  [*] Scoring {total:,} candidates...")
    print(f"      Integrity -> Tech Fit (65%) -> Feasibility (20%) -> Verification (15%)")
    print()

    results = []
    honeypots = 0
    t1 = time.time()

    # Update interval: every 1% or at least every 0.5 seconds
    update_every = max(1, total // 100)
    last_time = t1

    for i, cand in enumerate(candidates):
        features = extract_features(cand)
        result = compute_final_score(features)
        result["_raw_candidate"] = cand
        results.append(result)
        if result["integrity_gate"] == 0.0:
            honeypots += 1

        # Smooth progress: update every 1% or every 0.5s
        now = time.time()
        if (i + 1) % update_every == 0 or (now - last_time) > 0.5 or i == total - 1:
            last_time = now
            pct = (i + 1) / total
            elapsed = now - t1
            remaining = (elapsed / pct) * (1 - pct) if pct > 0 else 0
            filled = int(30 * pct)
            bar = "#" * filled + "-" * (30 - filled)
            sys.stdout.write(f"\r      [{bar}] {pct*100:.0f}% | {i+1:,}/{total:,} | {elapsed:.1f}s | ~{remaining:.0f}s left")
            sys.stdout.flush()

    scoring_time = time.time() - t1
    sys.stdout.write(f"\r      [{'#'*30}] 100% | {total:,}/{total:,} | {scoring_time:.1f}s                          \n")
    print()

    if honeypots > 0:
        print(f"  [!] {honeypots:,} honeypot profiles flagged")

    # Rank and save
    sys.stdout.write("  [*] Ranking and generating reasoning...")
    sys.stdout.flush()
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top = results[:top_k]

    output_path = args.out
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, result in enumerate(top, start=1):
            reasoning = generate_reasoning(result)
            writer.writerow([result["candidate_id"], rank, f"{result['final_score']:.6f}", reasoning])

    print(f" done")
    print(f"  [+] Saved: {output_path}")
    print()

    # Results preview
    print("  Rank | Candidate    | Score  | Title")
    print("  -----|--------------|--------|-------------------------------")
    show = min(15, top_k)
    for rank, result in enumerate(top[:show], start=1):
        cid = result["candidate_id"]
        score = result["final_score"]
        title = result["features"]["profile"]["current_title"][:30]
        print(f"  {rank:>4} | {cid} | {score:.4f} | {title}")
    if top_k > show:
        print(f"   ... | ({top_k - show} more in {output_path})")
    print()

    # Summary
    total_time = time.time() - t0
    print(f"  Done! {total:,} candidates -> top {top_k} in {total_time:.1f}s")
    print(f"  Best: {top[0]['final_score']:.4f} | Lowest: {top[-1]['final_score']:.4f} | Honeypots: {honeypots}")
    print()


if __name__ == "__main__":
    main()
