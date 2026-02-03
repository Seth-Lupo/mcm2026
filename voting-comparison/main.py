#!/usr/bin/env python3
"""
Compare RR (Rank-Rank) and PP (Percent-Percent) voting systems.

Runs BOTH systems on ALL events and compares:
1. RR_ELIM vs PP_ELIM - who gets eliminated
2. RR_FINAL vs PP_FINAL - full rankings (Kendall tau distance)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "region-analysis"))

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any, Set

from premises import TIEBREAKER_WEIGHT, ranks_1d


DATA_DIR = Path(__file__).parent.parent / "data"


def predict_rr_elim(judge_scores: np.ndarray, fan_votes: np.ndarray, n_elim: int) -> Set[int]:
    """RR elimination: sum ranks, lower = better, higher = eliminated."""
    judge_ranks = ranks_1d(judge_scores)
    fan_ranks = ranks_1d(fan_votes)
    combined = judge_ranks + fan_ranks
    sort_key = combined - fan_votes * TIEBREAKER_WEIGHT
    order = np.argsort(-sort_key)  # worst first
    return set(order[:n_elim])


def predict_pp_elim(judge_scores: np.ndarray, fan_votes: np.ndarray, n_elim: int) -> Set[int]:
    """PP elimination: sum percentages, higher = better, lower = eliminated."""
    judge_pct = judge_scores / judge_scores.sum()
    combined = judge_pct + fan_votes
    order = np.argsort(combined)  # worst first
    return set(order[:n_elim])


def predict_rr_ranking(judge_scores: np.ndarray, fan_votes: np.ndarray) -> List[int]:
    """RR ranking: 1st place to last."""
    judge_ranks = ranks_1d(judge_scores)
    fan_ranks = ranks_1d(fan_votes)
    combined = judge_ranks + fan_ranks
    sort_key = combined - fan_votes * TIEBREAKER_WEIGHT
    return list(np.argsort(sort_key))  # best first


def predict_pp_ranking(judge_scores: np.ndarray, fan_votes: np.ndarray) -> List[int]:
    """PP ranking: 1st place to last."""
    judge_pct = judge_scores / judge_scores.sum()
    combined = judge_pct + fan_votes
    return list(np.argsort(-combined))  # best first


def kendall_tau_distance(ranking1: List[int], ranking2: List[int]) -> int:
    """Count pairwise disagreements between two rankings."""
    n = len(ranking1)
    pos1 = {idx: pos for pos, idx in enumerate(ranking1)}
    pos2 = {idx: pos for pos, idx in enumerate(ranking2)}

    distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pos1[i] < pos1[j]) != (pos2[i] < pos2[j]):
                distance += 1
    return distance


def load_events() -> List[Dict[str, Any]]:
    with open(DATA_DIR / "events.json") as f:
        return json.load(f)


def load_vote_proportions() -> Dict[tuple, Dict[str, float]]:
    """
    Load fan vote proportions from votes.csv.
    Returns dict mapping (season, week, is_final) -> {name: proportion}
    """
    import csv
    votes_path = DATA_DIR / "votes.csv"
    if not votes_path.exists():
        return {}

    proportions = defaultdict(dict)
    with open(votes_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            season = int(row["season"])
            week = int(row["week"])
            is_final = row["is_final"].lower() == "true"
            name = row["name"]
            prop = row.get("proportion")
            if prop and prop != "":
                proportions[(season, week, is_final)][name] = float(prop)

    return dict(proportions)


def main():
    print("=" * 70)
    print("RR vs PP VOTING SYSTEM COMPARISON - ALL EVENTS")
    print("=" * 70)
    print()

    events = load_events()
    print(f"Total events: {len(events)}")

    # Load actual vote proportions from votes.csv
    vote_proportions = load_vote_proportions()
    print(f"Events with vote data: {len(vote_proportions)}")
    print()

    # =========================================================================
    # RR_ELIM vs PP_ELIM on ALL events
    # =========================================================================
    print("=" * 70)
    print("ELIMINATION COMPARISON: RR_ELIM vs PP_ELIM")
    print("=" * 70)
    print("Using fan proportions from votes.csv")
    print()

    elim_agree = 0
    elim_disagree = 0
    elim_disagree_list = []
    skipped = 0

    for event in events:
        judge_scores = np.array(event["judge_scores"], dtype=np.float64)
        n = len(judge_scores)
        contestants = event["contestants"]
        eliminated = event.get("eliminated", [])
        n_elim = max(1, len(eliminated))
        is_final = event.get("is_final", False)

        # Get vote proportions for this event
        key = (event["season"], event["week"], is_final)
        props = vote_proportions.get(key, {})

        if not props or len(props) != n:
            skipped += 1
            continue

        # Build fan_votes array in contestant order
        fan_votes = np.array([props.get(name, 1/n) for name in contestants], dtype=np.float64)
        # Normalize to sum to 1
        fan_votes = fan_votes / fan_votes.sum()

        rr_elim = predict_rr_elim(judge_scores, fan_votes, n_elim)
        pp_elim = predict_pp_elim(judge_scores, fan_votes, n_elim)

        if rr_elim == pp_elim:
            elim_agree += 1
        else:
            elim_disagree += 1
            elim_disagree_list.append({
                "season": event["season"],
                "week": event["week"],
                "is_final": event.get("is_final", False),
                "rr_elim": [contestants[i] for i in rr_elim],
                "pp_elim": [contestants[i] for i in pp_elim],
            })

    total_elim = elim_agree + elim_disagree
    print(f"Skipped (no vote data): {skipped}")
    print(f"Analyzed: {total_elim}")
    if total_elim > 0:
        print(f"Agree:    {elim_agree}/{total_elim} ({elim_agree/total_elim*100:.1f}%)")
        print(f"Disagree: {elim_disagree}/{total_elim} ({elim_disagree/total_elim*100:.1f}%)")
    print()

    if elim_disagree_list:
        print("Disagreements (first 15):")
        for ev in elim_disagree_list[:15]:
            tag = "FINAL" if ev["is_final"] else ""
            print(f"  S{ev['season']:02d}W{ev['week']:02d} {tag}: RR={ev['rr_elim']}, PP={ev['pp_elim']}")
        if len(elim_disagree_list) > 15:
            print(f"  ... and {len(elim_disagree_list) - 15} more")
    print()

    # =========================================================================
    # RR_FINAL vs PP_FINAL on ALL events (full ranking comparison)
    # =========================================================================
    print("=" * 70)
    print("RANKING COMPARISON: RR_FINAL vs PP_FINAL")
    print("=" * 70)
    print("Using fan proportions from votes.csv, compare full rankings via Kendall tau distance.")
    print()

    kendall_distances = []
    ranking_comparisons = []
    skipped_ranking = 0

    for event in events:
        judge_scores = np.array(event["judge_scores"], dtype=np.float64)
        n = len(judge_scores)
        contestants = event["contestants"]
        is_final = event.get("is_final", False)

        if n < 2:
            continue

        # Get vote proportions
        key = (event["season"], event["week"], is_final)
        props = vote_proportions.get(key, {})

        if not props or len(props) != n:
            skipped_ranking += 1
            continue

        fan_votes = np.array([props.get(name, 1/n) for name in contestants], dtype=np.float64)
        fan_votes = fan_votes / fan_votes.sum()

        rr_ranking = predict_rr_ranking(judge_scores, fan_votes)
        pp_ranking = predict_pp_ranking(judge_scores, fan_votes)

        distance = kendall_tau_distance(rr_ranking, pp_ranking)
        kendall_distances.append(distance)

        # Max possible distance for n items is n*(n-1)/2
        max_dist = n * (n - 1) // 2

        ranking_comparisons.append({
            "season": event["season"],
            "week": event["week"],
            "is_final": event.get("is_final", False),
            "n": n,
            "kendall_distance": distance,
            "max_distance": max_dist,
            "normalized": distance / max_dist if max_dist > 0 else 0,
            "rr_ranking": [contestants[i] for i in rr_ranking],
            "pp_ranking": [contestants[i] for i in pp_ranking],
        })

    print(f"Skipped (no vote data): {skipped_ranking}")
    print(f"Events analyzed: {len(kendall_distances)}")
    print()

    if kendall_distances:
        print("Kendall Tau Distance Statistics:")
        print(f"  Mean:   {np.mean(kendall_distances):.2f}")
        print(f"  Median: {np.median(kendall_distances):.2f}")
        print(f"  Min:    {min(kendall_distances)}")
        print(f"  Max:    {max(kendall_distances)}")
        print()

        perfect = sum(1 for d in kendall_distances if d == 0)
        print(f"Perfect agreement (distance=0): {perfect}/{len(kendall_distances)} ({perfect/len(kendall_distances)*100:.1f}%)")
        print()

        # Histogram
        print("HISTOGRAM: Kendall Tau Distance Frequency")
        print("-" * 50)
        distance_counts = defaultdict(int)
        for d in kendall_distances:
            distance_counts[d] += 1

        max_d = max(kendall_distances)
        max_count = max(distance_counts.values())
        scale = 40 / max_count if max_count > 0 else 1

        for d in range(max_d + 1):
            count = distance_counts[d]
            bar = "#" * int(count * scale)
            print(f"  {d:3d} | {bar} ({count})")
        print()

        # Save histogram as pyplot
        fig, ax = plt.subplots(figsize=(10, 6))
        distances_sorted = sorted(distance_counts.keys())
        counts = [distance_counts[d] for d in distances_sorted]
        ax.bar(distances_sorted, counts, color='steelblue', edgecolor='black')
        ax.set_xlabel('Kendall Tau Distance', fontsize=12)
        ax.set_ylabel('Frequency (Number of Events)', fontsize=12)
        ax.set_title('RR vs PP Ranking Disagreement: Kendall Tau Distance Histogram', fontsize=14)
        ax.set_xticks(distances_sorted)
        for i, (d, c) in enumerate(zip(distances_sorted, counts)):
            if c > 0:
                ax.annotate(str(c), xy=(d, c), ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        hist_path = Path(__file__).parent / "kendall_tau_histogram.png"
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"Histogram saved to {hist_path}")
        print()

        # Show worst disagreements
        worst = sorted(ranking_comparisons, key=lambda x: -x["kendall_distance"])[:10]
        print("Largest ranking differences:")
        print("-" * 60)
        for comp in worst:
            if comp["kendall_distance"] == 0:
                continue
            tag = "FINAL" if comp["is_final"] else ""
            print(f"  S{comp['season']:02d}W{comp['week']:02d} {tag} (n={comp['n']}, dist={comp['kendall_distance']}/{comp['max_distance']}):")
            print(f"    RR: {' > '.join(comp['rr_ranking'][:5])}{'...' if comp['n'] > 5 else ''}")
            print(f"    PP: {' > '.join(comp['pp_ranking'][:5])}{'...' if comp['n'] > 5 else ''}")
        print()

    # =========================================================================
    # Summary by season
    # =========================================================================
    print("=" * 70)
    print("SUMMARY BY SEASON")
    print("=" * 70)
    print()

    by_season = defaultdict(lambda: {"agree": 0, "disagree": 0, "total_kendall": 0, "count": 0})
    for comp in ranking_comparisons:
        s = comp["season"]
        by_season[s]["count"] += 1
        by_season[s]["total_kendall"] += comp["kendall_distance"]
        if comp["kendall_distance"] == 0:
            by_season[s]["agree"] += 1
        else:
            by_season[s]["disagree"] += 1

    print(f"{'Season':<8} {'Events':<8} {'Agree':<8} {'Disagree':<10} {'Avg Kendall':<12}")
    print("-" * 50)
    for s in sorted(by_season.keys()):
        data = by_season[s]
        avg_k = data["total_kendall"] / data["count"] if data["count"] > 0 else 0
        print(f"{s:<8} {data['count']:<8} {data['agree']:<8} {data['disagree']:<10} {avg_k:<12.2f}")
    print()

    # =========================================================================
    # Save results
    # =========================================================================
    output_path = Path(__file__).parent / "comparison_results.json"
    results = {
        "elimination_comparison": {
            "agree": elim_agree,
            "disagree": elim_disagree,
            "total": total_elim,
            "disagree_events": elim_disagree_list,
        },
        "ranking_comparison": {
            "total": len(ranking_comparisons),
            "kendall_distances": kendall_distances,
            "histogram": dict(distance_counts),
            "comparisons": ranking_comparisons,
        },
        "by_season": {str(k): v for k, v in by_season.items()},
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
