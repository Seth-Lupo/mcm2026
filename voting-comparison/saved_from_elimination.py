#!/usr/bin/env python3
"""
Calculate who was saved from elimination the most based on voting system differences.

For seasons 3-27 (PP system): people in rr_elim but NOT in pp_elim were saved
For seasons 1-2, 28-34 (RR system): people in pp_elim but NOT in rr_elim were saved
"""
import json
from collections import Counter
from pathlib import Path

def main():
    data_path = Path(__file__).parent / "comparison_results.json"

    with open(data_path) as f:
        data = json.load(f)

    disagree_events = data["elimination_comparison"]["disagree_events"]

    saved_counts = Counter()
    saved_details = []

    for event in disagree_events:
        season = event["season"]
        week = event["week"]
        rr_elim = set(event["rr_elim"])
        pp_elim = set(event["pp_elim"])

        # Determine which system was actually used
        if 3 <= season <= 27:
            # PP was used - people in rr_elim but not pp_elim were saved
            saved = rr_elim - pp_elim
            system = "PP"
        else:
            # RR was used - people in pp_elim but not rr_elim were saved
            saved = pp_elim - rr_elim
            system = "RR"

        for person in saved:
            saved_counts[person] += 1
            saved_details.append({
                "person": person,
                "season": season,
                "week": week,
                "system": system
            })

    # Sort by count descending
    sorted_saved = saved_counts.most_common()

    # Write results
    output_path = Path(__file__).parent / "saved_from_elimination.txt"

    lines = []
    lines.append("SAVED FROM ELIMINATION ANALYSIS")
    lines.append("=" * 50)
    lines.append("")
    lines.append("People saved from elimination (by count):")
    lines.append("-" * 50)

    for person, count in sorted_saved:
        lines.append(f"  {person}: {count} time(s)")

    lines.append("")
    lines.append(f"Total unique people saved: {len(sorted_saved)}")
    lines.append(f"Total save events: {sum(saved_counts.values())}")
    lines.append("")
    lines.append("Details:")
    lines.append("-" * 50)

    for detail in saved_details:
        lines.append(f"  S{detail['season']} W{detail['week']}: {detail['person']} (saved by {detail['system']} system)")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {output_path}")
    print()
    print("Top saved from elimination:")
    for person, count in sorted_saved[:10]:
        print(f"  {person}: {count}")

if __name__ == "__main__":
    main()
