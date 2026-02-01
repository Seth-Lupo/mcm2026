#!/usr/bin/env python3
"""
Export region analysis results to CSV and TXT summaries.

Converts:
  - regions.json -> regions.csv, regions.txt
  - regions-backprojected.json -> regions-backprojected.csv, regions-backprojected.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"


def flatten_region(d: Dict[str, Any], projection_type: Optional[str] = None) -> Dict[str, Any]:
    """Flatten nested region JSON to flat dict for CSV.

    Args:
        d: Region data dict
        projection_type: None, "back", or "forward"
    """
    event = d.get("event", {})
    region = d.get("region", {})

    row = {
        "season": event.get("season"),
        "week": event.get("week"),
        "is_final": event.get("is_final", False),
        "premise_type": event.get("premise_type"),
        "n_contestants": event.get("n_contestants"),
        "contestants": "|".join(event.get("contestants", [])),
        "has_hull": region.get("has_hull", False),
        "n_vertices": region.get("n_vertices", 0),
        "volume": region.get("volume", 0),
        "relative_volume": region.get("relative_volume", 0),
        "diameter": region.get("diameter", 0),
    }

    # Add centroid as separate columns
    centroid = region.get("centroid") or []
    for i, val in enumerate(centroid):
        row[f"centroid_{i}"] = val

    # Add dim_bounds as min/max/delta columns
    dim_bounds = region.get("dim_bounds") or []
    for i, bounds in enumerate(dim_bounds):
        if bounds:
            # Handle both old format [min, max] and new format {"min":, "max":, "delta":}
            if isinstance(bounds, dict):
                row[f"dim_{i}_min"] = bounds.get("min")
                row[f"dim_{i}_max"] = bounds.get("max")
                row[f"dim_{i}_delta"] = bounds.get("delta")
            else:
                row[f"dim_{i}_min"] = bounds[0]
                row[f"dim_{i}_max"] = bounds[1]
                row[f"dim_{i}_delta"] = bounds[1] - bounds[0] if len(bounds) >= 2 else None

    # Projection stats
    if projection_type == "back":
        bp = d.get("backprojection", {})
        row["constrained_by"] = bp.get("constrained_by")
        row["original_volume"] = bp.get("original_volume")
        row["filtered_volume"] = bp.get("filtered_volume")
        row["volume_lost"] = bp.get("volume_lost")
        row["volume_lost_pct"] = bp.get("volume_lost_pct")
        row["iou"] = bp.get("iou")
    elif projection_type == "forward":
        fp = d.get("forwardprojection", {})
        row["constrained_by"] = fp.get("constrained_by")
        row["original_volume"] = fp.get("original_volume")
        row["filtered_volume"] = fp.get("filtered_volume")
        row["volume_lost"] = fp.get("volume_lost")
        row["volume_lost_pct"] = fp.get("volume_lost_pct")
        row["iou"] = fp.get("iou")

    return row


def write_csv(regions: List[Dict[str, Any]], path: Path, projection_type: Optional[str] = None) -> None:
    """Write regions to CSV.

    Args:
        regions: List of region dicts
        path: Output path
        projection_type: None, "back", or "forward"
    """
    if not regions:
        return

    rows = [flatten_region(r, projection_type) for r in regions]

    # Get all unique keys across all rows
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    # Sort keys: main fields first, then centroid, then dim bounds
    main_fields = [
        "season", "week", "is_final", "premise_type", "n_contestants", "contestants",
        "has_hull", "n_vertices", "volume", "relative_volume", "diameter",
    ]
    if projection_type in ("back", "forward"):
        main_fields.extend([
            "constrained_by", "original_volume", "filtered_volume",
            "volume_lost", "volume_lost_pct", "iou",
        ])

    # Sort numerically, not lexicographically (so centroid_2 comes before centroid_10)
    def extract_num(s):
        parts = s.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    centroid_keys = sorted([k for k in all_keys if k.startswith("centroid_")], key=extract_num)

    # Sort dim keys: group by dimension number, then order min/max/delta within each
    def dim_sort_key(k):
        parts = k.split("_")
        dim_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        # Order: min=0, max=1, delta=2
        suffix_order = {"min": 0, "max": 1, "delta": 2}.get(parts[-1], 3)
        return (dim_num, suffix_order)

    dim_keys = sorted([k for k in all_keys if k.startswith("dim_")], key=dim_sort_key)

    fieldnames = main_fields + centroid_keys + dim_keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    regions: List[Dict[str, Any]],
    path: Path,
    projection_type: Optional[str] = None,
) -> None:
    """Write human-readable summary to TXT.

    Args:
        regions: List of region dicts
        path: Output path
        projection_type: None, "back", or "forward"
    """
    source_name = {
        None: "regions.json",
        "back": "regions-backprojected.json",
        "forward": "regions-forwardprojected.json",
    }.get(projection_type, "regions.json")

    lines = []
    lines.append("=" * 70)
    lines.append(f"REGION ANALYSIS SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source: {source_name}")
    lines.append("=" * 70)
    lines.append("")

    # Overall stats
    total = len(regions)
    with_hull = sum(1 for r in regions if r.get("region", {}).get("has_hull", False))
    zero_valid = sum(1 for r in regions if r.get("sampling", {}).get("n_valid", 0) == 0)
    low_valid = sum(1 for r in regions if 0 < r.get("sampling", {}).get("n_valid", 0) < 10)

    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total regions:        {total}")
    lines.append(f"With valid hull:      {with_hull} ({with_hull/total*100:.1f}%)")
    lines.append(f"Zero valid points:    {zero_valid} ({zero_valid/total*100:.1f}%)")
    lines.append(f"Low valid (<10):      {low_valid} ({low_valid/total*100:.1f}%)")
    lines.append("")

    if projection_type == "back":
        proj_key = "backprojection"
        proj_name = "BACKPROJECTION"
    elif projection_type == "forward":
        proj_key = "forwardprojection"
        proj_name = "FORWARD PROJECTION"
    else:
        proj_key = None
        proj_name = None

    if proj_key:
        constrained = sum(1 for r in regions if r.get(proj_key, {}).get("constrained_by") is not None)
        total_orig_vol = sum(r.get(proj_key, {}).get("original_volume", 0) or 0 for r in regions)
        total_filt_vol = sum(r.get(proj_key, {}).get("filtered_volume", 0) or 0 for r in regions)

        lines.append(f"{proj_name} STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Regions constrained:  {constrained}")
        lines.append(f"Original volume:      {total_orig_vol:.6f}")
        lines.append(f"Filtered volume:      {total_filt_vol:.6f}")
        if total_orig_vol > 0:
            lines.append(f"Overall vol retention: {total_filt_vol/total_orig_vol*100:.1f}%")
        lines.append("")

    # By premise type
    by_premise = {}
    for r in regions:
        ptype = r.get("event", {}).get("premise_type", "UNKNOWN")
        if ptype not in by_premise:
            by_premise[ptype] = {"count": 0, "valid": 0, "rates": []}
        by_premise[ptype]["count"] += 1
        n_valid = r.get("sampling", {}).get("n_valid", 0)
        if n_valid > 0:
            by_premise[ptype]["valid"] += 1
        by_premise[ptype]["rates"].append(r.get("sampling", {}).get("acceptance_rate", 0))

    lines.append("BY PREMISE TYPE")
    lines.append("-" * 40)
    lines.append(f"{'Premise':<15} {'Count':>6} {'Valid':>6} {'Avg Rate':>10}")
    for ptype in sorted(by_premise.keys()):
        stats = by_premise[ptype]
        avg_rate = sum(stats["rates"]) / len(stats["rates"]) if stats["rates"] else 0
        lines.append(f"{ptype:<15} {stats['count']:>6} {stats['valid']:>6} {avg_rate:>10.4%}")
    lines.append("")

    # Successful regions (non-zero valid points)
    lines.append("SUCCESSFUL REGIONS (non-zero valid points)")
    lines.append("-" * 70)

    successful = [r for r in regions if r.get("sampling", {}).get("n_valid", 0) > 0]
    successful.sort(key=lambda r: (r.get("event", {}).get("season", 0), r.get("event", {}).get("week", 0)))

    for r in successful:
        event = r.get("event", {})
        sampling = r.get("sampling", {})
        region = r.get("region", {})

        tag = f"S{event.get('season', '?'):02d}W{event.get('week', '?'):02d}"
        if event.get("is_final"):
            tag += " [FINAL]"

        n_valid = sampling.get("n_valid", 0)
        rate = sampling.get("acceptance_rate", 0)
        n_vert = region.get("n_vertices", 0)
        vol = region.get("relative_volume", 0)

        line = f"{tag:<18} {event.get('premise_type', '?'):<12} n={event.get('n_contestants', '?'):>2}  valid={n_valid:>6}  rate={rate:.4%}  vertices={n_vert:>4}  rel_vol={vol:.6f}"

        if proj_key:
            proj = r.get(proj_key, {})
            if proj.get("constrained_by"):
                orig_vol = proj.get("original_volume", 0) or 0
                filt_vol = proj.get("filtered_volume", 0) or 0
                iou = proj.get("iou", 0) or 0
                line += f"  [by W{proj['constrained_by']}: vol {orig_vol:.4f}->{filt_vol:.4f}, iou={iou:.2%}]"

        lines.append(line)

    lines.append("")

    # Failed regions
    failed = [r for r in regions if r.get("sampling", {}).get("n_valid", 0) == 0]
    if failed:
        lines.append("FAILED REGIONS (zero valid points)")
        lines.append("-" * 70)
        for r in failed:
            event = r.get("event", {})
            tag = f"S{event.get('season', '?'):02d}W{event.get('week', '?'):02d}"
            if event.get("is_final"):
                tag += " [FINAL]"
            lines.append(f"{tag:<18} {event.get('premise_type', '?'):<12} n={event.get('n_contestants', '?')}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def process_file(json_path: Path, projection_type: Optional[str] = None) -> None:
    """Process a single JSON file to CSV and TXT.

    Args:
        json_path: Path to JSON file
        projection_type: None, "back", or "forward"
    """
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    print(f"Processing {json_path.name}...")

    with open(json_path) as f:
        regions = json.load(f)

    print(f"  Loaded {len(regions)} regions")

    # Derive output paths
    stem = json_path.stem  # e.g., "regions" or "regions-backprojected"
    csv_path = json_path.parent / f"{stem}.csv"
    txt_path = json_path.parent / f"{stem}.txt"

    # Write CSV
    write_csv(regions, csv_path, projection_type)
    print(f"  Wrote {csv_path.name}")

    # Write TXT summary
    write_summary(regions, txt_path, projection_type)
    print(f"  Wrote {txt_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Export region analysis to CSV and TXT")
    parser.add_argument("--data-dir", "-d", default=str(DATA_DIR),
                        help="Data directory containing JSON files")
    parser.add_argument("--regions-only", action="store_true",
                        help="Only process regions.json")
    parser.add_argument("--backprojected-only", action="store_true",
                        help="Only process regions-backprojected.json")
    parser.add_argument("--forwardprojected-only", action="store_true",
                        help="Only process regions-forwardprojected.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Determine which files to process
    process_regions = not (args.backprojected_only or args.forwardprojected_only)
    process_back = not (args.regions_only or args.forwardprojected_only)
    process_forward = not (args.regions_only or args.backprojected_only)

    if process_regions:
        process_file(data_dir / "regions.json", projection_type=None)

    if process_back:
        process_file(data_dir / "regions-backprojected.json", projection_type="back")

    if process_forward:
        process_file(data_dir / "regions-forwardprojected.json", projection_type="forward")

    print("\nDone!")


if __name__ == "__main__":
    main()
