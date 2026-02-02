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
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"


def flatten_region(d: Dict[str, Any], projection_type: Optional[str] = None) -> Dict[str, Any]:
    """Flatten nested region JSON to flat dict for CSV.

    Args:
        d: Region data dict
        projection_type: None, "back", "forward", or "finalized"
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

    # Add representative point (nearest to mean) if available
    representative = region.get("representative_point") or []
    for i, val in enumerate(representative):
        row[f"representative_{i}"] = val

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

    # Finalization stats
    if projection_type == "finalized":
        fp = d.get("forwardprojection", {})
        row["fp_constrained_by"] = fp.get("constrained_by")
        row["fp_iou"] = fp.get("iou")

        fin = d.get("finalization", {})
        row["fin_status"] = fin.get("status")
        row["hull_acceptance"] = fin.get("hull_acceptance")
        row["simplex_accuracy"] = fin.get("simplex_accuracy")

        cloud_stats = fin.get("cloud_statistics", {})
        row["cloud_n_points"] = cloud_stats.get("n_points")


        extreme_stats = fin.get("extreme_statistics", {})
        row["cloud_diameter"] = extreme_stats.get("diameter")
        row["max_centroid_distance"] = extreme_stats.get("max_centroid_distance")

        # Add cloud dim_bounds (verified points) with extrema info
        cloud_dim_bounds = cloud_stats.get("dim_bounds") or []
        for i, bounds in enumerate(cloud_dim_bounds):
            if bounds:
                row[f"cloud_dim_{i}_min"] = bounds.get("min")
                row[f"cloud_dim_{i}_max"] = bounds.get("max")
                row[f"cloud_dim_{i}_delta"] = bounds.get("delta")

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
    elif projection_type == "finalized":
        main_fields.extend([
            "fp_constrained_by", "fp_iou",
            "fin_status", "hull_acceptance", "simplex_accuracy",
            "cloud_n_points", "cloud_diameter", "max_centroid_distance",
        ])

    # Sort numerically, not lexicographically (so centroid_2 comes before centroid_10)
    def extract_num(s):
        parts = s.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    centroid_keys = sorted([k for k in all_keys if k.startswith("centroid_")], key=extract_num)
    representative_keys = sorted([k for k in all_keys if k.startswith("representative_")], key=extract_num)

    # Sort dim keys: group by dimension number, then order min/max/delta within each
    def dim_sort_key(k):
        parts = k.split("_")
        dim_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        # Order: min=0, max=1, delta=2
        suffix_order = {"min": 0, "max": 1, "delta": 2}.get(parts[-1], 3)
        return (dim_num, suffix_order)

    dim_keys = sorted([k for k in all_keys if k.startswith("dim_") and not k.startswith("dim_bounds")], key=dim_sort_key)
    cloud_dim_keys = sorted([k for k in all_keys if k.startswith("cloud_dim_")], key=dim_sort_key)

    fieldnames = main_fields + centroid_keys + representative_keys + dim_keys + cloud_dim_keys

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
        "finalized": "regions-finalized.json",
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

    # Finalization stats
    if projection_type == "finalized":
        finalized = [r for r in regions if r.get("finalization", {}).get("status") == "success"]
        if finalized:
            hull_acceptances = [r["finalization"]["hull_acceptance"] for r in finalized if r["finalization"].get("hull_acceptance") is not None]
            simplex_accuracies = [r["finalization"]["simplex_accuracy"] for r in finalized if r["finalization"].get("simplex_accuracy") is not None]
            cloud_sizes = [r["finalization"]["cloud_statistics"]["n_points"] for r in finalized if r["finalization"].get("cloud_statistics")]

            lines.append("FINALIZATION STATISTICS")
            lines.append("-" * 40)
            lines.append(f"Successfully finalized: {len(finalized)}/{total}")

            if hull_acceptances:
                lines.append(f"\nHull acceptance (% of hull samples that verify):")
                lines.append(f"  Mean: {np.mean(hull_acceptances)*100:.1f}%")
                lines.append(f"  Min:  {np.min(hull_acceptances)*100:.1f}%")
                lines.append(f"  Max:  {np.max(hull_acceptances)*100:.1f}%")

            if simplex_accuracies:
                lines.append(f"\nSimplex accuracy (% of full simplex that's valid):")
                lines.append(f"  Mean: {np.mean(simplex_accuracies)*100:.4f}%")
                lines.append(f"  Min:  {np.min(simplex_accuracies)*100:.4f}%")
                lines.append(f"  Max:  {np.max(simplex_accuracies)*100:.4f}%")

            if cloud_sizes:
                lines.append(f"\nVerified point cloud sizes:")
                lines.append(f"  Total:  {sum(cloud_sizes):,}")
                lines.append(f"  Mean:   {np.mean(cloud_sizes):.0f}")
                lines.append(f"  Min:    {min(cloud_sizes)}")
                lines.append(f"  Max:    {max(cloud_sizes)}")
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

        if projection_type == "finalized":
            fin = r.get("finalization", {})
            if fin.get("status") == "success":
                hull_acc = fin.get("hull_acceptance", 0) or 0
                simp_acc = fin.get("simplex_accuracy", 0) or 0
                n_pts = fin.get("cloud_statistics", {}).get("n_points", 0) or 0
                line += f"  [hull={hull_acc:.0%}, simplex={simp_acc:.4%}, pts={n_pts}]"

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


def load_viewership(data_dir: Path) -> Dict[tuple, float]:
    """
    Load viewership data and create mapping from (season, week) to viewers in millions.

    Returns dict mapping (season, week) -> viewership_millions
    """
    viewership_path = data_dir / "viewership.csv"
    if not viewership_path.exists():
        return {}

    import re

    viewership = {}

    with open(viewership_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            season = int(row["Season"])
            label = row["EpisodeLabel"]
            viewers = float(row["ViewersMillions"])

            # Try to extract week number from label
            week = None

            # Match patterns like "Week 1", "Week 2:", "Week 10"
            week_match = re.search(r'Week\s*(\d+)', label, re.IGNORECASE)
            if week_match:
                week = int(week_match.group(1))

            # Match patterns like "Episode 101" -> week 1, "Episode 302" -> week 2
            if week is None:
                ep_match = re.search(r'Episode\s*(\d)0(\d)', label)
                if ep_match:
                    # Episode XYZ where X is season digit, YZ is episode/week
                    week = int(ep_match.group(2))

            # Match "Performance show X" or "Results show X"
            if week is None:
                perf_match = re.search(r'(?:Performance|Results)\s+show\s+(\d+)', label, re.IGNORECASE)
                if perf_match:
                    week = int(perf_match.group(1))

            # Match "Top X Perform" patterns (season 14)
            if week is None:
                top_match = re.search(r'Top\s+(\d+)\s+Perform', label, re.IGNORECASE)
                if top_match:
                    # Map top N to approximate week
                    top_n = int(top_match.group(1))
                    # Rough mapping - this varies by season
                    week = 14 - top_n  # approximate

            # Match simple row numbers for season 11
            if week is None and season == 11:
                row_match = re.search(r'Row\s+(\d+)', label)
                if row_match:
                    week = int(row_match.group(1))

            # Skip results shows (they don't have eliminations, just announcements)
            if week is not None and 'Result' in label and 'Final' not in label:
                continue

            # Skip specials
            if 'Special' in label or 'Road to' in label or 'Anniversary' in label:
                continue

            if week is not None:
                key = (season, week)
                # Keep maximum viewership for the week (performance show, not results)
                if key not in viewership or viewers > viewership[key]:
                    viewership[key] = viewers

    return viewership


def write_votes_csv(regions: List[Dict[str, Any]], output_path: Path, data_dir: Path) -> None:
    """
    Write votes.csv with per-contestant vote proportions from finalized regions.

    Each row is one contestant in one event, with columns:
    - season, week, name
    - proportion, proportion_min, proportion_max, proportion_range
    - absolute, absolute_min, absolute_max, absolute_range (scaled by viewership * votes_per_viewer)
    - proportion_volume, proportion_relative_volume, proportion_relative_volume_root (hull geometry)
    """
    from config import get_config
    cfg = get_config()
    votes_per_viewer = cfg.voting.votes_per_viewer

    # Load viewership data
    viewership = load_viewership(data_dir)

    rows = []

    for region in regions:
        event = region.get("event", {})
        region_data = region.get("region", {})
        finalization = region.get("finalization", {})

        season = event.get("season")
        week = event.get("week")
        is_final = event.get("is_final", False)
        contestants = event.get("contestants", [])

        # Get representative point (nearest to mean in verified cloud)
        representative = finalization.get("representative_point", [])
        if not representative:
            representative = region_data.get("representative_point", [])

        # Get dim_bounds for min/max
        dim_bounds = region_data.get("dim_bounds", [])

        # Get viewership for this season/week (in millions)
        viewers_millions = viewership.get((season, week), None)

        # Total votes = viewers * 1_000_000 * votes_per_viewer
        total_votes = None
        if viewers_millions is not None:
            total_votes = viewers_millions * 1_000_000 * votes_per_viewer

        # Get hull volume data from finalization or region
        hull_data = finalization.get("hull", {})
        proportion_volume = hull_data.get("volume") or region_data.get("volume")
        proportion_relative_volume = hull_data.get("relative_volume") or region_data.get("relative_volume")
        proportion_relative_volume_root = hull_data.get("relative_volume_root")

        # If relative_volume_root not in hull data, compute it
        if proportion_relative_volume_root is None and proportion_relative_volume is not None:
            hull_dim = hull_data.get("dimension") or (event.get("n_contestants", 2) - 1)
            if proportion_relative_volume > 0 and hull_dim > 0:
                proportion_relative_volume_root = proportion_relative_volume ** (1.0 / hull_dim)
            else:
                proportion_relative_volume_root = 0.0


        # Skip if no data
        if not representative or not contestants:
            continue

        # Create row for each contestant
        for i, name in enumerate(contestants):
            prop = representative[i] if i < len(representative) else None

            if i < len(dim_bounds):
                bounds = dim_bounds[i]
                prop_min = bounds.get("min")
                prop_max = bounds.get("max")
                prop_range = bounds.get("delta")
            else:
                prop_min = prop_max = prop_range = None

            # Calculate absolute votes
            abs_votes = None
            abs_min = None
            abs_max = None
            abs_range = None

            if total_votes is not None:
                if prop is not None:
                    abs_votes = prop * total_votes
                if prop_min is not None:
                    abs_min = prop_min * total_votes
                if prop_max is not None:
                    abs_max = prop_max * total_votes
                if prop_range is not None:
                    abs_range = prop_range * total_votes

            row = {
                "season": season,
                "week": week,
                "is_final": is_final,
                "name": name,
                "viewers_millions": round(viewers_millions, 2) if viewers_millions else None,
                "proportion": round(prop, 6) if prop is not None else None,
                "proportion_min": round(prop_min, 6) if prop_min is not None else None,
                "proportion_max": round(prop_max, 6) if prop_max is not None else None,
                "proportion_range": round(prop_range, 6) if prop_range is not None else None,
                "proportion_volume": float(proportion_volume) if proportion_volume is not None else None,
                "proportion_relative_volume": float(proportion_relative_volume) if proportion_relative_volume is not None else None,
                "proportion_relative_volume_root": round(float(proportion_relative_volume_root), 6) if proportion_relative_volume_root is not None else None,
                "absolute": round(abs_votes) if abs_votes is not None else None,
                "absolute_min": round(abs_min) if abs_min is not None else None,
                "absolute_max": round(abs_max) if abs_max is not None else None,
                "absolute_range": round(abs_range) if abs_range is not None else None,
            }
            rows.append(row)

    if not rows:
        print("  No data to write for votes.csv")
        return

    # Write CSV
    fieldnames = [
        "season", "week", "is_final", "name", "viewers_millions",
        "proportion", "proportion_min", "proportion_max", "proportion_range",
        "proportion_volume", "proportion_relative_volume", "proportion_relative_volume_root",
        "absolute", "absolute_min", "absolute_max", "absolute_range",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {output_path.name} ({len(rows)} rows)")


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
    parser.add_argument("--finalized-only", action="store_true",
                        help="Only process regions-finalized.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Determine which files to process
    # By default, process all including finalized
    only_flags = [args.regions_only, args.backprojected_only, args.forwardprojected_only, args.finalized_only]
    any_only = any(only_flags)

    process_regions = args.regions_only or (not any_only)
    process_back = args.backprojected_only or (not any_only)
    process_forward = args.forwardprojected_only or (not any_only)
    process_finalized = args.finalized_only or (not any_only)  # Now included by default

    if process_regions:
        process_file(data_dir / "regions.json", projection_type=None)

    if process_back:
        process_file(data_dir / "regions-backprojected.json", projection_type="back")

    if process_forward:
        process_file(data_dir / "regions-forwardprojected.json", projection_type="forward")

    if process_finalized:
        process_file(data_dir / "regions-finalized.json", projection_type="finalized")

        # Also write votes.csv from finalized regions
        finalized_path = data_dir / "regions-finalized.json"
        if finalized_path.exists():
            with open(finalized_path) as f:
                finalized_regions = json.load(f)
            write_votes_csv(finalized_regions, data_dir / "votes.csv", data_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
