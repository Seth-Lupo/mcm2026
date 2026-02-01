#!/usr/bin/env python3
"""
DWTS Vote Prediction - Simple Models

Entrypoint for evaluating vote prediction models against actual outcomes.

Usage:
    python main.py [--model MODEL] [--seasons S1,S2,...] [--verbose]
    python main.py --compare
    python main.py --build-events  # Rebuild week_events.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

from events import WeekEvent, load_events, build_week_events, save_events, load_main_data, DATA_DIR
from models import get_model, MODELS, BaseModel
from validation import evaluate_model, EvaluationResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def ensure_events_exist() -> Path:
    """Ensure week_events.json exists, build if needed."""
    path = DATA_DIR / "week_events.json"
    if not path.exists():
        log.info("Building week events (first run)...")
        df = load_main_data()
        events = build_week_events(df)
        save_events(events, path)
    return path


def load_features() -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """Load external features. Placeholder - implement to add real features."""
    return {}


def run_evaluation(
    model_name: str = "uniform",
    seasons: Optional[List[int]] = None,
    model_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> EvaluationResult:
    """Run evaluation for a model."""
    events_path = ensure_events_exist()

    log.info(f"Loading events from {events_path}")
    events = load_events(events_path)

    if seasons:
        events = [e for e in events if e.season in seasons]
    log.info(f"Evaluating on {len(events)} events")

    log.info(f"Building model: {model_name}")
    model = get_model(model_name, **(model_kwargs or {}))

    features = load_features()
    result = evaluate_model(model, events, features)

    # Print results
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)
    print(f"Accuracy: {result.accuracy:.1%} ({result.correct_weeks}/{result.total_weeks} weeks)")

    print("\nBy premise type:")
    for ptype in sorted(result.by_premise.keys()):
        correct, total = result.by_premise[ptype]
        acc = correct / total if total > 0 else 0
        print(f"  {ptype:<12}: {acc:>6.1%} ({correct}/{total})")

    if verbose:
        print("\nBy season:")
        for s in sorted(result.by_season.keys()):
            correct, total = result.by_season[s]
            acc = correct / total if total > 0 else 0
            print(f"  S{s:<2}: {acc:>6.1%} ({correct}/{total})")

    return result


def compare_models(seasons: Optional[List[int]] = None, wiki_only: bool = False, verbose: bool = False) -> None:
    """Compare all models."""
    events_path = ensure_events_exist()
    events = load_events(events_path)

    if seasons:
        events = [e for e in events if e.season in seasons]

    # If wiki_only, filter to events supported by wiki model for fair comparison
    if wiki_only:
        wiki_model = get_model("wiki_popularity")
        events = [e for e in events if wiki_model.supports_event(e.season, e.week, e.contestants)]
        print(f"\n[Filtered to {len(events)} events with wiki data for fair comparison]")

    features = load_features()

    # All premise types in order
    all_premises = ["PP_ELIM", "PP_FINAL", "PP_MULTI", "RR_B2", "RR_B2_MULTI", "RR_ELIM", "RR_FINAL"]

    if verbose:
        # Wide format with all premise columns
        header = f"{'Model':<22} {'Acc':>7} {'Events':>10}"
        for pt in all_premises:
            header += f" {pt:>12}"
        print("\n" + "=" * len(header))
        print("MODEL COMPARISON")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for name in MODELS:
            model = get_model(name)
            result = evaluate_model(model, events, features)

            row = f"{name:<22} {result.accuracy:>6.1%} {result.correct_weeks:>4}/{result.total_weeks:<4}"
            for pt in all_premises:
                c, t = result.by_premise.get(pt, (0, 0))
                if t > 0:
                    row += f" {c:>4}/{t:<4}"
                else:
                    row += f" {'--':>9}"
            print(row)
    else:
        # Compact format
        print("\n" + "=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)
        print(f"{'Model':<25} {'Accuracy':>10} {'Events':>12}")
        print("-" * 50)

        for name in MODELS:
            model = get_model(name)
            result = evaluate_model(model, events, features)
            print(f"{name:<25} {result.accuracy:>10.1%} {result.correct_weeks:>5}/{result.total_weeks:<5}")


def rebuild_events() -> None:
    """Rebuild week_events.json from main.csv."""
    log.info("Rebuilding week events...")
    df = load_main_data()
    events = build_week_events(df)
    path = DATA_DIR / "week_events.json"
    save_events(events, path)
    log.info(f"Saved {len(events)} events to {path}")


def describe_model(name: str) -> None:
    """Print description of a model."""
    if name == "all":
        print("\nAvailable models:\n")
        for model_name, model_cls in MODELS.items():
            model = model_cls() if model_name not in ["wiki_popularity", "wiki_season", "wiki_judge_combined"] else get_model(model_name)
            print(f"  {model_name:<22} {model.description}")
        print()
    elif name in MODELS:
        model = get_model(name)
        print(f"\n{name}:")
        print(f"  {model.description}\n")
    else:
        print(f"Unknown model: {name}. Use --ask all to list all models.")


def main():
    parser = argparse.ArgumentParser(description="DWTS Vote Prediction Models")
    parser.add_argument("--model", "-m", default="uniform", choices=list(MODELS.keys()))
    parser.add_argument("--seasons", "-s", default=None, help="Comma-separated seasons")
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    parser.add_argument("--wiki-only", action="store_true", help="Only evaluate on events with wiki data")
    parser.add_argument("--build-events", action="store_true", help="Rebuild events JSON")
    parser.add_argument("--ask", default=None, help="Describe a model (or 'all' for all models)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    seasons = None
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]

    if args.ask:
        describe_model(args.ask)
    elif args.build_events:
        rebuild_events()
    elif args.compare:
        compare_models(seasons, wiki_only=args.wiki_only, verbose=args.verbose)
    else:
        run_evaluation(model_name=args.model, seasons=seasons, verbose=args.verbose)


if __name__ == "__main__":
    main()
