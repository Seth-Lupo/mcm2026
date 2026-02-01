"""Validation logic for checking if predictions match outcomes."""
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from events import WeekEvent, load_events, DATA_DIR
from models import BaseModel, Prediction
from premises import validate, PremiseType


@dataclass
class ValidationResult:
    """Result of validating a single week."""
    season: int
    week: int
    is_final: bool
    correct: bool
    premise_type: str
    model: str


@dataclass
class EvaluationResult:
    """Aggregate evaluation result."""
    total_weeks: int
    correct_weeks: int
    accuracy: float
    by_premise: Dict[str, Tuple[int, int]]  # premise -> (correct, total)
    by_season: Dict[int, Tuple[int, int]]   # season -> (correct, total)
    details: List[ValidationResult]


def validate_event(
    event: WeekEvent,
    prediction: Prediction,
    model_name: str,
) -> ValidationResult:
    """Validate if a prediction matches the actual outcome for an event."""
    correct = validate(
        premise_type=event.premise_type,
        contestants=event.contestants,
        judge_scores=event.judge_scores,
        fan_votes=prediction.votes,
        actual_eliminated=event.eliminated if not event.is_final else None,
        actual_placements=event.placements if event.is_final else None,
    )

    return ValidationResult(
        season=event.season,
        week=event.week,
        is_final=event.is_final,
        correct=correct,
        premise_type=event.premise_type.name,
        model=model_name,
    )


def evaluate_model(
    model: BaseModel,
    events: List[WeekEvent],
    features_by_event: Dict[Tuple[int, int], Dict[str, np.ndarray]] = None,
) -> EvaluationResult:
    """
    Evaluate a model across all events it supports.

    Args:
        model: The model to evaluate.
        events: List of WeekEvent objects.
        features_by_event: Dict mapping (season, week) to feature dicts.

    Returns:
        EvaluationResult with accuracy metrics (only over supported events).
    """
    features_by_event = features_by_event or {}
    results = []

    for event in events:
        # Check if model supports this event
        if not model.supports_event(event.season, event.week, event.contestants):
            continue

        key = (event.season, event.week)
        features = features_by_event.get(key, {})

        # Create a minimal state object for the model
        class State:
            pass
        state = State()
        state.season = event.season
        state.week = event.week
        state.n = event.n
        state.judge_scores = event.judge_scores
        state.contestants = event.contestants

        def judge_share():
            total = event.judge_scores.sum()
            return event.judge_scores / total if total > 0 else np.ones(event.n) / event.n
        state.judge_share = judge_share

        prediction = model.predict(state, features)
        result = validate_event(event, prediction, model.name)
        results.append(result)

    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total > 0 else 0.0

    # By premise
    by_premise = {}
    for r in results:
        if r.premise_type not in by_premise:
            by_premise[r.premise_type] = [0, 0]
        by_premise[r.premise_type][1] += 1
        if r.correct:
            by_premise[r.premise_type][0] += 1
    by_premise = {k: tuple(v) for k, v in by_premise.items()}

    # By season
    by_season = {}
    for r in results:
        if r.season not in by_season:
            by_season[r.season] = [0, 0]
        by_season[r.season][1] += 1
        if r.correct:
            by_season[r.season][0] += 1
    by_season = {k: tuple(v) for k, v in by_season.items()}

    return EvaluationResult(
        total_weeks=total,
        correct_weeks=correct,
        accuracy=accuracy,
        by_premise=by_premise,
        by_season=by_season,
        details=results,
    )
