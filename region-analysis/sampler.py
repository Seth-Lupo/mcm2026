"""
Random sampling of probability vectors on the simplex.
Finds valid vote distributions that satisfy premise constraints.

Uses centralized premise logic from premises.py.
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from events import Event
from premises import (
    PremiseType,
    get_premise_type,
    ranks_1d,
    warmup_jit,
    check_rr_elim_batch,
    check_pp_elim_batch,
    check_rr_b2_batch,
    check_rr_final_batch,
    check_pp_final_batch,
)

log = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result of sampling valid vote vectors."""
    event: Event
    premise_type: PremiseType
    valid_samples: np.ndarray
    n_samples: int
    n_valid: int

    @property
    def acceptance_rate(self) -> float:
        return self.n_valid / self.n_samples if self.n_samples > 0 else 0.0


def _check_validity(
    samples: np.ndarray,
    event: Event,
    premise_type: PremiseType,
) -> np.ndarray:
    """
    Check validity of samples against premise constraints.

    Delegates to batch functions in premises.py.

    Args:
        samples: (n_samples, n_contestants) vote distributions
        event: Event data
        premise_type: Type of premise to check

    Returns:
        Boolean mask of valid samples
    """
    elim_indices = np.array(
        [event.contestants.index(name) for name in event.eliminated],
        dtype=np.int64
    )
    n_elim = len(elim_indices)
    judge = event.judge_scores.astype(np.float64)

    if event.is_final:
        expected_order = np.argsort(event.placements).astype(np.int64)
        if premise_type == PremiseType.RR_FINAL:
            return check_rr_final_batch(samples, ranks_1d(judge), expected_order)
        elif premise_type == PremiseType.PP_FINAL:
            return check_pp_final_batch(samples, judge / judge.sum(), expected_order)
    else:
        if premise_type in (PremiseType.RR_ELIM, PremiseType.RR_MULTI):
            return check_rr_elim_batch(samples, ranks_1d(judge), elim_indices, n_elim)
        elif premise_type in (PremiseType.PP_ELIM, PremiseType.PP_MULTI):
            return check_pp_elim_batch(samples, judge / judge.sum(), elim_indices, n_elim)
        elif premise_type in (PremiseType.RR_B2, PremiseType.RR_B2_MULTI):
            return check_rr_b2_batch(samples, ranks_1d(judge), elim_indices, max(2, n_elim))

    return np.zeros(len(samples), dtype=bool)


MAX_RETRIES = 20  # max retry attempts if 0 hits


def sample_valid_votes(
    event: Event,
    n_samples: int = 100000,
    seed: Optional[int] = None,
) -> SampleResult:
    """
    Sample random vote vectors and keep those satisfying the premise.

    Samples uniformly from the probability simplex using Dirichlet(1,...,1).
    Retries if 0 hits initially.

    Args:
        event: Event to sample for
        n_samples: Number of random samples to generate
        seed: Random seed for reproducibility

    Returns:
        SampleResult with valid samples and statistics
    """
    warmup_jit()

    rng = np.random.default_rng(seed)
    premise_type = get_premise_type(event.season, event.n_eliminated, event.is_final)
    n = event.n

    log.info(f"  [SAMPLE] Premise: {premise_type.name}, {n} contestants")

    # Keep sampling until we get at least 1 hit (or max retries)
    all_valid = []
    total_sampled = 0
    attempt = 0

    while len(all_valid) == 0 and attempt < MAX_RETRIES:
        attempt += 1
        if attempt > 1:
            log.info(f"  [SAMPLE] Retry {attempt}/{MAX_RETRIES}...")

        samples = rng.dirichlet(np.ones(n), size=n_samples)
        valid_mask = _check_validity(samples, event, premise_type)
        valid_samples = samples[valid_mask]
        total_sampled += n_samples

        if len(valid_samples) > 0:
            all_valid = valid_samples
            break

    if len(all_valid) == 0:
        log.info(f"  [SAMPLE] No valid samples after {total_sampled:,} attempts")
        return SampleResult(
            event=event,
            premise_type=premise_type,
            valid_samples=np.empty((0, n)),
            n_samples=total_sampled,
            n_valid=0,
        )

    acceptance = len(all_valid) / total_sampled
    log.info(f"  [SAMPLE] Found {len(all_valid):,} valid ({acceptance:.4%} acceptance, {total_sampled:,} total samples)")

    return SampleResult(
        event=event,
        premise_type=premise_type,
        valid_samples=all_valid,
        n_samples=total_sampled,
        n_valid=len(all_valid),
    )
