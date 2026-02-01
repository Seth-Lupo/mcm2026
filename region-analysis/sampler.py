"""
Random sampling of probability vectors on the simplex.
Finds valid vote distributions that satisfy premise constraints.
Vectorized + numba JIT for performance.
"""
import numpy as np
from numba import njit, prange
from typing import Optional
from dataclasses import dataclass
import logging

from events import Event
from premises import PremiseType, get_premise_type
from config import get_config

log = logging.getLogger(__name__)

# Track if JIT compilation has happened
_jit_compiled = False


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


@njit(cache=True)
def _argsort_desc(arr):
    """Argsort descending for 1D array."""
    return np.argsort(-arr)


@njit(cache=True)
def _ranks_1d(scores):
    """Convert 1D scores to ranks (1=best, highest score)."""
    n = len(scores)
    order = np.argsort(-scores)
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        ranks[order[i]] = i + 1
    return ranks


@njit(parallel=True, cache=True)
def _check_rr_elim(fan_votes, judge_ranks, elim_indices, n_elim):
    """Check RR elimination for batch. Returns boolean array."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        # Compute fan ranks for this sample
        fan_ranks = _ranks_1d(fan_votes[b])
        combined = judge_ranks + fan_ranks

        # Find worst n_elim indices (highest combined)
        order = np.argsort(-combined)
        worst = order[:n_elim]

        # Check if worst matches elim_indices (as sets)
        match = True
        for idx in elim_indices:
            found = False
            for w in worst:
                if w == idx:
                    found = True
                    break
            if not found:
                match = False
                break

        if match:
            # Also check reverse: all worst are in elim_indices
            for w in worst:
                found = False
                for idx in elim_indices:
                    if w == idx:
                        found = True
                        break
                if not found:
                    match = False
                    break

        valid[b] = match

    return valid


@njit(parallel=True, cache=True)
def _check_pp_elim(fan_votes, judge_pct, elim_indices, n_elim):
    """Check PP elimination for batch. Returns boolean array."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        # Fan votes already sum to 1 (they're on simplex)
        fan_pct = fan_votes[b]
        combined = judge_pct + fan_pct

        # Find worst n_elim indices (lowest combined)
        order = np.argsort(combined)
        worst = order[:n_elim]

        # Check if worst matches elim_indices
        match = True
        for idx in elim_indices:
            found = False
            for w in worst:
                if w == idx:
                    found = True
                    break
            if not found:
                match = False
                break

        if match:
            for w in worst:
                found = False
                for idx in elim_indices:
                    if w == idx:
                        found = True
                        break
                if not found:
                    match = False
                    break

        valid[b] = match

    return valid


@njit(parallel=True, cache=True)
def _check_rr_b2(fan_votes, judge_ranks, elim_indices, n_bottom):
    """Check RR bottom-2 rule. Eliminated must be IN bottom N."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        fan_ranks = _ranks_1d(fan_votes[b])
        combined = judge_ranks + fan_ranks
        order = np.argsort(-combined)
        bottom = order[:n_bottom]

        # All elim_indices must be in bottom
        match = True
        for idx in elim_indices:
            found = False
            for w in bottom:
                if w == idx:
                    found = True
                    break
            if not found:
                match = False
                break

        valid[b] = match

    return valid


@njit(parallel=True, cache=True)
def _check_rr_final(fan_votes, judge_ranks, expected_order):
    """Check RR final ranking."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        fan_ranks = _ranks_1d(fan_votes[b])
        combined = judge_ranks + fan_ranks
        pred_order = np.argsort(combined)

        match = True
        for i in range(n):
            if pred_order[i] != expected_order[i]:
                match = False
                break
        valid[b] = match

    return valid


@njit(parallel=True, cache=True)
def _check_pp_final(fan_votes, judge_pct, expected_order):
    """Check PP final ranking."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        fan_pct = fan_votes[b]
        combined = judge_pct + fan_pct
        pred_order = np.argsort(-combined)

        match = True
        for i in range(n):
            if pred_order[i] != expected_order[i]:
                match = False
                break
        valid[b] = match

    return valid


def warmup_jit():
    """Trigger JIT compilation with dummy data."""
    global _jit_compiled
    if _jit_compiled:
        return

    log.info("  [JIT] Starting compilation (one-time)...")

    # Small dummy arrays to trigger compilation
    dummy_votes = np.random.dirichlet(np.ones(5), size=10)
    dummy_judge = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dummy_elim = np.array([0], dtype=np.int64)
    dummy_order = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    log.info("  [JIT] Compiling _ranks_1d...")
    _ranks_1d(dummy_judge)

    log.info("  [JIT] Compiling _check_rr_elim...")
    _check_rr_elim(dummy_votes, _ranks_1d(dummy_judge), dummy_elim, 1)

    log.info("  [JIT] Compiling _check_pp_elim...")
    _check_pp_elim(dummy_votes, dummy_judge / dummy_judge.sum(), dummy_elim, 1)

    log.info("  [JIT] Compiling _check_rr_b2...")
    _check_rr_b2(dummy_votes, _ranks_1d(dummy_judge), dummy_elim, 2)

    log.info("  [JIT] Compiling _check_rr_final...")
    _check_rr_final(dummy_votes, _ranks_1d(dummy_judge), dummy_order)

    log.info("  [JIT] Compiling _check_pp_final...")
    _check_pp_final(dummy_votes, dummy_judge / dummy_judge.sum(), dummy_order)

    _jit_compiled = True
    log.info("  [JIT] All functions compiled!")


def _check_validity(
    samples: np.ndarray,
    event: Event,
    premise_type: PremiseType,
) -> np.ndarray:
    """Check validity of samples. Returns boolean mask."""
    elim_indices = np.array([event.contestants.index(name) for name in event.eliminated], dtype=np.int64)
    n_elim = len(elim_indices)
    judge = event.judge_scores.astype(np.float64)

    if event.is_final:
        expected_order = np.argsort(event.placements).astype(np.int64)
        if premise_type == PremiseType.RR_FINAL:
            return _check_rr_final(samples, _ranks_1d(judge), expected_order)
        elif premise_type == PremiseType.PP_FINAL:
            return _check_pp_final(samples, judge / judge.sum(), expected_order)
    else:
        if premise_type in (PremiseType.RR_ELIM, PremiseType.RR_MULTI):
            return _check_rr_elim(samples, _ranks_1d(judge), elim_indices, n_elim)
        elif premise_type in (PremiseType.PP_ELIM, PremiseType.PP_MULTI):
            return _check_pp_elim(samples, judge / judge.sum(), elim_indices, n_elim)
        elif premise_type in (PremiseType.RR_B2, PremiseType.RR_B2_MULTI):
            return _check_rr_b2(samples, _ranks_1d(judge), elim_indices, max(2, n_elim))

    return np.zeros(len(samples), dtype=bool)


def _project_to_simplex(points: np.ndarray) -> np.ndarray:
    """Project points onto simplex (ensure sum=1, all >= 0)."""
    points = np.maximum(points, 0)
    sums = points.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1)
    return points / sums


def refine_with_walks(
    seeds: np.ndarray,
    event: Event,
    premise_type: PremiseType,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Local random walk refinement from seed points.
    Explores the valid region by taking small steps from known valid points.
    """
    cfg = get_config().refinement
    n = event.n

    if len(seeds) == 0:
        return np.empty((0, n))

    # Limit seeds
    if len(seeds) > cfg.max_seeds:
        indices = rng.choice(len(seeds), cfg.max_seeds, replace=False)
        seeds = seeds[indices]

    log.info(f"  [REFINE] Walking from {len(seeds)} seeds, {cfg.walks_per_seed} walks x {cfg.steps_per_walk} steps...")

    all_valid = [seeds]  # include seeds

    for seed in seeds:
        for _ in range(cfg.walks_per_seed):
            current = seed.copy()
            walk_points = []

            for _ in range(cfg.steps_per_walk):
                # Random direction on simplex tangent space
                direction = rng.normal(size=n)
                direction = direction - direction.mean()  # zero-sum to stay on simplex
                direction = direction / (np.linalg.norm(direction) + 1e-10)

                # Take step
                new_point = current + cfg.step_size * direction
                new_point = _project_to_simplex(new_point.reshape(1, -1))[0]

                # Check validity
                valid = _check_validity(new_point.reshape(1, -1), event, premise_type)
                if valid[0]:
                    walk_points.append(new_point)
                    current = new_point

            if walk_points:
                all_valid.append(np.array(walk_points))

    result = np.vstack(all_valid)
    # Remove duplicates (approximately)
    result = np.unique(np.round(result, 8), axis=0)

    log.info(f"  [REFINE] Found {len(result)} total valid points")
    return result


def sample_valid_votes(
    event: Event,
    n_samples: int = 100000,
    seed: Optional[int] = None,
) -> SampleResult:
    """
    Sample random vote vectors and keep those satisfying the premise.
    Uses numba JIT for speed. Optionally refines with local walks.
    """
    warmup_jit()
    cfg = get_config()

    rng = np.random.default_rng(seed)
    premise_type = get_premise_type(event.season, event.n_eliminated, event.is_final)
    n = event.n

    # Initial random sampling
    log.info(f"  [SAMPLE] Generating {n_samples:,} samples for {n} contestants...")
    samples = rng.dirichlet(np.ones(n), size=n_samples)
    log.info(f"  [SAMPLE] Premise: {premise_type.name}")

    log.info(f"  [SAMPLE] Checking validity...")
    valid_mask = _check_validity(samples, event, premise_type)
    valid_samples = samples[valid_mask]
    acceptance = len(valid_samples) / n_samples

    log.info(f"  [SAMPLE] Initial: {len(valid_samples):,} valid ({acceptance:.2%} acceptance)")

    # Refinement if acceptance is low
    if cfg.refinement.enabled and acceptance < cfg.refinement.min_acceptance and len(valid_samples) > 0:
        log.info(f"  [REFINE] Low acceptance ({acceptance:.2%} < {cfg.refinement.min_acceptance:.0%}), refining...")
        valid_samples = refine_with_walks(valid_samples, event, premise_type, rng)

    return SampleResult(
        event=event,
        premise_type=premise_type,
        valid_samples=valid_samples,
        n_samples=n_samples,
        n_valid=len(valid_samples),
    )
