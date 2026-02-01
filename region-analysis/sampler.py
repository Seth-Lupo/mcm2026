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
    """Convert 1D scores to ranks (1=best, highest score).

    Competition ranking: ties share rank, next rank skips.
    E.g., [30, 30, 30, 28] -> [1, 1, 1, 4]
    """
    n = len(scores)
    order = np.argsort(-scores)
    ranks = np.empty(n, dtype=np.float64)

    current_rank = 1
    for i in range(n):
        if i > 0 and scores[order[i]] < scores[order[i - 1]]:
            current_rank = i + 1  # skip to position-based rank
        ranks[order[i]] = current_rank

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
        # Tiebreaker: lower fan votes = worse (higher in elimination order)
        sort_key = combined - fan_votes[b] * 1e-9
        order = np.argsort(-sort_key)
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
        # Tiebreaker: lower fan votes = worse
        sort_key = combined - fan_votes[b] * 1e-9
        order = np.argsort(-sort_key)
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
        # Tiebreaker: higher fan votes = better (lower sort key = ranks higher)
        sort_key = combined - fan_votes[b] * 1e-9
        pred_order = np.argsort(sort_key)

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


MAX_RETRIES = 20  # max retry attempts if 0 hits


def sample_valid_votes(
    event: Event,
    n_samples: int = 100000,
    seed: Optional[int] = None,
) -> SampleResult:
    """
    Sample random vote vectors and keep those satisfying the premise.
    Uses numba JIT for speed. Retries if 0 hits initially.
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
