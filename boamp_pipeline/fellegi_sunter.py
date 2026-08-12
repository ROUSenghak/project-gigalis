"""Fellegi-Sunter probabilistic record linkage for procurement successors.

The rule-based scorer in :mod:`boamp_pipeline.linkage` combines its four
components with weights chosen by domain reasoning. This module estimates those
weights from the data instead, using the classical probabilistic linkage model
of Fellegi and Sunter (1969).

Each candidate pair is reduced to a discrete comparison vector. For every
comparison field ``k`` and observed level ``l`` the model holds two quantities:

* ``m[k][l] = P(level l | the pair is a true match)`` -- data quality;
* ``u[k][l] = P(level l | the pair is not a match)`` -- chance agreement.

Under conditional independence between fields the log Bayes factors add, so a
pair's total match weight and its posterior probability are

    M     = log2(lambda / (1 - lambda)) + sum_k log2(m[k][l_k] / u[k][l_k])
    P     = 2**M / (1 + 2**M)

where ``lambda`` is the prior probability that an arbitrary candidate pair is a
match.

Two properties matter for this project.

*It needs no labels.* ``m``, ``u`` and ``lambda`` are fitted by expectation
maximisation over the unlabelled candidate pairs, so the 22 positively labelled
benchmark anchors are free to be spent on evaluation rather than training.

*It formalises how missing evidence is treated.* An unobserved comparison is
given its own level rather than being dropped or scored as disagreement. If the
fitted ``m/u`` for that level comes out at 1 its log weight is 0 and the field
contributes nothing -- which is exactly the neutrality the rule-based scorer
assumes. Estimating it turns that assumption into a measurement.

Conditional independence between fields is an assumption of the model and is
generally false in practice; :func:`cramers_v` is provided so the dependence
can be reported rather than ignored.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

#: Level assigned when a comparison could not be observed at all.
MISSING_LEVEL = "missing"

#: Upper bin edges for the text-similarity comparison. Chosen to spread the
#: observed distribution (median ~0.09) rather than to optimise any metric.
TEXT_BINS: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50)
TEXT_LABELS: tuple[str, ...] = ("text<0.05", "text0.05-0.15", "text0.15-0.30",
                                "text0.30-0.50", "text>=0.50")

#: CPV continuity: no shared category, partial overlap, or strong overlap.
CPV_BINS: tuple[float, ...] = (0.001, 0.50)
CPV_LABELS: tuple[str, ...] = ("cpv_none", "cpv_partial", "cpv_strong")

#: Timing plausibility relative to the anchor's declared contract end.
TIME_BINS: tuple[float, ...] = (0.001, 0.50)
TIME_LABELS: tuple[str, ...] = ("time_low", "time_medium", "time_high")

#: Smoothing added to every level count so an unobserved level can never make
#: a weight infinite. Small relative to the ~868k pairs being fitted.
LAPLACE_ALPHA = 0.5

#: Guardrails for the EM fit. ``lambda`` is expected around 5e-4, so a value
#: at either bound signals a degenerate solution rather than convergence.
MIN_LAMBDA = 1e-9
MAX_LAMBDA = 0.5


def bin_value(value: float | None, edges: Sequence[float], labels: Sequence[str]) -> str:
    """Map a continuous comparison score to a discrete level.

    Returns :data:`MISSING_LEVEL` when the score is unobservable, so that
    absence of evidence becomes its own category instead of being silently
    folded into the lowest bin.
    """
    if value is None:
        return MISSING_LEVEL
    numeric = float(value)
    if math.isnan(numeric):
        return MISSING_LEVEL
    return labels[int(np.searchsorted(edges, numeric, side="right"))]


def comparison_vectors(
    buyer_match_type: Sequence[str],
    text_component: Sequence[float],
    cpv_component: Sequence[float],
    time_component: Sequence[float],
) -> dict[str, np.ndarray]:
    """Build the per-field level arrays for a set of candidate pairs.

    The buyer field is deliberately not an agree/disagree comparison: blocking
    already restricts candidates to the same buyer, so what remains to
    distinguish is *verified* identity (a Luhn-validated SIREN) from identity
    asserted only by an exact name match.
    """
    return {
        "buyer": np.asarray([f"buyer_{value}" for value in buyer_match_type], dtype=object),
        "text": np.asarray([bin_value(v, TEXT_BINS, TEXT_LABELS) for v in text_component], dtype=object),
        "cpv": np.asarray([bin_value(v, CPV_BINS, CPV_LABELS) for v in cpv_component], dtype=object),
        "time": np.asarray([bin_value(v, TIME_BINS, TIME_LABELS) for v in time_component], dtype=object),
    }


def level_distribution(levels: np.ndarray, vocabulary: Sequence[str],
                       weights: np.ndarray | None = None) -> dict[str, float]:
    """Laplace-smoothed distribution over a field's levels.

    ``weights`` lets the same routine serve the E-step, where each pair counts
    according to its current posterior probability of being a match.
    """
    counts = {level: LAPLACE_ALPHA for level in vocabulary}
    if weights is None:
        uniques, totals = np.unique(levels, return_counts=True)
        for level, total in zip(uniques, totals):
            counts[str(level)] = counts.get(str(level), LAPLACE_ALPHA) + float(total)
    else:
        for level in vocabulary:
            counts[level] += float(weights[levels == level].sum())
    grand_total = sum(counts.values())
    return {level: value / grand_total for level, value in counts.items()}


def vocabularies(fields: Mapping[str, np.ndarray]) -> dict[str, list[str]]:
    return {name: sorted({str(v) for v in levels}) for name, levels in fields.items()}


def log_weights(m: Mapping[str, Mapping[str, float]],
                u: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Per-level log2 Bayes factors ``log2(m/u)``.

    A level with ``m == u`` yields exactly 0: it moves the total match weight
    not at all, which is the formal statement of "this comparison carries no
    information about whether the pair matches".
    """
    return {
        field: {level: math.log2(m[field][level] / u[field][level]) for level in m[field]}
        for field in m
    }


def total_match_weight(fields: Mapping[str, np.ndarray],
                       weights: Mapping[str, Mapping[str, float]],
                       lambda_prior: float) -> np.ndarray:
    """Match weight ``M`` for every pair, including the prior term.

    Note that ``lambda_prior`` enters as a constant added to every pair, so it
    shifts the whole distribution of weights without changing their order. The
    *ranking* of candidates therefore depends only on the estimated
    ``log2(m/u)`` terms. This matters here: the unsupervised mixture cannot
    recover the true successor rate (see ``scripts/fit_fellegi_sunter.py``), so
    the fitted prior is not trustworthy as a calibration, but the ranking it
    produces is unaffected by that.
    """
    size = len(next(iter(fields.values())))
    total = np.full(size, math.log2(lambda_prior / (1.0 - lambda_prior)), dtype=float)
    for field, levels in fields.items():
        lookup = weights[field]
        total += np.asarray([lookup.get(str(level), 0.0) for level in levels], dtype=float)
    return total


def match_probability(weight: np.ndarray) -> np.ndarray:
    """Convert match weights to posterior probabilities.

    ``2**M / (1 + 2**M)`` is the logistic function of ``M * ln 2``, evaluated
    branchwise so that strongly negative weights cannot overflow ``exp``.
    """
    scaled = np.asarray(weight, dtype=float) * math.log(2.0)
    output = np.empty_like(scaled)
    positive = scaled >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-scaled[positive]))
    exponentiated = np.exp(scaled[~positive])
    output[~positive] = exponentiated / (1.0 + exponentiated)
    return output


def encode(fields: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Replace level strings with integer codes once, up front.

    EM revisits every pair on every iteration, so keeping the levels as Python
    strings would mean hundreds of millions of dict lookups. Encoding once lets
    each iteration be a handful of vectorised gathers instead.
    """
    codes: dict[str, np.ndarray] = {}
    vocab: dict[str, list[str]] = {}
    for field, levels in fields.items():
        uniques, inverse = np.unique(levels.astype(str), return_inverse=True)
        vocab[field] = [str(value) for value in uniques]
        codes[field] = inverse.astype(np.int32)
    return codes, vocab


def _log_component(codes: Mapping[str, np.ndarray], vocab: Mapping[str, list[str]],
                   distribution: Mapping[str, Mapping[str, float]], size: int) -> np.ndarray:
    """Summed log probability of the observed levels under one mixture class."""
    total = np.zeros(size, dtype=float)
    for field, code in codes.items():
        table = np.array([distribution[field][level] for level in vocab[field]], dtype=float)
        total += np.log(table)[code]
    return total


def log_likelihood(fields: Mapping[str, np.ndarray],
                   m: Mapping[str, Mapping[str, float]],
                   u: Mapping[str, Mapping[str, float]],
                   lambda_prior: float) -> float:
    """Observed-data log-likelihood of the two-class mixture."""
    codes, vocab = encode(fields)
    size = len(next(iter(codes.values())))
    log_m = math.log(lambda_prior) + _log_component(codes, vocab, m, size)
    log_u = math.log1p(-lambda_prior) + _log_component(codes, vocab, u, size)
    peak = np.maximum(log_m, log_u)
    return float(np.sum(peak + np.log(np.exp(log_m - peak) + np.exp(log_u - peak))))


def fit_em(
    fields: Mapping[str, np.ndarray],
    *,
    fixed_u: Mapping[str, Mapping[str, float]] | None = None,
    lambda_init: float = 5e-4,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> dict:
    """Fit ``m``, ``u`` and ``lambda`` by expectation maximisation.

    ``fixed_u`` holds the non-match distributions at values estimated from the
    pair population. That is the stable default here: with roughly one match in
    every two thousand candidate pairs, the marginal level frequencies are
    almost exactly the non-match distribution, and freeing ``u`` mostly adds a
    way for EM to wander into a degenerate solution.

    The returned ``log_likelihood_trace`` is kept so that monotonicity can be
    asserted rather than assumed -- with a prior this extreme, silent
    convergence to a useless fit is a real risk.
    """
    codes, vocab = encode(fields)
    size = len(next(iter(codes.values())))
    population = {field: level_distribution(levels, vocab[field])
                  for field, levels in fields.items()}

    u = {f: dict(d) for f, d in (fixed_u or population).items()}
    # Start m away from u, biased towards the levels that indicate agreement,
    # so the two mixture components are distinguishable at iteration 1.
    m = {}
    for field, levels in vocab.items():
        skew = {level: population[field][level] * (3.0 if _indicates_agreement(level) else 1.0)
                for level in levels}
        norm = sum(skew.values())
        m[field] = {level: value / norm for level, value in skew.items()}

    lambda_prior = float(lambda_init)
    trace: list[float] = []
    converged = False

    for _ in range(max_iterations):
        # E-step: posterior probability that each pair is a match.
        log_m = math.log(lambda_prior) + _log_component(codes, vocab, m, size)
        log_u = math.log1p(-lambda_prior) + _log_component(codes, vocab, u, size)
        peak = np.maximum(log_m, log_u)
        exp_m, exp_u = np.exp(log_m - peak), np.exp(log_u - peak)
        responsibility = exp_m / (exp_m + exp_u)
        trace.append(float(np.sum(peak + np.log(exp_m + exp_u))))

        # M-step: re-estimate each level's probability, weighting every pair by
        # its current posterior. Counting is a bincount over the integer codes.
        lambda_prior = float(np.clip(responsibility.mean(), MIN_LAMBDA, MAX_LAMBDA))
        for field, code in codes.items():
            width = len(vocab[field])
            matched = np.bincount(code, weights=responsibility, minlength=width) + LAPLACE_ALPHA
            m[field] = dict(zip(vocab[field], matched / matched.sum()))
            if fixed_u is None:
                unmatched = np.bincount(code, weights=1.0 - responsibility, minlength=width) + LAPLACE_ALPHA
                u[field] = dict(zip(vocab[field], unmatched / unmatched.sum()))

        # Relative tolerance: the log-likelihood here is of order 1e6, so an
        # absolute threshold would demand more significant digits than double
        # precision carries and could never be met.
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < tolerance * abs(trace[-1]):
            converged = True
            break

    return {
        "m": m,
        "u": u,
        "lambda": lambda_prior,
        "log_likelihood_trace": trace,
        "iterations": len(trace),
        "converged": converged,
        "u_estimated_from_population": fixed_u is not None,
        "population_distribution": population,
    }


def _indicates_agreement(level: str) -> bool:
    """Whether a level is the kind of evidence a true match should show."""
    if level == MISSING_LEVEL:
        return False
    return level in {"buyer_siren", "cpv_strong", "cpv_partial", "time_high",
                     "text>=0.50", "text0.30-0.50"}


def cramers_v(left: np.ndarray, right: np.ndarray) -> float:
    """Cramer's V between two categorical arrays.

    Used to report the dependence between comparison fields, since the model
    assumes they are conditionally independent and they generally are not.
    """
    left_levels, left_index = np.unique(left, return_inverse=True)
    right_levels, right_index = np.unique(right, return_inverse=True)
    if len(left_levels) < 2 or len(right_levels) < 2:
        return 0.0
    table = np.zeros((len(left_levels), len(right_levels)), dtype=float)
    np.add.at(table, (left_index, right_index), 1.0)
    total = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    chi2 = float(np.sum((table - expected) ** 2 / np.where(expected > 0, expected, 1.0)))
    return math.sqrt((chi2 / total) / (min(table.shape) - 1))
