import math

import numpy as np

from boamp_pipeline.fellegi_sunter import (
    MISSING_LEVEL,
    bin_value,
    comparison_vectors,
    cramers_v,
    fit_em,
    log_weights,
    match_probability,
    total_match_weight,
)


def test_unobservable_comparison_gets_its_own_level() -> None:
    """Missing evidence must not collapse into the lowest agreement bin."""
    assert bin_value(None, (0.5,), ("low", "high")) == MISSING_LEVEL
    assert bin_value(float("nan"), (0.5,), ("low", "high")) == MISSING_LEVEL
    assert bin_value(0.0, (0.5,), ("low", "high")) == "low"
    assert bin_value(0.9, (0.5,), ("low", "high")) == "high"


def test_uninformative_level_contributes_exactly_zero_weight() -> None:
    """m == u means the comparison says nothing about whether the pair matches.

    This is the formal version of the project's "missing evidence is not
    negative evidence" rule.
    """
    m = {"cpv": {"cpv_none": 0.5, MISSING_LEVEL: 0.5}}
    u = {"cpv": {"cpv_none": 0.5, MISSING_LEVEL: 0.5}}

    weights = log_weights(m, u)

    assert weights["cpv"][MISSING_LEVEL] == 0.0
    assert weights["cpv"]["cpv_none"] == 0.0


def test_match_weight_is_prior_plus_summed_log_bayes_factors() -> None:
    """Hand-computable case: two fields, each doubling the odds."""
    fields = {"a": np.array(["hit"], dtype=object), "b": np.array(["hit"], dtype=object)}
    weights = {"a": {"hit": 1.0}, "b": {"hit": 2.0}}
    lambda_prior = 0.5  # log2(0.5/0.5) == 0, so the prior term drops out

    weight = total_match_weight(fields, weights, lambda_prior)

    assert weight[0] == 3.0
    # 2**3 / (1 + 2**3) == 8/9
    assert math.isclose(float(match_probability(weight)[0]), 8 / 9, rel_tol=1e-9)


def test_match_probability_is_overflow_safe() -> None:
    extreme = np.array([-5000.0, 0.0, 5000.0])

    probabilities = match_probability(extreme)

    assert probabilities[0] == 0.0
    assert math.isclose(float(probabilities[1]), 0.5)
    assert probabilities[2] == 1.0
    assert np.all(np.isfinite(probabilities))


def test_em_increases_log_likelihood_and_separates_the_classes() -> None:
    """A planted set of matches must be recovered without any labels."""
    rng = np.random.default_rng(0)
    n_non_match, n_match = 4000, 200
    text = np.concatenate([
        rng.choice(["text<0.05", "text0.05-0.15"], n_non_match),
        rng.choice(["text>=0.50", "text0.30-0.50"], n_match),
    ])
    cpv = np.concatenate([
        rng.choice(["cpv_none", MISSING_LEVEL], n_non_match),
        rng.choice(["cpv_strong", "cpv_partial"], n_match),
    ])
    fields = {"text": text.astype(object), "cpv": cpv.astype(object)}

    fitted = fit_em(fields, lambda_init=0.05)

    trace = fitted["log_likelihood_trace"]
    assert all(b >= a - 1e-6 for a, b in zip(trace, trace[1:])), "EM must not decrease the likelihood"
    assert fitted["converged"]
    # The planted match rate is 200/4200; EM should land in the right region.
    assert 0.01 < fitted["lambda"] < 0.15
    # Agreement levels must be more likely among matches than among non-matches.
    assert fitted["m"]["text"]["text>=0.50"] > fitted["u"]["text"]["text>=0.50"]
    assert fitted["m"]["cpv"]["cpv_strong"] > fitted["u"]["cpv"]["cpv_strong"]


def test_distributions_are_normalised_per_field() -> None:
    fields = comparison_vectors(
        buyer_match_type=["siren", "normalized_name", "siren"],
        text_component=[0.9, 0.01, float("nan")],
        cpv_component=[0.8, float("nan"), 0.0],
        time_component=[float("nan"), float("nan"), 0.7],
    )

    fitted = fit_em(fields, lambda_init=0.2, max_iterations=20)

    assert fields["text"][2] == MISSING_LEVEL
    assert fields["time"][0] == MISSING_LEVEL
    for field in fields:
        assert math.isclose(sum(fitted["m"][field].values()), 1.0, rel_tol=1e-9)
        assert math.isclose(sum(fitted["u"][field].values()), 1.0, rel_tol=1e-9)


def test_cramers_v_detects_dependence_between_fields() -> None:
    independent_left = np.array(["a", "a", "b", "b"] * 50, dtype=object)
    independent_right = np.array(["x", "y", "x", "y"] * 50, dtype=object)
    identical = independent_left.copy()

    assert cramers_v(independent_left, independent_right) < 0.05
    assert cramers_v(independent_left, identical) > 0.95
