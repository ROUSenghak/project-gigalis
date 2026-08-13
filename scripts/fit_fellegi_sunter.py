#!/usr/bin/env python3
"""Fit the Fellegi-Sunter probabilistic linkage model to the candidate pairs.

The rule-based scorer weights its four components by domain reasoning. This
script estimates those weights instead, without using any of the manually
reviewed labels, so the benchmark stays available for evaluation.

Two fits are produced:

``population_u``
    ``u`` is fixed at the marginal level frequencies of the candidate pairs.
    With roughly one true match in every two thousand pairs, those frequencies
    are almost exactly the non-match distribution. Only ``m`` and ``lambda``
    are free. This is the primary fit because it is by far the more stable.

``free_u``
    ``m``, ``u`` and ``lambda`` are all free. Reported as a robustness check:
    if the two fits disagree sharply, the primary fit is suspect and the
    disagreement is recorded rather than hidden.

The script also measures the association between comparison fields, because
the additive weight formula assumes they are conditionally independent and in
procurement data they are not: a pair with similar text also tends to share a
CPV code.

Outputs
    fellegi_sunter_model.json          fitted parameters and diagnostics
    linkage_candidates_scored.parquet  candidates + fs_match_weight/probability
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.fellegi_sunter import (
    comparison_vectors,
    cramers_v,
    fit_em,
    log_weights,
    match_probability,
    total_match_weight,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/boamp")
MODEL_VERSION = "boamp_fellegi_sunter_v1.0"

#: Expected order of magnitude for the match prior, from the benchmark base
#: rate (~26% of anchors have a successor) over ~230 candidates per anchor.
PLAUSIBLE_LAMBDA = (1e-5, 5e-2)


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "fit_fellegi_sunter.log", encoding="utf-8"),
        ],
    )


def independence_report(fields: dict[str, np.ndarray]) -> dict[str, float]:
    """Pairwise Cramer's V between comparison fields.

    The model assumes these are conditionally independent given match status.
    Since the overwhelming majority of pairs are non-matches, the unconditional
    association measured here is a close proxy for the within-non-match
    association the assumption actually concerns.
    """
    return {
        f"{left}~{right}": round(cramers_v(fields[left], fields[right]), 4)
        for left, right in combinations(sorted(fields), 2)
    }


def serialise(fit: dict[str, Any], fields: dict[str, np.ndarray]) -> dict[str, Any]:
    weights = log_weights(fit["m"], fit["u"])
    trace = fit["log_likelihood_trace"]
    monotone = all(later >= earlier - 1e-6 for earlier, later in zip(trace, trace[1:]))
    return {
        "lambda": fit["lambda"],
        "converged": fit["converged"],
        "iterations": fit["iterations"],
        "log_likelihood_final": trace[-1] if trace else None,
        "log_likelihood_monotone": monotone,
        "lambda_within_plausible_range": PLAUSIBLE_LAMBDA[0] <= fit["lambda"] <= PLAUSIBLE_LAMBDA[1],
        "m": {f: {level: round(v, 6) for level, v in d.items()} for f, d in fit["m"].items()},
        "u": {f: {level: round(v, 6) for level, v in d.items()} for f, d in fit["u"].items()},
        "log2_bayes_factor": {f: {level: round(v, 4) for level, v in d.items()}
                              for f, d in weights.items()},
    }


def build(project_root: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    scored_path = output_dir / "linkage_candidates_scored.parquet"
    model_path = output_dir / "fellegi_sunter_model.json"
    if scored_path.exists() and not force:
        raise FileExistsError(f"{scored_path} already exists. Use --force to rebuild.")

    candidates = pd.read_parquet(output_dir / "linkage_candidates.parquet")
    logging.info("Candidate pairs: %s", f"{len(candidates):,}")

    fields = comparison_vectors(
        buyer_match_type=candidates["buyer_match_type"].tolist(),
        text_component=candidates["text_component"].tolist(),
        cpv_component=candidates["cpv_component"].tolist(),
        time_component=candidates["time_component"].tolist(),
    )
    for name, levels in fields.items():
        counts = pd.Series(levels).value_counts().to_dict()
        logging.info("field %s levels: %s", name, counts)

    logging.info("Fitting primary model (u fixed at population frequencies)")
    population_u = {
        field: dict(distribution)
        for field, distribution in fit_em(fields, max_iterations=1)["population_distribution"].items()
    }
    primary = fit_em(fields, fixed_u=population_u)
    logging.info("  lambda=%.3g converged=%s iterations=%d",
                 primary["lambda"], primary["converged"], primary["iterations"])

    logging.info("Fitting robustness model (m, u and lambda all free)")
    free = fit_em(fields, fixed_u=None)
    logging.info("  lambda=%.3g converged=%s iterations=%d",
                 free["lambda"], free["converged"], free["iterations"])

    weights = log_weights(primary["m"], primary["u"])
    weight_values = total_match_weight(fields, weights, primary["lambda"])
    probabilities = match_probability(weight_values)

    scored = candidates.copy()
    scored["fs_match_weight"] = weight_values
    scored["fs_match_probability"] = probabilities
    scored["fs_model_version"] = MODEL_VERSION
    scored.to_parquet(scored_path, index=False, compression="zstd")

    # How far the two fits disagree on the quantity that actually drives
    # decisions -- the per-level log Bayes factors.
    free_weights = log_weights(free["m"], free["u"])
    divergence = {
        field: round(max(abs(weights[field][level] - free_weights[field][level])
                         for level in weights[field]), 4)
        for field in weights
    }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "source": str(output_dir / "linkage_candidates.parquet"),
        "outputs": {"scored_candidates": str(scored_path), "model": str(model_path)},
        "candidate_pairs": int(len(candidates)),
        "field_levels": {name: sorted({str(v) for v in levels}) for name, levels in fields.items()},
        "primary_fit_population_u": serialise(primary, fields),
        "robustness_fit_free_u": serialise(free, fields),
        "max_log2_bayes_factor_divergence_between_fits": divergence,
        "conditional_independence_cramers_v": independence_report(fields),
        "match_probability_distribution": {
            "mean": round(float(probabilities.mean()), 6),
            "p50": round(float(np.percentile(probabilities, 50)), 6),
            "p99": round(float(np.percentile(probabilities, 99)), 6),
            "above_0.50": int((probabilities >= 0.50).sum()),
            "above_0.90": int((probabilities >= 0.90).sum()),
            "above_0.95": int((probabilities >= 0.95).sum()),
            "above_0.99": int((probabilities >= 0.99).sum()),
        },
        # The fitted prior is a diagnostic, not a pass/fail criterion. An
        # inflated lambda shifts every pair's weight by the same constant and
        # so cannot change the ranking; it means the posterior should not be
        # read as a calibrated probability, which is reported rather than
        # treated as a failed fit.
        "validation_passed": bool(
            primary["converged"] and serialise(primary, fields)["log_likelihood_monotone"]
        ),
        "prior_is_calibrated": serialise(primary, fields)["lambda_within_plausible_range"],
    }
    model_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    configure_logging(project_root)
    summary = build(project_root, output_dir, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise RuntimeError("Fellegi-Sunter fit failed its convergence checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
