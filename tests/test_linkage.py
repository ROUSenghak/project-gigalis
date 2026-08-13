import math

from boamp_pipeline.linkage import (
    MAX_GAP_DAYS,
    MIN_GAP_DAYS,
    MISSING,
    classify_buyer_match,
    compatible_buyer_identifiers,
    cpv_continuity_score,
    evidence_completeness,
    is_digital,
    normalize_buyer_for_blocking,
    time_plausibility_score,
    weighted_score,
)


def test_missing_cpv_is_unobservable_not_disagreement() -> None:
    """A missing CPV must not be scored as if the categories disagreed."""
    assert math.isnan(cpv_continuity_score([], ["72000000"]))
    assert cpv_continuity_score(["72000000"], ["45000000"]) == 0.0
    assert cpv_continuity_score(["72212000"], ["72212100"]) > 0.0


def test_missing_duration_does_not_assume_four_years() -> None:
    """Notebook 05 assumed a 4-year contract when duration was unknown."""
    assert math.isnan(time_plausibility_score(1460, None))
    assert math.isnan(time_plausibility_score(1460, float("nan")))
    assert time_plausibility_score(1461, 48.0) > 0.9


def test_weighted_score_renormalises_over_available_evidence() -> None:
    weights = {"buyer": 0.5, "text": 0.25, "cpv": 0.25}

    complete = weighted_score({"buyer": 1.0, "text": 1.0, "cpv": 0.0}, weights)
    missing_cpv = weighted_score({"buyer": 1.0, "text": 1.0, "cpv": MISSING}, weights)

    assert complete == 75.0
    # Dropping unobservable CPV must not drag the score down like a zero would.
    assert missing_cpv == 100.0
    assert math.isnan(weighted_score({"buyer": MISSING}, weights))
    assert evidence_completeness({"buyer": 1.0, "cpv": MISSING}) == 1


def test_unknown_buyer_identifiers_never_match_each_other() -> None:
    """Two empty SIRENs are both unknown, which is not evidence of identity."""
    assert classify_buyer_match("", "", "ville de nantes", "nantes metropole") != "siren"
    assert classify_buyer_match("264400136", "264400136", "a", "b") == "siren"
    assert classify_buyer_match("", "", "nantes", "nantes") == "normalized_name"


def test_conflicting_verified_buyer_identifiers_block_name_match() -> None:
    """A shared label cannot override a validated SIREN conflict."""
    assert compatible_buyer_identifiers("213502388", "213502388") is True
    assert compatible_buyer_identifiers("213502388", "") is True
    assert compatible_buyer_identifiers("", "243500139") is True
    assert compatible_buyer_identifiers("213502388", "243500139") is False
    assert (
        classify_buyer_match(
            "213502388",
            "243500139",
            "rennes",
            "rennes",
        )
        == "none"
    )


def test_succession_window_excludes_concurrent_procurement() -> None:
    """The floor keeps parallel lots out; the shortest confirmed link is 139d."""
    assert 0 < MIN_GAP_DAYS < 139
    assert MAX_GAP_DAYS > 2644  # longest confirmed benchmark link


def test_generic_public_body_prefixes_block_together() -> None:
    assert normalize_buyer_for_blocking("Commune de Nantes") == normalize_buyer_for_blocking("NANTES")
    assert is_digital(["72212000", "45000000"]) is True
    assert is_digital(["45000000"]) is False


def test_intercommunal_legal_forms_are_not_removed_from_buyer_blocking() -> None:
    """A commune and its intercommunal authority are distinct legal buyers."""
    assert normalize_buyer_for_blocking("Ville de Rennes") == "rennes"
    assert normalize_buyer_for_blocking("Commune de Rennes") == "rennes"
    assert normalize_buyer_for_blocking("Mairie de Rennes") == "rennes"
    assert normalize_buyer_for_blocking("Rennes Métropole") == "rennes metropole"
    assert normalize_buyer_for_blocking("Nantes Métropole") != normalize_buyer_for_blocking("Ville de Nantes")
    assert normalize_buyer_for_blocking("Communauté d'agglomération de Saint-Malo") != "saint malo"
    assert normalize_buyer_for_blocking("Communauté de communes du Pays de Redon") != "pays de redon"
