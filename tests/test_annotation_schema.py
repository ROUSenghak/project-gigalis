"""The annotation contract, tested against the ways v1's failed.

Each test names the v1 defect it prevents. The suite is deliberately hostile:
it tries to submit fabricated quotations, boilerplate reasoning, mechanically
assigned confidence and silently skipped candidates, and requires the schema to
refuse all of them.
"""

import pytest

from boamp_pipeline.annotation_schema import (
    ANCHOR_UNUSABLE,
    EVENT_SETS,
    EXTENSION_SAME_CONTRACT,
    HAS_RENEWAL_SUCCESSOR,
    LABELS,
    NO_RENEWAL_AMONG_SHOWN,
    NO_RENEWAL_POOL_EXHAUSTIVE,
    PARALLEL_LOT,
    RENEWAL_OF_EXPIRING,
    UNRELATED,
    agreement_report,
    cohen_kappa,
    conditional_entropy,
    quote_is_verbatim,
    similarity,
    validate_anchor_record,
    validate_batch,
    validate_label_record,
)

ANCHOR_TEXT = (
    "Marche d'exploitation et de maintenance des installations thermiques de la "
    "ville de Trappes, conclu pour une duree de quatre ans a compter de sa "
    "notification."
)
CANDIDATE_TEXT = (
    "La presente consultation a pour objet le renouvellement des contrats "
    "d'exploitation des installations thermiques incluant le chauffage et la "
    "ventilation des batiments communaux."
)


def dossier() -> dict:
    return {
        "dossier_id": "D-000001",
        "anchor": {
            "slot_id": "ANCHOR",
            "objet": "Exploitation des installations thermiques",
            "full_text": ANCHOR_TEXT,
            "award_date": "2019-04-28",
        },
        "candidates": [
            {
                "slot_id": "C01",
                "objet": "Renouvellement des contrats d'exploitation",
                "full_text": CANDIDATE_TEXT,
                "first_notice_date": "2023-02-11",
            },
            {
                "slot_id": "C02",
                "objet": "Fourniture de vehicules electriques",
                "full_text": "Acquisition de vehicules electriques pour la flotte municipale.",
                "first_notice_date": "2021-06-01",
            },
        ],
        "pool_disclosure": {"exposure_mode": "EXHAUSTIVE", "candidates_shown": 2,
                            "pool_size_bucket": "1-30"},
    }


def good_label(slot: str = "C01", label: str = RENEWAL_OF_EXPIRING,
               confidence: str = "HIGH", reasoning: str | None = None) -> dict:
    return {
        "candidate_slot": slot,
        "label": label,
        "confidence": confidence,
        "reasoning": reasoning or (
            "The candidate states it renews the thermal installation operating "
            "contracts, and the anchor is that operating contract with a four-year term."
        ),
        "evidence": [
            {"slot_id": "ANCHOR", "field": "full_text",
             "quote": "Marche d'exploitation et de maintenance des installations thermiques"},
            {"slot_id": "C01", "field": "full_text",
             "quote": "le renouvellement des contrats d'exploitation des installations thermiques"},
        ],
    }


# ---------------------------------------------------------------------------
# Fabricated evidence
# ---------------------------------------------------------------------------


def test_a_quote_must_actually_appear_in_the_field_it_cites() -> None:
    """The check that makes model-produced evidence auditable."""
    record = good_label()
    record["evidence"][1]["quote"] = "le renouvellement du marche de televisions numeriques"

    problems = validate_label_record(record, dossier())

    assert any("not a verbatim substring" in p for p in problems)


def test_whitespace_differences_do_not_break_a_genuine_quote() -> None:
    assert quote_is_verbatim("installations   thermiques", "des installations thermiques de")
    assert not quote_is_verbatim("installations electriques", "des installations thermiques de")


def test_evidence_citing_an_unknown_slot_or_field_is_rejected() -> None:
    record = good_label()
    record["evidence"].append({"slot_id": "C99", "field": "full_text", "quote": "x" * 30})
    record["evidence"].append({"slot_id": "C01", "field": "secret_score", "quote": "x" * 30})

    problems = validate_label_record(record, dossier())

    assert any("unknown slot" in p for p in problems)
    assert any("absent from slot" in p for p in problems)


def test_a_relation_needs_evidence_from_both_sides() -> None:
    """A claim about two procurements cannot rest on one of them."""
    record = good_label()
    record["evidence"] = [record["evidence"][1]]  # candidate only

    problems = validate_label_record(record, dossier())

    assert any("both the anchor and the candidate" in p for p in problems)


def test_unrelated_may_rest_on_the_candidate_alone() -> None:
    record = good_label(slot="C02", label=UNRELATED)
    record["evidence"] = [
        {"slot_id": "C02", "field": "full_text",
         "quote": "Acquisition de vehicules electriques pour la flotte municipale"}
    ]

    assert validate_label_record(record, dossier()) == []


def test_a_renewal_must_follow_the_anchor_award() -> None:
    record = good_label()
    doc = dossier()
    doc["candidates"][0]["first_notice_date"] = "2018-01-01"

    problems = validate_label_record(record, doc)

    assert any("requires the candidate to follow" in p for p in problems)


# ---------------------------------------------------------------------------
# Boilerplate: v1's 70 negatives shared one reason string
# ---------------------------------------------------------------------------


def test_identical_reasoning_across_records_fails_the_batch() -> None:
    boilerplate = (
        "No later episode in the supplied review candidates matched both the buyer "
        "continuity and the anchor's specific functional need."
    )
    records = [good_label(reasoning=boilerplate) for _ in range(5)]

    report = validate_batch(records)

    assert report["distinct_reasoning_ratio"] < 1.0
    assert not report["passed"]


def test_near_identical_reasoning_also_fails() -> None:
    """Swapping a noun is not per-item reasoning."""
    records = [
        good_label(reasoning=(
            "The candidate renews the thermal installation operating contract held by "
            f"the buyer, which is the anchor contract for site number {i}."
        ))
        for i in range(4)
    ]

    report = validate_batch(records)

    assert report["max_reasoning_similarity"] >= 0.85
    assert not report["passed"]


def test_genuinely_distinct_reasoning_passes() -> None:
    records = [
        good_label(reasoning=(
            "The candidate names the renewal of thermal operating contracts and the "
            "anchor is that contract, with a four-year term ending in 2023."
        )),
        good_label(slot="C02", label=UNRELATED, confidence="MEDIUM", reasoning=(
            "Vehicle acquisition has no functional overlap with thermal installation "
            "maintenance, and no shared reference or supplier connects them."
        )),
    ]
    records[1]["evidence"] = [
        {"slot_id": "C02", "field": "full_text",
         "quote": "Acquisition de vehicules electriques pour la flotte municipale"}
    ]

    report = validate_batch(records)

    assert report["distinct_reasoning_ratio"] == 1.0
    assert report["passed"]


def test_similarity_is_zero_for_unrelated_text() -> None:
    assert similarity("the contract renews thermal maintenance for the city", "") == 0.0


# ---------------------------------------------------------------------------
# Degenerate confidence: v1's was a function of the label
# ---------------------------------------------------------------------------


def test_confidence_fixed_by_label_fails_the_batch() -> None:
    """In v1 every negative was MEDIUM and every out-of-scope call HIGH, so
    confidence carried no information at all."""
    records = [
        good_label(confidence="HIGH", reasoning=f"Distinct reasoning number {i} about "
                   f"the renewal of the thermal contract at site {i}.")
        for i in range(12)
    ]

    report = validate_batch(records)

    assert RENEWAL_OF_EXPIRING in report["labels_with_degenerate_confidence"]
    assert not report["passed"]


def test_varying_confidence_within_a_label_passes() -> None:
    records = []
    for i in range(12):
        records.append(good_label(
            confidence="HIGH" if i % 3 else "MEDIUM",
            reasoning=(
                f"Site {i}: the candidate cites renewal of the operating contract and "
                f"the anchor term ends the same year, so the succession is explicit."
            ),
        ))

    report = validate_batch(records)

    assert report["confidence_entropy_by_label"][RENEWAL_OF_EXPIRING] > 0
    assert not report["labels_with_degenerate_confidence"]


def test_conditional_entropy_is_zero_only_when_mechanical() -> None:
    assert conditional_entropy([("A", "HIGH"), ("A", "HIGH")])["A"] == 0.0
    assert conditional_entropy([("A", "HIGH"), ("A", "LOW")])["A"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Anchor verdicts: what a negative is allowed to mean
# ---------------------------------------------------------------------------


def test_exhaustion_cannot_be_claimed_when_only_part_was_shown() -> None:
    """v1's central defect: 'no successor' meant 'not in the 25 I was shown'."""
    doc = dossier()
    doc["pool_disclosure"]["exposure_mode"] = "SAMPLED"
    record = {
        "anchor_verdict": NO_RENEWAL_POOL_EXHAUSTIVE,
        "labels": [
            good_label(slot="C01", label=UNRELATED),
            good_label(slot="C02", label=UNRELATED),
        ],
    }
    for item in record["labels"]:
        item["evidence"] = [
            {"slot_id": item["candidate_slot"], "field": "full_text",
             "quote": (CANDIDATE_TEXT if item["candidate_slot"] == "C01"
                       else "Acquisition de vehicules electriques pour la flotte municipale")[:60]}
        ]

    problems = validate_anchor_record(record, doc)

    assert any("cannot claim the pool was exhausted" in p for p in problems)


def test_every_shown_candidate_must_be_labelled() -> None:
    """Silent truncation is impossible by construction."""
    record = {"anchor_verdict": HAS_RENEWAL_SUCCESSOR, "labels": [good_label("C01")]}

    problems = validate_anchor_record(record, dossier())

    assert any("slots left unlabelled" in p for p in problems)
    assert any("C02" in p for p in problems)


def test_a_slot_cannot_be_labelled_twice() -> None:
    record = {
        "anchor_verdict": HAS_RENEWAL_SUCCESSOR,
        "labels": [good_label("C01"), good_label("C01"), good_label("C02", UNRELATED)],
    }
    record["labels"][2]["evidence"] = [
        {"slot_id": "C02", "field": "full_text",
         "quote": "Acquisition de vehicules electriques pour la flotte municipale"}
    ]

    problems = validate_anchor_record(record, dossier())

    assert any("labelled more than once" in p for p in problems)


def test_verdict_and_labels_must_agree() -> None:
    record = {
        "anchor_verdict": NO_RENEWAL_POOL_EXHAUSTIVE,
        "labels": [good_label("C01", RENEWAL_OF_EXPIRING), good_label("C02", UNRELATED)],
    }
    record["labels"][1]["evidence"] = [
        {"slot_id": "C02", "field": "full_text",
         "quote": "Acquisition de vehicules electriques pour la flotte municipale"}
    ]

    problems = validate_anchor_record(record, dossier())

    assert any("claims no renewal but a slot is labelled" in p for p in problems)


def test_a_complete_consistent_anchor_record_validates() -> None:
    record = {
        "anchor_verdict": HAS_RENEWAL_SUCCESSOR,
        "labels": [
            good_label("C01", RENEWAL_OF_EXPIRING),
            good_label("C02", UNRELATED, confidence="MEDIUM", reasoning=(
                "Electric vehicle acquisition shares no functional need with thermal "
                "installation operation and no identifier links the two procurements."
            )),
        ],
    }
    record["labels"][1]["evidence"] = [
        {"slot_id": "C02", "field": "full_text",
         "quote": "Acquisition de vehicules electriques pour la flotte municipale"}
    ]

    assert validate_anchor_record(record, dossier()) == []


# ---------------------------------------------------------------------------
# Taxonomy and agreement
# ---------------------------------------------------------------------------


def test_event_sets_nest_from_strict_to_broad() -> None:
    assert EVENT_SETS["strict"] < EVENT_SETS["primary"] < EVENT_SETS["broad"]
    assert PARALLEL_LOT not in EVENT_SETS["broad"]
    assert set(EVENT_SETS["broad"]) <= set(LABELS)


def test_kappa_is_chance_corrected() -> None:
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == pytest.approx(1.0)
    assert cohen_kappa(["A", "A", "B", "B"], ["B", "B", "A", "A"]) < 0
    assert cohen_kappa([], []) == 0.0


def test_agreement_report_lists_disagreements_and_states_its_own_limit() -> None:
    a = {("EP-1", "C01"): RENEWAL_OF_EXPIRING, ("EP-1", "C02"): UNRELATED}
    b = {("EP-1", "C01"): RENEWAL_OF_EXPIRING, ("EP-1", "C02"): PARALLEL_LOT}

    report = agreement_report(a, b)

    assert report["overlapping_pairs"] == 2
    assert report["raw_agreement"] == 0.5
    assert len(report["disagreements"]) == 1
    assert "self-consistency" in report["interpretation"]
