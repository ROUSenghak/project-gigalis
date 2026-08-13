"""Coverage of the national geography map and the notice-link repair."""

from boamp_pipeline.geography import (
    CVL_DOM_OTHER,
    DEPARTMENT_GROUPS,
    GRAND_OUEST,
    GROUP_BY_REGION,
    REGION_BY_DEPARTMENT,
    UNRESOLVED_DEPARTMENT,
    department_group,
    is_grand_ouest,
    normalize_department,
    region_for_department,
)
from boamp_pipeline.notice_links import (
    clean_linked_notice_ids,
    is_plausible_id,
    link_partners,
    looks_like_date_fragment,
    pollution_report,
)
from boamp_pipeline.standardize import GRAND_OUEST_REGION_BY_DEPARTMENT


def test_every_metropolitan_department_has_a_region() -> None:
    """96 metropolitan departments: 01-19, 21-95, plus Corsica's 2A and 2B."""
    expected = {f"{n:02d}" for n in range(1, 96)} - {"20"} | {"2A", "2B"}

    missing = expected - set(REGION_BY_DEPARTMENT)

    assert not missing, f"unmapped departments: {sorted(missing)}"
    assert len(expected) == 96


def test_every_region_maps_to_a_sampling_group() -> None:
    unmapped = set(REGION_BY_DEPARTMENT.values()) - set(GROUP_BY_REGION)

    assert not unmapped, f"regions with no sampling group: {sorted(unmapped)}"
    assert set(GROUP_BY_REGION.values()) <= set(DEPARTMENT_GROUPS)


def test_grand_ouest_group_matches_the_frozen_study_scope() -> None:
    """The new national map must not silently redefine the existing cohort."""
    for department in GRAND_OUEST_REGION_BY_DEPARTMENT:
        assert is_grand_ouest(department), department

    # And nothing outside the frozen fourteen may join the group.
    in_group = {d for d in REGION_BY_DEPARTMENT if department_group(d) == GRAND_OUEST}
    assert in_group == set(GRAND_OUEST_REGION_BY_DEPARTMENT)


def test_corsica_and_overseas_shapes_from_department_from_postcode() -> None:
    """standardize emits "2A/2B" for 20xxx and catch-alls 97/98."""
    assert region_for_department("2A/2B") == "Corse"
    assert region_for_department("971") == "Guadeloupe"
    assert department_group("974") == CVL_DOM_OTHER
    assert department_group("98") == CVL_DOM_OTHER


def test_unresolved_geography_is_representable_not_dropped() -> None:
    """A blank department keeps a positive inclusion probability."""
    assert normalize_department("") == UNRESOLVED_DEPARTMENT
    assert normalize_department(None) == UNRESOLVED_DEPARTMENT
    assert region_for_department("") == ""
    assert department_group("") == CVL_DOM_OTHER
    assert normalize_department("1") == "01"
    assert normalize_department("2a") == "2A"


def test_date_fragments_are_stripped_from_linked_notice_ids() -> None:
    """The standardize.py:53 defect: "2014-10-15" yields a bogus "10-15"."""
    known = {"14-151460", "14-91521", "14-102587"}

    assert clean_linked_notice_ids('["14-151460","10-15"]', known) == ["14-151460"]
    assert clean_linked_notice_ids(
        '["14-91521","14-102587","07-03","07-08","07-09"]', known
    ) == ["14-91521", "14-102587"]


def test_genuine_pre_corpus_references_survive() -> None:
    """443 real identifiers with pre-2015 prefixes are referenced from inside
    the corpus; all carry 4-6 digit sequences, unlike two-digit day fragments."""
    assert clean_linked_notice_ids('["12-172293","06-259536"]', set()) == [
        "12-172293",
        "06-259536",
    ]
    assert is_plausible_id("06-259536")
    assert not is_plausible_id("03-31")
    assert not is_plausible_id("99-123456")


def test_fragment_shape_never_collides_with_a_real_identifier() -> None:
    """Measured over all 1,620,712 corpus identifiers: zero collisions."""
    assert looks_like_date_fragment("03-31")
    assert looks_like_date_fragment("09-30")
    assert not looks_like_date_fragment("15-21191")
    # A month-like prefix with a real sequence length is not a fragment.
    assert not looks_like_date_fragment("12-172293")


def test_link_partners_is_symmetric_and_ignores_self_links() -> None:
    partners = link_partners(
        [("15-1000", '["15-2000","10-15"]'), ("15-3000", '["15-3000"]')],
        {"15-1000", "15-2000", "15-3000"},
    )

    assert partners["15-1000"] == {"15-2000"}
    assert partners["15-2000"] == {"15-1000"}
    assert "15-3000" not in partners


def test_pollution_report_separates_fragments_from_other_rejects() -> None:
    report = pollution_report(
        ['["14-151460","10-15"]', '["14-151460"]', "[]", None], {"14-151460"}
    )

    assert report["notices_with_links"] == 2
    assert report["notices_with_fabricated_ids"] == 1
    assert report["raw_tokens"] == 3
    assert report["kept_tokens"] == 2
    assert report["date_fragment_tokens"] == 1
    assert report["other_rejected_tokens"] == 0
