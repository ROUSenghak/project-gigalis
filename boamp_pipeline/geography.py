"""National French administrative geography for benchmark stratification.

``boamp_pipeline.standardize`` resolves a department for 97.95% of notices
nationally but only assigns ``buyer_region`` for the fourteen Grand Ouest
departments, because that map exists to define the study scope rather than to
describe France. Outside Grand Ouest ``buyer_region`` is therefore empty *by
construction*, not by extraction failure -- 88% blank across the national
episode layer.

A national benchmark has to be stratified on geography, so this module supplies
the map that was missing: every department to its 2016 region, and every region
to one of six sampling groups. It deliberately does not touch
:data:`boamp_pipeline.standardize.GRAND_OUEST_REGION_BY_DEPARTMENT`, which is
frozen -- changing it would silently redefine the existing study cohort.

Department codes follow the conventions produced by
:func:`boamp_pipeline.standardize.department_from_postcode`:

* two digits for metropolitan France;
* the literal ``"2A/2B"`` for Corsica, which that function does not split
  because postcode 20xxx is ambiguous between the two departments;
* three digits for the overseas departments 971-976;
* ``"97"`` and ``"98"`` as catch-alls for the remaining overseas postcodes
  (Saint-Pierre-et-Miquelon, Saint-Martin, New Caledonia, Polynesia, Monaco),
  which fall through to the generic two-digit branch.
"""

from __future__ import annotations

from typing import Mapping

#: Sampling groups. Six is a compromise: the 103 department codes present in
#: the digital cohort cannot each carry a stratum, and the thirteen metropolitan
#: regions leave cells too small to allocate against. Grand Ouest is kept whole
#: rather than merged into a larger west group so that v3 results stay directly
#: comparable with the existing Grand Ouest study.
GRAND_OUEST = "GRAND_OUEST"
ILE_DE_FRANCE = "ILE_DE_FRANCE"
NORD_EST = "NORD_EST"
SUD_EST = "SUD_EST"
SUD_OUEST = "SUD_OUEST"
CVL_DOM_OTHER = "CVL_DOM_OTHER"

DEPARTMENT_GROUPS: tuple[str, ...] = (
    GRAND_OUEST,
    ILE_DE_FRANCE,
    NORD_EST,
    SUD_EST,
    SUD_OUEST,
    CVL_DOM_OTHER,
)

#: Marks an episode whose notices disagree on department, or whose postcode
#: never resolved. Blankness is informative -- it correlates with poor buyer
#: extraction -- so it gets its own value rather than being dropped or imputed.
UNRESOLVED_DEPARTMENT = "UNRESOLVED"

REGION_BY_DEPARTMENT: Mapping[str, str] = {
    # Auvergne-Rhone-Alpes
    **{d: "Auvergne-Rhone-Alpes" for d in
       ("01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74")},
    # Bourgogne-Franche-Comte
    **{d: "Bourgogne-Franche-Comte" for d in
       ("21", "25", "39", "58", "70", "71", "89", "90")},
    # Bretagne
    **{d: "Bretagne" for d in ("22", "29", "35", "56")},
    # Centre-Val de Loire
    **{d: "Centre-Val de Loire" for d in ("18", "28", "36", "37", "41", "45")},
    # Corse. "2A/2B" is the form standardize.department_from_postcode emits;
    # the split codes are accepted too in case a cleaner source appears.
    **{d: "Corse" for d in ("2A", "2B", "2A/2B")},
    # Grand Est
    **{d: "Grand Est" for d in
       ("08", "10", "51", "52", "54", "55", "57", "67", "68", "88")},
    # Hauts-de-France
    **{d: "Hauts-de-France" for d in ("02", "59", "60", "62", "80")},
    # Ile-de-France
    **{d: "Ile-de-France" for d in
       ("75", "77", "78", "91", "92", "93", "94", "95")},
    # Normandie
    **{d: "Normandie" for d in ("14", "27", "50", "61", "76")},
    # Nouvelle-Aquitaine
    **{d: "Nouvelle-Aquitaine" for d in
       ("16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87")},
    # Occitanie
    **{d: "Occitanie" for d in
       ("09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82")},
    # Pays de la Loire
    **{d: "Pays de la Loire" for d in ("44", "49", "53", "72", "85")},
    # Provence-Alpes-Cote d'Azur
    **{d: "Provence-Alpes-Cote d'Azur" for d in
       ("04", "05", "06", "13", "83", "84")},
    # Overseas departments
    "971": "Guadeloupe",
    "972": "Martinique",
    "973": "Guyane",
    "974": "La Reunion",
    "976": "Mayotte",
    # Catch-alls produced by the generic two-digit branch for the remaining
    # overseas postcodes. Kept distinct from the resolved DOM codes.
    "97": "Outre-mer (autre)",
    "98": "Outre-mer (autre)",
}

GROUP_BY_REGION: Mapping[str, str] = {
    "Bretagne": GRAND_OUEST,
    "Pays de la Loire": GRAND_OUEST,
    "Normandie": GRAND_OUEST,
    "Ile-de-France": ILE_DE_FRANCE,
    "Hauts-de-France": NORD_EST,
    "Grand Est": NORD_EST,
    "Bourgogne-Franche-Comte": NORD_EST,
    "Auvergne-Rhone-Alpes": SUD_EST,
    "Provence-Alpes-Cote d'Azur": SUD_EST,
    "Corse": SUD_EST,
    "Nouvelle-Aquitaine": SUD_OUEST,
    "Occitanie": SUD_OUEST,
    "Centre-Val de Loire": CVL_DOM_OTHER,
    "Guadeloupe": CVL_DOM_OTHER,
    "Martinique": CVL_DOM_OTHER,
    "Guyane": CVL_DOM_OTHER,
    "La Reunion": CVL_DOM_OTHER,
    "Mayotte": CVL_DOM_OTHER,
    "Outre-mer (autre)": CVL_DOM_OTHER,
}


def normalize_department(value: object) -> str:
    """Canonicalise a department code, or return :data:`UNRESOLVED_DEPARTMENT`.

    Accepts the ``"2A/2B"`` literal and single-digit inputs such as ``"1"``,
    which appear when a department is read from a source other than a postcode.
    """
    text = str(value or "").strip().upper()
    if not text:
        return UNRESOLVED_DEPARTMENT
    if text in REGION_BY_DEPARTMENT:
        return text
    if len(text) == 1 and text.isdigit():
        padded = f"0{text}"
        if padded in REGION_BY_DEPARTMENT:
            return padded
    return UNRESOLVED_DEPARTMENT


def region_for_department(value: object) -> str:
    """Region name for a department code, or ``""`` when unresolved."""
    department = normalize_department(value)
    return REGION_BY_DEPARTMENT.get(department, "")


def department_group(value: object) -> str:
    """Sampling group for a department code.

    Unresolved departments fall in :data:`CVL_DOM_OTHER` rather than being
    excluded, so that episodes with weak buyer geography still have a positive
    inclusion probability and remain representable in the national estimate.
    """
    region = region_for_department(value)
    return GROUP_BY_REGION.get(region, CVL_DOM_OTHER)


def is_grand_ouest(value: object) -> bool:
    """Whether a department is in the frozen fourteen-department study scope."""
    return department_group(value) == GRAND_OUEST
