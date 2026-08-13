"""Declaration patterns, checked against strings taken verbatim from BOAMP.

Every example in this file is a real corpus fragment. The point of the suite is
not that the regexes compile but that they separate three things the corpus
mixes freely: a contract being re-procured, an asset being replaced, and a
contract's own extension option.
"""

from datetime import date

from boamp_pipeline.renewal_language import (
    EXCLUDED_ADVISORY_MISSION,
    EXCLUDED_ASSET_RENEWAL,
    EXCLUDED_MODIFICATION_BOILERPLATE,
    EXPIRY_DECLARATION,
    INCUMBENT_MENTION,
    POSITIVE_CLASSES,
    PRIOR_SPEND,
    RENEWAL_DECLARATION,
    RETENDER_AFTER_FAILURE,
    STAFF_TRANSFER,
    classify_renouvellement,
    declaration_strength,
    extract_dates,
    extract_references,
    find_matches,
    sentence_span,
)


def fired(text: str, field: str = "notice_text") -> set[str]:
    """Classes claimed by a text, excluding filtered matches."""
    return {m.klass for m in find_matches(text, field) if not m.is_excluded}


def patterns(text: str) -> set[str]:
    return {m.pattern for m in find_matches(text, "notice_text") if not m.is_excluded}


# ---------------------------------------------------------------------------
# The asset-renewal trap
# ---------------------------------------------------------------------------


def test_contract_renewal_is_detected() -> None:
    """Verbatim from notices 16-128525, 20-96583, 24-58736, 23-37182, 22-106500."""
    for text in [
        "Il s'agit d'un renouvellement de contrat portant sur la gestion et "
        "l'exploitation d'un equipement multi-accueil deja existant",
        "Il s'agit du renouvellement du contrat d'exploitation des installations thermiques",
        "Ce marche concerne le renouvellement du marche de maintenance et motorisation des portes",
        "Renouvellement du marche Peoplenet",
        "L'objet de l'accord-cadre porte sur le renouvellement du marche de tierce "
        "maintenance applicative",
        "Cette consultation est le renouvellement d'un marche precedemment attribue",
    ]:
        assert RENEWAL_DECLARATION in fired(text), text


def test_asset_renewal_is_excluded() -> None:
    """The bare word appears 28,475 times; ~90% is infrastructure, not contracts.

    Head-noun frequencies after "renouvellement de/du/des": reseau 4,844,
    canalisations 1,272, parc 995 -- against contrat 1,009 and marche 744.
    """
    for text in [
        "renouvellement des canalisations d'eau potable",
        "Renouvellement du reseau d'eclairage urbain",
        "renouvellement du parc de vehicules",
        "Travaux de renouvellement des equipements de la station",
        "renouvellement des branchements en plomb",
        "renouvellement des compteurs et des colonnes montantes",
    ]:
        assert RENEWAL_DECLARATION not in fired(text), text

    excluded = [m for m in find_matches("renouvellement du reseau", "objet") if m.is_excluded]
    assert excluded and excluded[0].excluded_reason == EXCLUDED_ASSET_RENEWAL


def test_asset_noun_between_trigger_and_contract_noun_blocks_the_match() -> None:
    """"renouvellement du reseau prevu au marche" is asset work, not re-procurement."""
    accepted, reason = classify_renouvellement("renouvellement du reseau prevu au marche", 15)

    assert not accepted
    assert reason == EXCLUDED_ASSET_RENEWAL


def test_contract_noun_in_a_later_sentence_does_not_bind() -> None:
    text = "renouvellement des canalisations. Le marche sera notifie en mai."

    assert RENEWAL_DECLARATION not in fired(text)


# ---------------------------------------------------------------------------
# Expiry declarations, the highest-value family
# ---------------------------------------------------------------------------


def test_expiry_declaration_with_its_stated_date() -> None:
    """Verbatim from 22-119302 and 24-53912. These name the predecessor's term,
    which is the strongest predecessor key available in the corpus."""
    text = (
        "L'ANGDM souhaite relancer un nouveau marche de TMA, le marche actuel "
        "arrivant a echeance le 2 mai 2023."
    )

    assert EXPIRY_DECLARATION in fired(text)
    expiry = [m for m in find_matches(text, "notice_text")
              if m.klass == EXPIRY_DECLARATION and not m.is_excluded][0]
    dates = extract_dates(text, near=expiry.end)
    assert dates[0].value == date(2023, 5, 2)
    assert dates[0].precision == "day"


def test_expiry_without_a_contract_noun_is_caught_by_the_relaunch_phrase() -> None:
    """Verbatim from 21-133174: the sentence never says "marche", but announcing
    a new consultation is itself the succession claim."""
    text = (
        "logiciel relatif a la gestion et la facturation des abonnes arrivant a "
        "echeance en Mars 2022, une nouvelle consultation est lancee."
    )

    assert EXPIRY_DECLARATION in fired(text)
    assert "echeance_avec_relance" in patterns(text)


def test_declared_date_precision_drives_matching_tolerance() -> None:
    assert extract_dates("arrive a echeance fin decembre 2023")[0] .precision == "month"
    assert extract_dates("arrive a echeance fin decembre 2023")[0].value == date(2023, 12, 31)
    assert extract_dates("arrive a echeance fin 2024")[0].value == date(2024, 12, 31)
    assert extract_dates("arrive a echeance fin 2024")[0].precision == "year"
    assert extract_dates("expire le 01/09/2025")[0].value == date(2025, 9, 1)
    assert extract_dates("echeance le 1er janvier 2023")[0].value == date(2023, 1, 1)


def test_a_date_far_from_the_trigger_is_not_bound_to_it() -> None:
    text = (
        "Le marche actuel arrive a echeance prochainement. " + "x" * 400
        + " Le 15 juillet 2024 se tiendra une reunion publique."
    )
    expiry = [m for m in find_matches(text, "notice_text")
              if m.klass == EXPIRY_DECLARATION and not m.is_excluded][0]

    assert extract_dates(text, near=expiry.end) == []


# ---------------------------------------------------------------------------
# Confirmed non-signals
# ---------------------------------------------------------------------------


def test_reconduction_is_not_a_succession_signal() -> None:
    """43,807 notices mention it, but only 129 in `objet`: it is the current
    contract's own extension option, verbatim from 15-37182 and 15-41772."""
    for text in [
        "Il est ensuite reconductible chaque annee a la date anniversaire par "
        "tacite reconduction sans que la duree totale du marche ne puisse exceder 4 ans.",
        "le marche est conclu pour une duree de 1 an a compter de sa notification, "
        "il est reconductible 3 fois par periode successive de 1 an",
        "Les reconductions se font sur decision expresse du Pouvoir Adjudicateur",
    ]:
        assert not (fired(text) & set(POSITIVE_CLASSES)), text


def test_modification_boilerplate_does_not_fire() -> None:
    """"marche en cours" is overwhelmingly the article R2194-1 amendment clause."""
    text = (
        "Les modifications du marche en cours d'execution sont possibles "
        "conformement a l'article R2194-1"
    )

    assert not (fired(text) & set(POSITIVE_CLASSES))
    excluded = [m for m in find_matches(text, "notice_text") if m.is_excluded]
    assert any(m.excluded_reason == EXCLUDED_MODIFICATION_BOILERPLATE for m in excluded)


def test_an_ordinary_duration_clause_is_silent() -> None:
    text = "La duree du marche est de 4 ans. Le titulaire sera designe apres analyse des offres."

    assert not (fired(text) & set(POSITIVE_CLASSES))


# ---------------------------------------------------------------------------
# Failure modes found by the first precision audit (20/60 correct)
#
# Each case below is a real notice that the unguarded patterns accepted. They
# are kept as regressions because every one of them would have entered the
# benchmark as a fabricated positive.
# ---------------------------------------------------------------------------


def test_advisory_missions_about_a_renewal_are_not_the_successor() -> None:
    """Largest false-positive source: 16 of 40. The renewal is real, but this
    notice is the adviser's own contract, not the contract that replaces."""
    for text in [
        "Mission d'assistance a maitrise d'ouvrage pour le renouvellement de la "
        "Delegation de Service Public Aeroport Le Mans",
        "Assistance a Maitrise d'Ouvrage dans le cadre du renouvellement du marche "
        "frais de sante et prevoyance",
        "Marche AMO technique, juridique et financiere pour le renouvellement du "
        "contrat de delegation de service public de transports urbains",
        "ATMO pour l'elaboration ou le renouvellement des marches de maintenance",
        "marche public d'assistance et de conseil dans le renouvellement des marches "
        "publics d'assurances",
        "Evaluation des actions et diagnostic en vue de la definition du projet de "
        "renouvellement de la convention",
        "contrôle de la concession et accompagnement de la procedure de renouvellement du contrat",
    ]:
        assert RENEWAL_DECLARATION not in fired(text, "objet"), text

    excluded = [
        m for m in find_matches("AMO pour le renouvellement du marche", "objet")
        if m.is_excluded
    ]
    assert excluded[0].excluded_reason == EXCLUDED_ADVISORY_MISSION


def test_a_contracts_own_end_date_is_not_a_predecessor_expiry() -> None:
    """Second largest source: 14 of 40. Without an incumbent marker these are
    just the notice stating its own term."""
    for text in [
        "L'accord-cadre prendra fin le 29 fevrier 2028",
        "L'accord-cadre debutera a compter de la date de notification du contrat et "
        "se terminera le 31 decembre 2025",
        "Le marche public prendra effet a compter de sa date de notification et prendra fin",
        "Le contrat est conclu a compter de sa date de notification et prend fin a "
        "l'achevement de la periode de garantie",
        "L'accord-cadre qui n'est pas reconductible se terminera le 31/12/2017",
    ]:
        assert EXPIRY_DECLARATION not in fired(text), text


def test_an_incumbent_marker_restores_the_expiry_declaration() -> None:
    """The same verb, with the contract marked as the incumbent one."""
    for text in [
        "Le marche en cours arrive a echeance le 14/07/2022",
        "Le marche actuel arrive a echeance le 16 janvier 2016",
        "Le contrat actuel arrivant a son terme le 31 decembre 2025",
        "compte-tenu du marche existant encore en cours de validite qui prend fin le 26/11/2019",
    ]:
        assert fired(text) & {EXPIRY_DECLARATION, INCUMBENT_MENTION}, text


def test_cemetery_concessions_are_not_public_service_delegations() -> None:
    """"Concession" is both a service delegation and a burial plot."""
    for text in [
        "Reprise de concessions funeraires echues",
        "TRAVAUX DE REPRISE DES CONCESSIONS ECHUES OU EN ETAT D'ABANDON",
        "la vente, le renouvellement ou la retrocession des concessions traditionnelles "
        "ou cineraires, la location aux familles des caveaux",
    ]:
        assert not (fired(text) & set(POSITIVE_CLASSES)), text


def test_renouvellement_used_for_a_tacit_option_is_excluded() -> None:
    """"Renouvellement" sometimes names the contract's own extension right."""
    for text in [
        "Le marche comporte une option au sens du droit communautaire: Renouvellement "
        "possible du marche deux fois un an de facon tacite",
        "Chaque renouvellement se fait pour une duree egale a celle du marche initial",
    ]:
        assert RENEWAL_DECLARATION not in fired(text), text


def test_works_and_remaining_asset_nouns_are_excluded() -> None:
    for text in [
        "LOT 1 : Renouvellement des escaliers mecaniques et maintenance associee",
        "Gestion du patrimoine existant et la realisation des travaux de renouvellement "
        "prevus par le contrat",
    ]:
        assert RENEWAL_DECLARATION not in fired(text), text


def test_payment_schedule_idiom_is_not_a_contract_expiry() -> None:
    """"mois echu" is an accounting term for a completed month."""
    text = "le paiement de la prestation s'effectuera mensuellement, mois echu"

    assert not (fired(text) & set(POSITIVE_CLASSES))


def test_an_application_for_a_concession_is_not_a_procurement_renewal() -> None:
    text = "la Commune a adresse a l'Etat une demande de renouvellement de la concession des plages"

    assert RENEWAL_DECLARATION not in fired(text)


# ---------------------------------------------------------------------------
# The contaminant class
# ---------------------------------------------------------------------------


def test_retender_after_failure_is_its_own_class() -> None:
    """Verbatim from 16-75229 and 24-139904. These reference a prior procedure
    of the wrong kind: one that never produced a contract."""
    for text in [
        "la procedure no 15-144845 publiee le 18 janvier 2016 etant declaree sans "
        "suite pour motifs d'interet general",
        "La presente consultation n 2024011 fait suite a la procedure n 2024007 qui "
        "ete declaree sans suite",
        "REFECTION DU CONDUIT DE CHEMINEE (relance du MP 2015-22 Infructueux)",
    ]:
        assert RETENDER_AFTER_FAILURE in fired(text), text
        assert RENEWAL_DECLARATION not in fired(text), text


def test_explicit_prior_reference_is_extracted() -> None:
    text = "La presente consultation fait suite a la procedure n 2024007 declaree sans suite"

    assert extract_references(text)[0][0] == "2024007"


# ---------------------------------------------------------------------------
# Weaker families and strength scoring
# ---------------------------------------------------------------------------


def test_prior_spend_and_staff_transfer_imply_a_predecessor() -> None:
    """Verbatim from 18-85658, 21-96612 and 19-22521."""
    spend = ("les depenses du present lot pour la duree totale du precedent marche "
             "se sont elevees a 233 000 euros")
    staff = ("le titulaire du marche devra proceder a la reprise du personnel en poste "
             "dans le cadre du precedent marche")

    assert PRIOR_SPEND in fired(spend)
    assert STAFF_TRANSFER in fired(staff)
    assert INCUMBENT_MENTION in fired(staff)


def test_declaration_strength_counts_distinct_families() -> None:
    """A notice asserting a predecessor two independent ways is better evidence."""
    weak = find_matches("le titulaire actuel assure la prestation", "notice_text")
    strong = find_matches(
        "Le marche actuel arrive a echeance le 2 mai 2023. Il s'agit du "
        "renouvellement du contrat d'exploitation.",
        "notice_text",
    )

    assert declaration_strength(weak) == 1
    assert declaration_strength(strong) >= 2


def test_offsets_are_exact_substrings_of_the_source() -> None:
    """The annotation layer verifies every quote against its field, so matching
    must run on raw text and preserve offsets."""
    text = "Il s'agit du renouvellement du marche de nettoyage des locaux."

    for match in find_matches(text, "objet"):
        assert text[match.start:match.end] == match.matched_text
        assert match.snippet in text


def test_sentence_span_bounds_a_position() -> None:
    text = "Premiere phrase. Deuxieme phrase ici. Troisieme."
    start, end = sentence_span(text, text.index("Deuxieme"))

    assert text[start:end].strip() == "Deuxieme phrase ici"
