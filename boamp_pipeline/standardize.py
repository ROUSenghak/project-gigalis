"""Schema-aware BOAMP notice standardization.

The functions in this module deliberately favor unresolved values over generic
JSON guesses. Downstream linkage must not treat an unrelated identifier, date,
or amount as contracting-buyer or contract evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Any, Iterable


PARSER_VERSION = "boamp_standardizer_v2.2"

GRAND_OUEST_REGION_BY_DEPARTMENT = {
    "14": "Normandie",
    "22": "Bretagne",
    "27": "Normandie",
    "29": "Bretagne",
    "35": "Bretagne",
    "44": "Pays de la Loire",
    "49": "Pays de la Loire",
    "50": "Normandie",
    "53": "Pays de la Loire",
    "56": "Bretagne",
    "61": "Normandie",
    "72": "Pays de la Loire",
    "76": "Normandie",
    "85": "Pays de la Loire",
}

RAW_FIELD_NAMES = [
    "idweb", "id", "contractfolderid", "objet", "filename", "famille",
    "code_departement", "code_departement_prestation", "famille_libelle",
    "dateparution", "datefindiffusion", "datelimitereponse", "nomacheteur",
    "titulaire", "perimetre", "type_procedure", "soustype_procedure",
    "procedure_libelle", "procedure_categorise", "nature", "sousnature",
    "nature_libelle", "sousnature_libelle", "nature_categorise",
    "nature_categorise_libelle", "criteres", "marche_public_simplifie",
    "marche_public_simplifie_label", "etat", "descripteur_code", "dc",
    "descripteur_libelle", "type_marche", "type_marche_facette", "type_avis",
    "annonce_lie", "annonces_anterieures", "source_schema", "gestion",
    "donnees", "url_avis",
]

EMPTY_STRINGS = {"", "none", "null", "nan", "<na>", "[]", "{}"}
CPV_RE = re.compile(r"(?<!\d)(\d{8})(?:-\d)?(?!\d)")
IDWEB_RE = re.compile(r"(?<!\d)(\d{2}-\d+)(?!\d)")
NUMBER_RE = re.compile(r"-?\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?")
REFERENCE_NOISE_RE = re.compile(
    r"^(?:sans objet|neant|n/?a|non renseigne|aucun|fra|dce|voir(?: dce)?|cf\.?|"
    r"org\s*\d+|tpo\s*\d+|default[_ ]value[_ ]change[_ ]me|"
    r"documents?(?: de| du)? march[eé]|https?\b.*|www\b.*)$",
    re.I,
)


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict)):
        return not value
    return str(value).strip().casefold() in EMPTY_STRINGS


def text_value(value: Any) -> str:
    if is_empty(value):
        return ""
    if isinstance(value, dict):
        if "#text" in value:
            return text_value(value["#text"])
        return ""
    if isinstance(value, list):
        return " ".join(filter(None, (text_value(item) for item in value)))
    return str(value).strip()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def get_path(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def walk_json(value: Any, path: str = "") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            child_path = f"{path}.{key}" if path else key
            yield child_path, key, nested
            yield from walk_json(nested, child_path)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_json(nested, f"{path}[]")


def normalize_name(value: Any) -> str:
    """Apply formatting-only normalization without removing legal terms."""
    if is_empty(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_reference(value: Any) -> str:
    text = normalize_name(value)
    if len(text) < 3 or REFERENCE_NOISE_RE.match(text):
        return ""
    return text


def normalize_document_text(value: Any) -> str:
    if is_empty(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def parse_json_cell(value: Any) -> Any:
    if is_empty(value):
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}


def parse_date(value: Any) -> date | None:
    text = text_value(value)
    if not text:
        return None
    match = re.search(r"(20\d{2})[-/]([01]\d)[-/]([0-3]\d)", text)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def parse_datetime(value: Any) -> datetime | None:
    text = text_value(value)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError:
        parsed_date = parse_date(text)
        return datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc) if parsed_date else None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def digits(value: Any) -> str:
    return re.sub(r"\D", "", text_value(value))


def luhn_valid(value: str) -> bool:
    if not value.isdigit() or len(value) not in {9, 14}:
        return False
    total = 0
    parity = len(value) % 2
    for index, character in enumerate(value):
        number = int(character)
        if index % 2 == parity:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def validated_buyer_identifier(value: Any) -> tuple[str, str, str]:
    candidate = digits(value)
    if len(candidate) == 14 and luhn_valid(candidate):
        return candidate, candidate[:9], "validated_siret"
    if len(candidate) == 9 and luhn_valid(candidate):
        return "", candidate, "validated_siren"
    return "", "", ""


def department_from_postcode(value: Any) -> str:
    postcode = digits(value)
    if len(postcode) != 5:
        return ""
    if postcode.startswith(("971", "972", "973", "974", "976")):
        return postcode[:3]
    if postcode.startswith("20"):
        return "2A/2B"
    return postcode[:2]


def _organization_id(organization: dict[str, Any]) -> str:
    companies = [item for item in as_list(organization.get("efac:Company")) if isinstance(item, dict)]
    return next(
        (text_value(get_path(company, "cac:PartyIdentification", "cbc:ID")) for company in companies
         if text_value(get_path(company, "cac:PartyIdentification", "cbc:ID"))),
        "",
    )


def _eforms_notice(root: dict[str, Any]) -> dict[str, Any] | None:
    eforms = root.get("EFORMS")
    if not isinstance(eforms, dict):
        return None
    return next((value for value in eforms.values() if isinstance(value, dict)), None)


def _find_key(value: Any, target_key: str) -> Any:
    if isinstance(value, dict):
        if target_key in value:
            return value[target_key]
        for nested in value.values():
            found = _find_key(nested, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_key(nested, target_key)
            if found is not None:
                return found
    return None


def eforms_contracting_organizations(root: dict[str, Any]) -> list[dict[str, Any]]:
    notice = _eforms_notice(root)
    if not notice:
        return []
    contracting_ids: list[str] = []
    for contracting_party in as_list(notice.get("cac:ContractingParty")):
        identifier = text_value(
            get_path(contracting_party, "cac:Party", "cac:PartyIdentification", "cbc:ID")
        )
        if identifier and identifier not in contracting_ids:
            contracting_ids.append(identifier)

    organizations = as_list(_find_key(notice.get("ext:UBLExtensions", {}), "efac:Organization"))
    by_id = {_organization_id(item): item for item in organizations if isinstance(item, dict)}
    return [by_id[identifier] for identifier in contracting_ids if identifier in by_id]


def extract_buyer(root: dict[str, Any], fallback_name: Any) -> dict[str, Any]:
    name = text_value(fallback_name)
    identifier_value = ""
    postcode = ""
    city = ""
    id_source = ""
    geography_source = ""
    identifier_candidates: list[str] = []
    postcode_candidates: list[str] = []
    multiple_contracting_buyers = False

    eforms_organizations = eforms_contracting_organizations(root)
    if eforms_organizations:
        cities: list[str] = []
        for organization in eforms_organizations:
            companies = [item for item in as_list(organization.get("efac:Company")) if isinstance(item, dict)]
            for company in companies:
                legal_entities = [item for item in as_list(company.get("cac:PartyLegalEntity")) if isinstance(item, dict)]
                postal_addresses = [item for item in as_list(company.get("cac:PostalAddress")) if isinstance(item, dict)]
                for legal in legal_entities:
                    candidate_id = text_value(legal.get("cbc:CompanyID"))
                    if candidate_id and candidate_id not in identifier_candidates:
                        identifier_candidates.append(candidate_id)
                for postal in postal_addresses:
                    candidate_postcode = text_value(postal.get("cbc:PostalZone"))
                    candidate_city = text_value(postal.get("cbc:CityName"))
                    if candidate_postcode and candidate_postcode not in postcode_candidates:
                        postcode_candidates.append(candidate_postcode)
                    if candidate_city and candidate_city not in cities:
                        cities.append(candidate_city)
        validated_ids = [candidate for candidate in identifier_candidates if validated_buyer_identifier(candidate)[2]]
        distinct_sirens = {validated_buyer_identifier(candidate)[1] for candidate in validated_ids}
        multiple_contracting_buyers = len(distinct_sirens) > 1
        identifier_value = validated_ids[0] if len(distinct_sirens) == 1 else ""
        departments = {department_from_postcode(candidate) for candidate in postcode_candidates}
        departments.discard("")
        if len(departments) == 1:
            postcode = postcode_candidates[0]
            city = cities[0] if len(cities) == 1 else ""
        id_source = "eforms_contracting_party_organization"
        geography_source = "eforms_contracting_party_postal_address"
    elif isinstance(root.get("IDENTITE"), dict):
        identity = root["IDENTITE"]
        name = text_value(identity.get("DENOMINATION")) or name
        identifier_value = text_value(identity.get("CODE_IDENT_NATIONAL"))
        if not identifier_value:
            identifier_value = text_value(get_path(identity, "TYPE_IDENT_NATIONAL", "SIRET"))
        if not identifier_value:
            identifier_value = text_value(get_path(identity, "TYPE_IDENT_NATIONAL", "SIREN"))
        postcode = text_value(identity.get("CP"))
        city = text_value(identity.get("VILLE"))
        id_source = "IDENTITE.CODE_IDENT_NATIONAL"
        geography_source = "IDENTITE.CP"
    elif isinstance(get_path(root, "FNSimple", "organisme"), dict):
        organization = get_path(root, "FNSimple", "organisme")
        name = text_value(organization.get("nomOfficiel")) or name
        identifier_value = text_value(organization.get("codeIdentificationNational"))
        postcode = text_value(organization.get("cp"))
        city = text_value(organization.get("ville"))
        id_source = "FNSimple.organisme.codeIdentificationNational"
        geography_source = "FNSimple.organisme.cp"
    else:
        for root_name in ("MAPA", "DSP", "DIVERS"):
            organization = get_path(root, root_name, "organisme")
            if not isinstance(organization, dict):
                continue
            address = organization.get("adr", {})
            name = text_value(organization.get("nom")) or name
            identifier_value = text_value(organization.get("codeIdentificationNational"))
            postcode = text_value(address.get("cp")) if isinstance(address, dict) else ""
            city = text_value(address.get("ville")) if isinstance(address, dict) else ""
            id_source = f"{root_name}.organisme.codeIdentificationNational"
            geography_source = f"{root_name}.organisme.adr.cp"
            break

    if identifier_value and identifier_value not in identifier_candidates:
        identifier_candidates.append(identifier_value)
    if postcode and postcode not in postcode_candidates:
        postcode_candidates.append(postcode)

    siret, siren, id_quality = validated_buyer_identifier(identifier_value)
    if not id_quality:
        if multiple_contracting_buyers:
            id_source = "multiple_contracting_buyers"
        else:
            id_source = "unresolved" if not identifier_value else f"{id_source}:invalid"
    normalized_name = normalize_name(name)
    buyer_id = siren
    buyer_key = f"siren:{siren}" if siren else (f"name:{normalized_name}" if normalized_name else "")
    buyer_id_source = id_source if siren else ("normalized_name" if normalized_name else "unresolved")
    buyer_id_quality = (
        "multiple_contracting_buyers" if multiple_contracting_buyers
        else id_quality if siren
        else "name_only" if normalized_name
        else "unresolved"
    )

    department = department_from_postcode(postcode)
    region = GRAND_OUEST_REGION_BY_DEPARTMENT.get(department, "")
    geography_quality = "structured_postcode" if department else "unresolved"
    if not department:
        geography_source = "unresolved"

    return {
        "buyer_name_raw": name,
        "buyer_name_normalized": normalized_name,
        "buyer_siret": siret,
        "buyer_siren": siren,
        "buyer_id": buyer_id,
        "buyer_id_source": buyer_id_source,
        "buyer_id_quality": buyer_id_quality,
        "buyer_key": buyer_key,
        "buyer_identifier_candidates_json": json_compact(identifier_candidates),
        "buyer_multi_organization_flag": multiple_contracting_buyers,
        "buyer_postcode": postcode,
        "buyer_postcode_candidates_json": json_compact(postcode_candidates),
        "buyer_city": city,
        "buyer_department": department,
        "buyer_region": region,
        "geography_source": geography_source,
        "geography_quality": geography_quality,
        "grand_ouest_flag": bool(region),
    }


def extract_linked_notice_ids(record: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for field in ("annonce_lie", "annonces_anterieures"):
        value = record.get(field)
        parsed = parse_json_cell(value)
        candidates = IDWEB_RE.findall(json.dumps(parsed, ensure_ascii=False))
        if not candidates:
            candidates = IDWEB_RE.findall(text_value(value))
        for candidate in candidates:
            if candidate not in found:
                found.append(candidate)
    return found


def extract_cpv(root: dict[str, Any], record: dict[str, Any]) -> tuple[str, list[str]]:
    found: list[str] = []
    preferred: list[str] = []
    for path, key, value in walk_json(root):
        upper_path = path.upper()
        upper_key = key.upper()
        eforms_cpv = (
            upper_key == "CBC:ITEMCLASSIFICATIONCODE"
            and isinstance(value, dict)
            and text_value(value.get("@listName")).casefold() == "cpv"
        )
        if "CPV" not in upper_path and "COMMONPROCUREMENTVOCABULARY" not in upper_path and not eforms_cpv:
            continue
        codes = CPV_RE.findall(json.dumps(value, ensure_ascii=False))
        if upper_key in {"PRINCIPAL", "CLASSPRINCIPALE", "MAINCLASSIFICATIONCODE"} or "MAINCOMMODITYCLASSIFICATION" in upper_path:
            preferred.extend(codes)
        for code in codes:
            if code not in found:
                found.append(code)
    for field in ("descripteur_code", "dc"):
        for code in CPV_RE.findall(text_value(record.get(field))):
            if code not in found:
                found.append(code)
    main_cpv = next((code for code in preferred if code in found), found[0] if found else "")
    return main_cpv, found


def extract_text_parts(root: dict[str, Any], record: dict[str, Any]) -> tuple[list[str], list[str], str]:
    titles: list[str] = []
    descriptions: list[str] = []
    object_text = normalize_document_text(record.get("objet"))
    if object_text:
        titles.append(object_text)
    for path, key, value in walk_json(root):
        upper = key.upper()
        if upper in {"INTITULE", "TITLE", "CBC:NAME"}:
            target = titles
        elif upper in {"DESCRIPTION", "CBC:DESCRIPTION", "OBJETCOMPLET"}:
            target = descriptions
        else:
            continue
        candidate = normalize_document_text(text_value(value))
        if candidate and candidate not in target:
            target.append(candidate)
    parts = []
    for candidate in [*titles, *descriptions]:
        if candidate not in parts:
            parts.append(candidate)
    return titles, descriptions, "\n".join(parts)


def extract_procedure_reference(root: dict[str, Any]) -> tuple[str, str]:
    preferred_paths = (
        ("OBJET", "REF_MARCHE"),
        ("FNSimple", "initial", "numeroReference"),
        ("MAPA", "initial", "numeroReference"),
        ("DSP", "initial", "descriptionMarche", "numeroReference"),
        ("DIVERS", "initial", "description", "numeroReference"),
    )
    for path in preferred_paths:
        reference = normalize_reference(get_path(root, *path))
        if reference:
            return reference, ".".join(path)

    notice = _eforms_notice(root)
    if notice:
        reference = normalize_reference(notice.get("cbc:ContractFolderID"))
        if reference:
            return reference, "eforms:cbc:ContractFolderID"
    return "", "unresolved"


def _numeric(value: Any) -> float | None:
    text = text_value(value).replace("\u00a0", " ")
    match = NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def _duration_from_measure(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    amount = _numeric(value.get("#text"))
    if amount is None:
        return None
    unit = text_value(value.get("@unitCode")).upper()
    factors = {"DAY": 1 / 30.4375, "DAYS": 1 / 30.4375, "MON": 1.0, "MONTH": 1.0, "YEAR": 12.0, "ANN": 12.0}
    return amount * factors[unit] if unit in factors else None


def extract_duration(root: dict[str, Any]) -> dict[str, Any]:
    candidates: list[tuple[float, str, str]] = []
    start_dates: list[date] = []
    end_dates: list[date] = []

    duration_delay = get_path(root, "OBJET", "DUREE_DELAI")
    if isinstance(duration_delay, dict):
        months = _numeric(duration_delay.get("DUREE_MOIS"))
        days = _numeric(duration_delay.get("DUREE_JOURS"))
        if months is not None:
            candidates.append((months, "OBJET.DUREE_DELAI.DUREE_MOIS", text_value(duration_delay.get("DUREE_MOIS"))))
        elif days is not None:
            candidates.append((days / 30.4375, "OBJET.DUREE_DELAI.DUREE_JOURS", text_value(duration_delay.get("DUREE_JOURS"))))
        start = parse_date(duration_delay.get("DATE_DEBUT"))
        end = parse_date(duration_delay.get("DATE_FIN"))
        if start:
            start_dates.append(start)
        if end:
            end_dates.append(end)

    for path in (
        ("FNSimple", "initial", "natureMarche", "dureeMois"),
        ("MAPA", "initial", "duree"),
    ):
        months = _numeric(get_path(root, *path))
        if months is not None:
            candidates.append((months, ".".join(path), text_value(get_path(root, *path))))

    notice = _eforms_notice(root)
    if notice:
        eforms_months: list[float] = []
        for path, key, value in walk_json(notice):
            if "PLANNEDPERIOD" not in path.upper():
                continue
            if key == "cbc:DurationMeasure":
                months = _duration_from_measure(value)
                if months is not None:
                    eforms_months.append(months)
            elif key == "cbc:StartDate":
                parsed = parse_date(value)
                if parsed:
                    start_dates.append(parsed)
            elif key == "cbc:EndDate":
                parsed = parse_date(value)
                if parsed:
                    end_dates.append(parsed)
        rounded = {round(value, 3) for value in eforms_months if 0 < value <= 240}
        if len(rounded) == 1:
            candidates.append((rounded.pop(), "eforms.PlannedPeriod.DurationMeasure:consensus", ""))

    valid_candidates = [item for item in candidates if 0 < item[0] <= 240]
    start = min(start_dates) if start_dates else None
    end = max(end_dates) if end_dates else None
    if not valid_candidates and start and end and end >= start:
        valid_candidates.append(((end - start).days / 30.4375, "typed_start_end_dates", f"{start}/{end}"))

    if not valid_candidates:
        return {
            "duration_raw": "", "duration_months": None, "duration_source": "unresolved",
            "duration_quality": "unresolved", "contract_start_date": start,
            "contract_end_date": end,
        }

    values = {round(item[0], 1) for item in valid_candidates}
    if len(values) > 1:
        return {
            "duration_raw": json.dumps([item[2] for item in valid_candidates], ensure_ascii=False),
            "duration_months": None, "duration_source": "conflicting_typed_values",
            "duration_quality": "conflict", "contract_start_date": start,
            "contract_end_date": end,
        }
    months, source, raw = valid_candidates[0]
    return {
        "duration_raw": raw, "duration_months": round(months, 2), "duration_source": source,
        "duration_quality": "typed", "contract_start_date": start,
        "contract_end_date": end,
    }


def extract_amounts(root: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path, key, value in walk_json(root):
        upper = key.upper()
        amount_type = ""
        if "ESTIMATION" in upper or "ESTIMATED" in upper:
            amount_type = "estimated"
        elif "VALEUR_MAX" in upper or "MAXIMUM" in upper:
            amount_type = "maximum"
        elif "VALEUR_MIN" in upper or "MINIMUM" in upper:
            amount_type = "minimum"
        elif upper in {"MONTANT", "VALEUR", "VALEUR_TOTALE", "CBC:VALUATIONAMOUNT", "CBC:TAXEXCLUSIVEAMOUNT"}:
            amount_type = "declared_or_awarded"
        if not amount_type:
            continue
        amount = _numeric(value)
        if amount is None or amount < 0 or amount > 100_000_000_000:
            continue
        currency = ""
        if isinstance(value, dict):
            currency = text_value(value.get("@currencyID")) or text_value(value.get("DEVISE"))
        candidate = {"value": amount, "type": amount_type, "currency": currency, "source": path}
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= 50:
            break
    return candidates


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def preserve_raw_value(value: Any) -> str:
    """Preserve nested source values as valid JSON and scalar values as text."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json_compact(value)
    return str(value)


def standardize_record(record: dict[str, Any], source_file: str) -> dict[str, Any]:
    raw = {field: preserve_raw_value(record.get(field)) for field in RAW_FIELD_NAMES}
    root = parse_json_cell(record.get("donnees"))
    root = root if isinstance(root, dict) else {}
    buyer = extract_buyer(root, record.get("nomacheteur"))
    main_cpv, all_cpvs = extract_cpv(root, record)
    titles, descriptions, notice_text = extract_text_parts(root, record)
    procedure_reference, procedure_reference_source = extract_procedure_reference(root)
    duration = extract_duration(root)
    amounts = extract_amounts(root)
    linked_ids = extract_linked_notice_ids(record)

    publication_date = parse_date(record.get("dateparution"))
    diffusion_end_date = parse_date(record.get("datefindiffusion"))
    response_deadline = parse_datetime(record.get("datelimitereponse"))
    deadline_before_publication = bool(
        publication_date and response_deadline and response_deadline.date() < publication_date
    )
    end_before_start = bool(
        duration["contract_start_date"] and duration["contract_end_date"]
        and duration["contract_end_date"] < duration["contract_start_date"]
    )
    date_outside_study = bool(
        publication_date and not (date(2015, 1, 1) <= publication_date < date(2026, 1, 1))
    )

    divisions = sorted({code[:2] for code in all_cpvs})
    groups = sorted({code[:3] for code in all_cpvs})
    classes = sorted({code[:4] for code in all_cpvs})
    categories = sorted({code[:5] for code in all_cpvs})
    framework_text = normalize_name(" ".join([text_value(record.get("objet")), notice_text]))
    framework_flag = bool(re.search(r"\baccord cadre\b|\bframework agreement\b", framework_text))
    serialized_record = json_compact(record)

    return {
        **raw,
        **buyer,
        "publication_date": publication_date,
        "diffusion_end_date": diffusion_end_date,
        "response_deadline": response_deadline,
        "deadline_before_publication": deadline_before_publication,
        "contract_end_before_start": end_before_start,
        "date_outside_study": date_outside_study,
        "main_cpv": main_cpv,
        "all_cpvs_json": json_compact(all_cpvs),
        "cpv_divisions_json": json_compact(divisions),
        "cpv_groups_json": json_compact(groups),
        "cpv_classes_json": json_compact(classes),
        "cpv_categories_json": json_compact(categories),
        "has_any_cpv": bool(all_cpvs),
        "notice_titles_json": json_compact(titles),
        "notice_descriptions_json": json_compact(descriptions),
        "notice_text": notice_text,
        "has_episode_text": bool(notice_text),
        "procedure_reference": procedure_reference,
        "procedure_reference_source": procedure_reference_source,
        "linked_notice_ids_json": json_compact(linked_ids),
        "framework_flag": framework_flag,
        **duration,
        "amount_candidates_json": json_compact(amounts),
        "amount_candidate_count": len(amounts),
        "parser_version": PARSER_VERSION,
        "raw_source_file": source_file,
        "raw_record_sha256": hashlib.sha256(serialized_record.encode("utf-8")).hexdigest(),
    }
