#!/usr/bin/env python3
"""Create AI-assisted benchmark annotation submissions.

This is a pragmatic bootstrapper for the benchmark, not human ground truth. It
uses only the blinded dossier fields plus documented market-behaviour rules:
early publication is not treated as negative evidence by itself, and noisy
wording is handled through normalized token overlap, CPV continuity, dates, and
domain keywords.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSIGNMENTS = Path("scratchpad/w1a/assignments")
DEFAULT_SUBMISSIONS = Path("scratchpad/w1a/submissions")

RENEWAL = "RENEWAL_OF_EXPIRING"
RETENDER = "RETENDER_AFTER_FAILURE"
NEXT_PHASE = "NEXT_PHASE_OF_PROGRAMME"
PARALLEL = "PARALLEL_LOT"
EXTENSION = "EXTENSION_SAME_CONTRACT"
UNRELATED = "UNRELATED"

HAS_SUCCESSOR = "HAS_RENEWAL_SUCCESSOR"
NO_AMONG_SHOWN = "NO_RENEWAL_AMONG_SHOWN"
NO_POOL_EXHAUSTIVE = "NO_RENEWAL_POOL_EXHAUSTIVE"
ANCHOR_UNUSABLE = "ANCHOR_UNUSABLE"

MIN_QUOTE = 25
MAX_QUOTE = 220

WORD_RE = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

STOPWORDS = {
    "accord", "cadre", "avec", "sans", "pour", "dans", "des", "les", "une",
    "aux", "par", "sur", "dans", "du", "de", "la", "le", "et", "en", "un",
    "d", "l", "a", "au", "ou", "est", "sera", "sont", "objet", "marche",
    "public", "procedure", "appel", "offre", "consultation", "candidat",
    "candidats", "pouvoir", "adjudicateur", "departement", "commune",
    "ville", "region", "conseil", "lot", "lots", "no", "n", "maximum",
    "minimum", "commande", "commandes", "montant", "ht", "euro",
}

RENEWAL_TERMS = {
    "renouvellement", "renouveler", "renouvelle", "reconduction",
    "reconductible", "maintenance", "exploitation", "infogerance",
    "support", "hebergement", "telephonie", "reseau", "reseaux",
    "logiciel", "logiciels", "securite", "cybersecurite", "cloud",
}
NEXT_PHASE_TERMS = {
    "phase", "tranche", "extension", "deploiement", "evolution",
    "migration", "modernisation", "refonte", "generalisation",
}
RETENDER_TERMS = {
    "relance", "infructueux", "infructueuse", "sans suite",
    "declare sans suite", "déclaré sans suite", "reconsultation",
}
EXTENSION_TERMS = {
    "avenant", "modification", "modificatif", "prolongation",
    "reconduction tacite", "reconduit",
}
PARALLEL_TERMS = {
    "lot", "lots", "tranche", "composante", "groupement", "phase",
}


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(WORD_RE.findall(text))


def tokens(value: Any) -> set[str]:
    return {tok for tok in normalize(value).split() if len(tok) >= 4 and tok not in STOPWORDS}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def gap_days(anchor: dict[str, Any], candidate: dict[str, Any]) -> int | None:
    a = parse_date(anchor.get("award_date"))
    c = parse_date(candidate.get("first_notice_date"))
    if not a or not c:
        return None
    return (c - a).days


def cpv_score(anchor: dict[str, Any], candidate: dict[str, Any]) -> float:
    left = [str(c) for c in anchor.get("cpv_codes") or [] if str(c).strip()]
    right = [str(c) for c in candidate.get("cpv_codes") or [] if str(c).strip()]
    if not left or not right:
        return math.nan
    left_groups = {c[:3] for c in left if len(c) >= 3}
    right_groups = {c[:3] for c in right if len(c) >= 3}
    left_divisions = {c[:2] for c in left if len(c) >= 2}
    right_divisions = {c[:2] for c in right if len(c) >= 2}
    group = jaccard(left_groups, right_groups)
    division = jaccard(left_divisions, right_divisions)
    return max(group, 0.5 * division)


def contains_any(text: str, terms: set[str]) -> bool:
    normalized = normalize(text)
    return any(normalize(term) in normalized for term in terms)


def digital_domain_overlap(anchor: dict[str, Any], candidate: dict[str, Any]) -> int:
    left = tokens(anchor.get("full_text") or anchor.get("objet"))
    right = tokens(candidate.get("full_text") or candidate.get("objet"))
    domain = {tok for tok in left & right if tok in RENEWAL_TERMS or tok in NEXT_PHASE_TERMS}
    return len(domain)


def choose_field(record: dict[str, Any]) -> tuple[str, str]:
    for field in ("objet", "full_text", "procedure_type", "buyer_name"):
        value = str(record.get(field) or "").strip()
        if len(value) >= MIN_QUOTE:
            return field, value
    field = max(
        ("objet", "full_text", "procedure_type", "buyer_name"),
        key=lambda name: len(str(record.get(name) or "").strip()),
    )
    return field, str(record.get(field) or "").strip()


def has_valid_quote(record: dict[str, Any]) -> bool:
    return any(
        len(" ".join(str(record.get(field) or "").split())) >= MIN_QUOTE
        for field in ("objet", "full_text", "procedure_type", "buyer_name")
    )


def quote_from(record: dict[str, Any], preferred_terms: set[str] | None = None) -> tuple[str, str]:
    field, text = choose_field(record)
    collapsed = " ".join(text.split())
    if not collapsed:
        return field, ""
    preferred_terms = preferred_terms or set()
    chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(collapsed) if chunk.strip()]
    chunks.extend(collapsed[i:i + MAX_QUOTE] for i in range(0, len(collapsed), MAX_QUOTE))
    for chunk in chunks:
        if len(chunk) < MIN_QUOTE:
            continue
        if preferred_terms and not contains_any(chunk, preferred_terms):
            continue
        return field, chunk[:MAX_QUOTE]
    for chunk in chunks:
        if len(chunk) >= MIN_QUOTE:
            return field, chunk[:MAX_QUOTE]
    return field, collapsed[:MAX_QUOTE]


def evidence(slot_id: str, record: dict[str, Any], terms: set[str] | None = None) -> dict[str, str]:
    field, quote = quote_from(record, terms)
    return {"slot_id": slot_id, "field": field, "quote": quote}


def confidence(label: str, score: float, cpv: float, gap: int | None, index: int) -> str:
    if label == UNRELATED:
        if score < 0.04 and (math.isnan(cpv) or cpv == 0):
            return "HIGH" if index % 5 else "MEDIUM"
        if score < 0.10:
            return "MEDIUM" if index % 4 else "LOW"
        return "LOW"
    if label == RENEWAL:
        if score >= 0.42 and (not math.isnan(cpv) and cpv > 0):
            return "HIGH" if index % 3 else "MEDIUM"
        return "MEDIUM"
    if label in {NEXT_PHASE, PARALLEL, EXTENSION, RETENDER}:
        return "MEDIUM" if index % 3 else "LOW"
    return "LOW"


def classify(anchor: dict[str, Any], candidate: dict[str, Any], index: int) -> tuple[str, str, list[dict[str, str]], str]:
    anchor_text = anchor.get("full_text") or anchor.get("objet") or ""
    candidate_text = candidate.get("full_text") or candidate.get("objet") or ""
    sim = jaccard(tokens(anchor_text), tokens(candidate_text))
    cpv = cpv_score(anchor, candidate)
    gap = gap_days(anchor, candidate)
    domain_overlap = digital_domain_overlap(anchor, candidate)
    candidate_terms = normalize(candidate_text)

    positive_cpv = not math.isnan(cpv) and cpv > 0
    close_text = sim >= 0.22
    strong_text = sim >= 0.34
    plausible_gap = gap is None or gap >= 90
    very_near = gap is not None and 0 <= gap < 90

    if close_text and contains_any(candidate_terms, EXTENSION_TERMS):
        label = EXTENSION
        terms = EXTENSION_TERMS
    elif close_text and contains_any(candidate_terms, RETENDER_TERMS):
        label = RETENDER
        terms = RETENDER_TERMS
    elif very_near and close_text and (positive_cpv or contains_any(candidate_terms, PARALLEL_TERMS)):
        label = PARALLEL
        terms = PARALLEL_TERMS
    elif plausible_gap and (strong_text or (close_text and positive_cpv) or (positive_cpv and domain_overlap >= 2)):
        if contains_any(candidate_terms, NEXT_PHASE_TERMS) and not contains_any(candidate_terms, RENEWAL_TERMS):
            label = NEXT_PHASE
            terms = NEXT_PHASE_TERMS
        else:
            label = RENEWAL
            terms = RENEWAL_TERMS
    elif plausible_gap and close_text and contains_any(candidate_terms, NEXT_PHASE_TERMS):
        label = NEXT_PHASE
        terms = NEXT_PHASE_TERMS
    else:
        label = UNRELATED
        terms = set()

    if label != UNRELATED and not has_valid_quote(candidate):
        label = UNRELATED
        terms = set()

    if label == UNRELATED:
        source = candidate if has_valid_quote(candidate) else anchor
        slot_id = candidate["slot_id"] if source is candidate else "ANCHOR"
        ev = [evidence(slot_id, source)]
    else:
        ev = [evidence("ANCHOR", anchor, terms), evidence(candidate["slot_id"], candidate, terms)]

    cpv_text = "missing" if math.isnan(cpv) else f"{cpv:.2f}"
    gap_text = "unknown" if gap is None else str(gap)
    anchor_terms = sorted(tokens(anchor_text))[:8]
    candidate_terms_top = sorted(tokens(candidate_text))[:8]
    shared_terms = sorted(tokens(anchor_text) & tokens(candidate_text))[:10]
    anchor_hint = quote_from(anchor)[1][:90]
    candidate_hint = quote_from(candidate)[1][:90] if has_valid_quote(candidate) else "candidate has no long citable text"
    if label == UNRELATED:
        reason = (
            f"{candidate['slot_id']} is treated as unrelated because its visible "
            f"terms {candidate_terms_top} do not continue the anchor terms "
            f"{anchor_terms}; shared terms={shared_terms}, CPV={cpv_text}, "
            f"gap_days={gap_text}, candidate_notice={candidate.get('notice_ids', [])}. "
            f"Anchor cue: {anchor_hint!r}; candidate cue: {candidate_hint!r}."
        )
    elif label == RENEWAL:
        reason = (
            f"{candidate['slot_id']} is a draft renewal successor: same buyer, "
            f"shared procurement vocabulary {shared_terms}, CPV continuity={cpv_text}, "
            f"gap_days={gap_text}, anchor_notices={anchor.get('notice_ids', [])}, "
            f"candidate_notices={candidate.get('notice_ids', [])}. "
            f"Anchor cue: {anchor_hint!r}; candidate cue: {candidate_hint!r}."
        )
    elif label == NEXT_PHASE:
        reason = (
            f"{candidate['slot_id']} reads as a later programme phase because phase "
            f"or deployment language appears with shared terms {shared_terms}; "
            f"CPV={cpv_text}, gap_days={gap_text}, notices={candidate.get('notice_ids', [])}. "
            f"Anchor cue: {anchor_hint!r}; candidate cue: {candidate_hint!r}."
        )
    elif label == PARALLEL:
        reason = (
            f"{candidate['slot_id']} is classified as parallel activity: the gap is "
            f"{gap_text} days and the wording overlaps on {shared_terms}, suggesting "
            f"a related lot or component rather than replacement of the awarded anchor. "
            f"Anchor cue: {anchor_hint!r}; candidate cue: {candidate_hint!r}."
        )
    elif label == EXTENSION:
        reason = (
            f"{candidate['slot_id']} looks like continuation of the same contract, "
            f"with amendment/prolongation wording and shared terms {shared_terms}; "
            f"CPV={cpv_text}, gap_days={gap_text}, notices={candidate.get('notice_ids', [])}. "
            f"Anchor cue: {anchor_hint!r}; candidate cue: {candidate_hint!r}."
        )
    else:
        reason = (
            f"{candidate['slot_id']} is a relaunch/failure case because relance or "
            f"infructuous wording appears with shared terms {shared_terms}; "
            f"CPV={cpv_text}, gap_days={gap_text}, notices={candidate.get('notice_ids', [])}. "
            f"Anchor cue: {anchor_hint!r}; candidate cue: {candidate_hint!r}."
        )
    return label, confidence(label, sim, cpv, gap, index), ev, reason


def verdict_for(dossier: dict[str, Any], labels: list[dict[str, Any]]) -> str:
    if not str(dossier.get("anchor", {}).get("full_text") or dossier.get("anchor", {}).get("objet") or "").strip():
        return ANCHOR_UNUSABLE
    if any(item["label"] == RENEWAL for item in labels):
        return HAS_SUCCESSOR
    mode = dossier.get("pool_disclosure", {}).get("exposure_mode")
    if mode in {"EXHAUSTIVE", "NEAR_EXHAUSTIVE"}:
        return NO_POOL_EXHAUSTIVE
    return NO_AMONG_SHOWN


def annotate_assignment(path: Path, context_prefix: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = []
    global_index = 0
    for dossier in payload.get("dossiers", []):
        labels = []
        for candidate in dossier.get("candidates", []):
            global_index += 1
            label, conf, ev, reason = classify(dossier["anchor"], candidate, global_index)
            labels.append({
                "candidate_slot": candidate["slot_id"],
                "label": label,
                "confidence": conf,
                "reasoning": reason,
                "evidence": ev,
            })
        annotations.append({
            "dossier_id": dossier["dossier_id"],
            "annotator_context_id": f"{context_prefix}_{path.stem}",
            "anchor_verdict": verdict_for(dossier, labels),
            "labels": labels,
        })
    return {
        "assignment_id": payload.get("assignment_id", path.stem),
        "pass_id": payload.get("pass_id", "A"),
        "annotations": annotations,
    }


def build(assignments_dir: Path, submissions_dir: Path, force: bool) -> dict[str, Any]:
    assignments = sorted(
        path for path in assignments_dir.glob("[A-Z][0-9][0-9][0-9].json")
        if path.stem != "MANIFEST"
    )
    if not assignments:
        raise FileNotFoundError(f"no assignments found in {assignments_dir}")

    submissions_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(submissions_dir.glob("A[0-9][0-9][0-9].json"))
    if existing and not force:
        raise FileExistsError(f"{submissions_dir} already contains submissions; use --force")
    if force:
        for path in existing:
            path.unlink()

    totals = {"assignments": 0, "anchor_records": 0, "candidate_labels": 0}
    label_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for path in assignments:
        output = annotate_assignment(path, "ai_assisted_wave1a_rules_v1")
        out_path = submissions_dir / path.name
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        totals["assignments"] += 1
        totals["anchor_records"] += len(output["annotations"])
        for record in output["annotations"]:
            verdict_counts[record["anchor_verdict"]] = verdict_counts.get(record["anchor_verdict"], 0) + 1
            for item in record["labels"]:
                totals["candidate_labels"] += 1
                label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1

    summary = {
        **totals,
        "label_counts": dict(sorted(label_counts.items())),
        "anchor_verdict_counts": dict(sorted(verdict_counts.items())),
        "output_dir": str(submissions_dir),
    }
    (submissions_dir / "AUTO_ANNOTATION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments-dir", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--submissions-dir", type=Path, default=DEFAULT_SUBMISSIONS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignments_dir = args.assignments_dir if args.assignments_dir.is_absolute() else PROJECT_ROOT / args.assignments_dir
    submissions_dir = args.submissions_dir if args.submissions_dir.is_absolute() else PROJECT_ROOT / args.submissions_dir
    summary = build(assignments_dir, submissions_dir, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
