"""The annotation contract, and the validation that makes it worth trusting.

The v1 benchmark shipped a 49-column schema designed for dual annotation with
adjudication and then used none of it: ``reviewer_1_*``, ``reviewer_2_*`` and
``adjudicated_*`` are null on all 120 rows, ``annotation_status`` reads
``NOT_STARTED`` throughout, all 120 rows carry the same review date, all 70
negatives share one identical ``final_reason``, and ``final_confidence`` is a
deterministic function of ``final_outcome``.

Every check below exists because one of those things happened. They are
enforced at ingest and a batch that fails is rejected rather than recorded with
a warning, because a schema that permits empty adjudication columns is how v1
ended up with empty adjudication columns.

The active labels were produced by the deterministic bootstrap rules in
``scripts/auto_annotate_wave1a.py``. That provenance is disclosed in the
datasheet rather than disguised. Two audit safeguards are retained here: a
label must quote verbatim evidence from the source records, and confidence must
vary within a label class. These safeguards make rule outputs inspectable; they
do not turn the bootstrap labels into independent human ground truth.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "boamp_annotation_schema_v3.0"
INSTRUCTIONS_VERSION = "v3.0"

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

RENEWAL_OF_EXPIRING = "RENEWAL_OF_EXPIRING"
RETENDER_AFTER_FAILURE = "RETENDER_AFTER_FAILURE"
NEXT_PHASE_OF_PROGRAMME = "NEXT_PHASE_OF_PROGRAMME"
PARALLEL_LOT = "PARALLEL_LOT"
EXTENSION_SAME_CONTRACT = "EXTENSION_SAME_CONTRACT"
UNRELATED = "UNRELATED"

#: Six classes, applied in one pass. The classes exist mainly to *exclude*
#: contaminants that the v1 rubric ("same buyer, same functional need")
#: silently admitted -- re-tenders after a failed procedure, and concurrent
#: lots of one programme. Recording them separately also lets a strict and a
#: broad event definition be derived without re-labelling.
LABELS: tuple[str, ...] = (
    RENEWAL_OF_EXPIRING,
    RETENDER_AFTER_FAILURE,
    NEXT_PHASE_OF_PROGRAMME,
    PARALLEL_LOT,
    EXTENSION_SAME_CONTRACT,
    UNRELATED,
)

#: Event sets derived from the taxonomy. Which one is primary is a reported
#: choice, so the collapse is a sensitivity rather than a hidden decision.
EVENT_SETS: Mapping[str, frozenset[str]] = {
    "strict": frozenset({RENEWAL_OF_EXPIRING}),
    "primary": frozenset({RENEWAL_OF_EXPIRING, NEXT_PHASE_OF_PROGRAMME}),
    "broad": frozenset({
        RENEWAL_OF_EXPIRING, NEXT_PHASE_OF_PROGRAMME,
        EXTENSION_SAME_CONTRACT, RETENDER_AFTER_FAILURE,
    }),
}

CONFIDENCE_LEVELS: tuple[str, ...] = ("HIGH", "MEDIUM", "LOW")

# ---------------------------------------------------------------------------
# Anchor verdicts
# ---------------------------------------------------------------------------

HAS_RENEWAL_SUCCESSOR = "HAS_RENEWAL_SUCCESSOR"
NO_RENEWAL_AMONG_SHOWN = "NO_RENEWAL_AMONG_SHOWN"
NO_RENEWAL_POOL_EXHAUSTIVE = "NO_RENEWAL_POOL_EXHAUSTIVE"
ANCHOR_UNUSABLE = "ANCHOR_UNUSABLE"

#: The distinction v1 could not express. Its 70 negatives all meant "not found
#: in a truncated list of 25", but were reported as if they meant "no successor
#: exists". Only an anchor whose whole pool was shown can carry the second
#: claim, so the vocabulary keeps them apart and the metrics respect it.
ANCHOR_VERDICTS: tuple[str, ...] = (
    HAS_RENEWAL_SUCCESSOR,
    NO_RENEWAL_AMONG_SHOWN,
    NO_RENEWAL_POOL_EXHAUSTIVE,
    ANCHOR_UNUSABLE,
)

VERDICT_BY_EXPOSURE_MODE = {
    "EXHAUSTIVE": NO_RENEWAL_POOL_EXHAUSTIVE,
    "NEAR_EXHAUSTIVE": NO_RENEWAL_POOL_EXHAUSTIVE,
    "SAMPLED": NO_RENEWAL_AMONG_SHOWN,
}

# ---------------------------------------------------------------------------
# Evidence rules
# ---------------------------------------------------------------------------

MIN_QUOTE_CHARS = 25
MAX_QUOTE_CHARS = 400

#: Two reasonings sharing this much of their 5-gram content are one template
#: with a noun swapped. Set from a measured case: four reasonings differing
#: only in a trailing site number scored 0.889, and that is boilerplate.
#: Genuinely per-item reasoning about different contracts shares almost no
#: 5-grams, so the threshold has room beneath it.
MAX_REASONING_SIMILARITY = 0.85
REASONING_SHINGLE = 5
MIN_REASONING_CHARS = 40

_WHITESPACE = re.compile(r"\s+")


class AnnotationError(ValueError):
    """A label record that the schema refuses to accept."""


def collapse(text: Any) -> str:
    """Whitespace-collapsed text, for substring verification.

    Only whitespace is normalised. Accents and case are preserved, because a
    quote that does not appear in the source in that form is not a quote.
    """
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def quote_is_verbatim(quote: str, source: Any) -> bool:
    """Whether a quotation really appears in the field it cites.

    This is the check that makes model-produced evidence auditable: a label can
    only cite text that exists, so a reader can confirm every decision against
    the notice without re-reading the whole corpus.
    """
    return bool(quote) and collapse(quote) in collapse(source)


def shingles(text: str, size: int = REASONING_SHINGLE) -> set[str]:
    tokens = collapse(text).casefold().split()
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i: i + size]) for i in range(len(tokens) - size + 1)}


def similarity(left: str, right: str) -> float:
    a, b = shingles(left), shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def conditional_entropy(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    """Entropy of confidence within each label class.

    Zero for a class means confidence was assigned mechanically from the label
    and carries no information -- exactly what v1 shipped, where every negative
    was MEDIUM and every out-of-scope call was HIGH.
    """
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for label, confidence in pairs:
        grouped[label][confidence] += 1
    entropies: dict[str, float] = {}
    for label, counts in grouped.items():
        total = sum(counts.values())
        entropies[label] = -sum(
            (n / total) * math.log2(n / total) for n in counts.values() if n
        )
    return entropies


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


def validate_label_record(
    record: Mapping[str, Any], dossier: Mapping[str, Any]
) -> list[str]:
    """Check one candidate-level label against its dossier. Returns problems."""
    problems: list[str] = []
    slot = record.get("candidate_slot")
    label = record.get("label")
    confidence = record.get("confidence")

    slots = {candidate["slot_id"]: candidate for candidate in dossier.get("candidates", [])}
    fields = {"ANCHOR": dossier.get("anchor", {})}
    fields.update(slots)

    if slot not in slots:
        problems.append(f"candidate_slot {slot!r} is not in this dossier")
    if label not in LABELS:
        problems.append(f"label {label!r} is not one of {LABELS}")
    if confidence not in CONFIDENCE_LEVELS:
        problems.append(f"confidence {confidence!r} is not one of {CONFIDENCE_LEVELS}")

    reasoning = collapse(record.get("reasoning"))
    if len(reasoning) < MIN_REASONING_CHARS:
        problems.append(f"reasoning is shorter than {MIN_REASONING_CHARS} characters")

    evidence = record.get("evidence") or []
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        problems.append("evidence must be a list of quotations")
        return problems

    cited_sides: set[str] = set()
    for item in evidence:
        cited_slot = item.get("slot_id")
        field = item.get("field")
        quote = item.get("quote", "")
        source_record = fields.get(cited_slot)
        if source_record is None:
            problems.append(f"evidence cites unknown slot {cited_slot!r}")
            continue
        if field not in source_record:
            problems.append(f"evidence cites field {field!r} absent from slot {cited_slot!r}")
            continue
        if not quote_is_verbatim(quote, source_record.get(field)):
            problems.append(
                f"quote from {cited_slot}.{field} is not a verbatim substring of the source"
            )
            continue
        length = len(collapse(quote))
        if not MIN_QUOTE_CHARS <= length <= MAX_QUOTE_CHARS:
            problems.append(
                f"quote from {cited_slot}.{field} is {length} characters, "
                f"outside [{MIN_QUOTE_CHARS}, {MAX_QUOTE_CHARS}]"
            )
        cited_sides.add("ANCHOR" if cited_slot == "ANCHOR" else "CANDIDATE")

    # A relation is a claim about two procurements, so it has to be evidenced
    # on both. Only UNRELATED can rest on the candidate alone.
    if label != UNRELATED and cited_sides != {"ANCHOR", "CANDIDATE"}:
        problems.append(
            f"label {label} requires verbatim evidence from both the anchor and the "
            f"candidate; got {sorted(cited_sides) or 'none'}"
        )

    anchor_date = dossier.get("anchor", {}).get("award_date")
    candidate_date = slots.get(slot, {}).get("first_notice_date")
    if label in {RENEWAL_OF_EXPIRING, EXTENSION_SAME_CONTRACT} and anchor_date and candidate_date:
        if str(candidate_date) <= str(anchor_date):
            problems.append(
                f"label {label} requires the candidate to follow the anchor's award "
                f"({candidate_date} is not after {anchor_date})"
            )
    return problems


def validate_anchor_record(
    record: Mapping[str, Any], dossier: Mapping[str, Any]
) -> list[str]:
    """Check one anchor-level verdict, including that nothing was skipped."""
    problems: list[str] = []
    verdict = record.get("anchor_verdict")
    if verdict not in ANCHOR_VERDICTS:
        problems.append(f"anchor_verdict {verdict!r} is not one of {ANCHOR_VERDICTS}")

    mode = dossier.get("pool_disclosure", {}).get("exposure_mode")
    if verdict == NO_RENEWAL_POOL_EXHAUSTIVE and mode == "SAMPLED":
        problems.append(
            "cannot claim the pool was exhausted: only part of it was shown "
            "(exposure_mode=SAMPLED). Use NO_RENEWAL_AMONG_SHOWN."
        )

    labels = record.get("labels") or []
    shown = {candidate["slot_id"] for candidate in dossier.get("candidates", [])}
    labelled = [item.get("candidate_slot") for item in labels]
    missing = shown - set(labelled)
    duplicated = {slot for slot, n in Counter(labelled).items() if n > 1}
    if missing:
        problems.append(f"slots left unlabelled: {sorted(missing)}")
    if duplicated:
        problems.append(f"slots labelled more than once: {sorted(duplicated)}")

    positive = [item for item in labels if item.get("label") == RENEWAL_OF_EXPIRING]
    if verdict == HAS_RENEWAL_SUCCESSOR and not positive:
        problems.append("verdict claims a renewal successor but no slot is labelled as one")
    if verdict in {NO_RENEWAL_AMONG_SHOWN, NO_RENEWAL_POOL_EXHAUSTIVE} and positive:
        problems.append("verdict claims no renewal but a slot is labelled RENEWAL_OF_EXPIRING")

    for item in labels:
        problems.extend(validate_label_record(item, dossier))
    return problems


def validate_batch(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Batch-level checks that no single record can reveal.

    Boilerplate and degenerate confidence are properties of a *set* of labels,
    which is why v1's identical negative reason survived: nothing ever looked
    at the labels together.
    """
    reasonings = [collapse(r.get("reasoning")) for r in records if r.get("reasoning")]
    distinct = len(set(reasonings))
    worst = 0.0
    worst_pair: tuple[int, int] | None = None
    shingle_sets = [shingles(text) for text in reasonings]
    occurrences: dict[str, list[int]] = defaultdict(list)
    for index, values in enumerate(shingle_sets):
        for value in values:
            occurrences[value].append(index)

    shared_counts: Counter[tuple[int, int]] = Counter()
    for indexes in occurrences.values():
        for left_pos, left in enumerate(indexes):
            for right in indexes[left_pos + 1:]:
                shared_counts[(left, right)] += 1

    for (i, j), intersection in shared_counts.items():
        denominator = len(shingle_sets[i]) + len(shingle_sets[j]) - intersection
        score = intersection / denominator if denominator else 0.0
        if score > worst:
            worst, worst_pair = score, (i, j)

    entropies = conditional_entropy(
        (str(r.get("label")), str(r.get("confidence"))) for r in records
    )
    counts = Counter(str(r.get("label")) for r in records)
    degenerate = [
        label for label, entropy in entropies.items()
        if counts[label] >= 10 and entropy == 0.0
    ]

    return {
        "records": len(records),
        "distinct_reasoning_ratio": round(distinct / len(reasonings), 4) if reasonings else None,
        "max_reasoning_similarity": round(worst, 4),
        "most_similar_pair": worst_pair,
        "label_counts": dict(counts),
        "confidence_entropy_by_label": {k: round(v, 4) for k, v in entropies.items()},
        "labels_with_degenerate_confidence": degenerate,
        "passed": bool(
            (not reasonings or distinct == len(reasonings))
            and worst < MAX_REASONING_SIMILARITY
            and not degenerate
        ),
    }


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    """Chance-corrected agreement between two annotation passes.

    Reported with an explicit caveat wherever it appears: both passes here come
    from the same model, so this measures self-consistency under re-presentation
    with a different candidate order, different slot identifiers and a different
    framing. It is not agreement between independent intelligences, and it must
    never be described as if it were.
    """
    if len(first) != len(second):
        raise ValueError("both passes must label the same items")
    if not first:
        return 0.0
    categories = sorted(set(first) | set(second))
    if len(categories) < 2:
        return 1.0 if list(first) == list(second) else 0.0
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / total
    left = Counter(first)
    right = Counter(second)
    expected = sum((left[c] / total) * (right[c] / total) for c in categories)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def agreement_report(
    pass_a: Mapping[tuple[str, str], str], pass_b: Mapping[tuple[str, str], str]
) -> dict[str, Any]:
    """Six-class and binary agreement over the pairs both passes labelled."""
    shared = sorted(set(pass_a) & set(pass_b))
    if not shared:
        return {"overlapping_pairs": 0}
    first = [pass_a[key] for key in shared]
    second = [pass_b[key] for key in shared]
    binary_a = [
        "RENEWAL" if label == RENEWAL_OF_EXPIRING else "NOT" for label in first
    ]
    binary_b = [
        "RENEWAL" if label == RENEWAL_OF_EXPIRING else "NOT" for label in second
    ]
    disagreements = [
        {"pair": list(key), "pass_a": a, "pass_b": b}
        for key, a, b in zip(shared, first, second) if a != b
    ]
    return {
        "overlapping_pairs": len(shared),
        "raw_agreement": round(sum(a == b for a, b in zip(first, second)) / len(shared), 4),
        "cohen_kappa_six_class": round(cohen_kappa(first, second), 4),
        "cohen_kappa_renewal_binary": round(cohen_kappa(binary_a, binary_b), 4),
        "disagreements": disagreements,
        "interpretation": (
            "Both passes were produced by the same model under different candidate "
            "orders, slot identifiers and framing. This is self-consistency, not "
            "inter-annotator agreement."
        ),
    }
