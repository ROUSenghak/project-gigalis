"""Repair of the explicit notice-link field, applied downstream of extraction.

``standardize.extract_linked_notice_ids`` searches ``annonce_lie`` and
``annonces_anterieures`` with :data:`standardize.IDWEB_RE`
(``(?<!\\d)(\\d{2}-\\d+)(?!\\d)``). ``annonces_anterieures`` is a nested
structure that also carries ``DATE_PUBLICATION`` values such as
``"2014-10-15"``, and the ``MM-DD`` tail of an ISO date has exactly the shape of
a notice identifier: in ``2014-10-15`` the ``20`` shields ``14-10`` from the
lookbehind, but ``10-15`` matches cleanly.

The result is that **450,397 of the 524,636 notices carrying links (86%)
contain at least one fabricated identifier**. The most frequent are plainly
calendrical -- ``03-31`` 7,458 times, ``09-30`` 7,343, ``06-30`` 7,095.

The repair is applied here rather than in
:mod:`boamp_pipeline.standardize` on purpose: fixing the regex would require
re-standardising 1,620,712 notices and would invalidate every frozen v2
artifact, including the episode layer the existing study is built on. Two
measured facts make a downstream filter exactly as reliable:

* no real identifier collides with the fragment shape -- of 1,620,712 known
  identifiers, **zero** have a month-like prefix (01-12) together with a
  two-digit suffix;
* genuine identifiers always carry a 4-6 digit sequence, including the 443
  pre-2015 identifiers referenced from inside the corpus (``06-259536``,
  ``12-172293``), whereas date fragments always end in exactly two digits.

Links matter to the benchmark as a *negative* constraint. Research over the
585,516 resolved links showed 97.1% join notices published in the same year and
only 0.47% exceed 24 months -- and those are amendment notices pointing back at
their own award. So an explicit link never marks a renewal; it marks the *same
procurement*, and pairs it joins must be removed from the candidate pool before
anything is called a successor.
"""

from __future__ import annotations

import json
import re
from typing import Any, Collection, Iterable

#: Shape of a notice identifier: a two-digit year prefix and a numeric sequence.
IDWEB_SHAPE = re.compile(r"^(\d{2})-(\d+)$")

#: Shortest sequence accepted for an identifier that cannot be verified against
#: the corpus. Every one of the 443 pre-2015 identifiers referenced from within
#: the corpus has at least four digits; date fragments have exactly two.
MIN_UNVERIFIED_SEQUENCE_DIGITS = 4

#: Plausible two-digit year prefixes. BOAMP identifiers observed in and around
#: this corpus span 2006-2025; the ceiling allows for notices published after
#: the current corpus without admitting arbitrary tokens.
MIN_YEAR_PREFIX = 5
MAX_YEAR_PREFIX = 26


def parse_ids(value: Any) -> list[str]:
    """Parse a ``linked_notice_ids_json`` cell into a list of raw tokens."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item not in (None, "")]


def looks_like_date_fragment(token: str) -> bool:
    """Whether a token is a ``MM-DD`` tail extracted from a publication date.

    Used for reporting how much of the field was fabricated. Detection is not
    what the filter relies on -- verification against the real identifier
    universe is -- but quantifying the defect keeps it visible.
    """
    match = IDWEB_SHAPE.match(token)
    if not match:
        return False
    prefix, sequence = match.group(1), match.group(2)
    if len(sequence) != 2:
        return False
    return 1 <= int(prefix) <= 12 and 1 <= int(sequence) <= 31


def is_plausible_id(token: str) -> bool:
    """Structural plausibility, for identifiers outside the corpus.

    Applies only when a token cannot be checked against the known universe:
    references to notices published before 2015 are real but unverifiable here.
    """
    match = IDWEB_SHAPE.match(token)
    if not match:
        return False
    prefix, sequence = match.group(1), match.group(2)
    if len(sequence) < MIN_UNVERIFIED_SEQUENCE_DIGITS:
        return False
    return MIN_YEAR_PREFIX <= int(prefix) <= MAX_YEAR_PREFIX


def clean_linked_notice_ids(
    value: Any, known_idweb: Collection[str] | None = None
) -> list[str]:
    """Return the genuine notice identifiers in one ``linked_notice_ids_json``.

    A token is kept when it is a known corpus identifier, or -- when it is not
    -- when it is structurally plausible as a reference to a notice outside the
    corpus. Order is preserved and duplicates are dropped.

    >>> clean_linked_notice_ids('["14-151460","10-15"]', {"14-151460"})
    ['14-151460']
    >>> clean_linked_notice_ids('["12-172293"]', set())
    ['12-172293']
    """
    known = known_idweb if known_idweb is not None else ()
    kept: list[str] = []
    for token in parse_ids(value):
        if token in kept:
            continue
        if token in known or is_plausible_id(token):
            kept.append(token)
    return kept


def link_partners(
    ids_by_notice: Iterable[tuple[str, Any]], known_idweb: Collection[str]
) -> dict[str, set[str]]:
    """Build the symmetric same-procurement adjacency over notice identifiers.

    Explicit links are directional in the source but describe a mutual
    "these are the same procurement" relation, so both directions are recorded.
    """
    partners: dict[str, set[str]] = {}
    for notice_id, raw in ids_by_notice:
        for target in clean_linked_notice_ids(raw, known_idweb):
            if target == notice_id:
                continue
            partners.setdefault(str(notice_id), set()).add(target)
            partners.setdefault(target, set()).add(str(notice_id))
    return partners


def pollution_report(values: Iterable[Any], known_idweb: Collection[str]) -> dict[str, int]:
    """Count how much of the field is fabricated, for the run summary."""
    report = {
        "notices_with_links": 0,
        "notices_with_fabricated_ids": 0,
        "raw_tokens": 0,
        "kept_tokens": 0,
        "date_fragment_tokens": 0,
        "other_rejected_tokens": 0,
    }
    for value in values:
        tokens = parse_ids(value)
        if not tokens:
            continue
        report["notices_with_links"] += 1
        report["raw_tokens"] += len(tokens)
        kept = clean_linked_notice_ids(value, known_idweb)
        report["kept_tokens"] += len(kept)
        rejected = [token for token in tokens if token not in kept]
        if rejected:
            report["notices_with_fabricated_ids"] += 1
        for token in rejected:
            if looks_like_date_fragment(token):
                report["date_fragment_tokens"] += 1
            else:
                report["other_rejected_tokens"] += 1
    return report
