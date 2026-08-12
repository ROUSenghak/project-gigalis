"""Conservative deterministic reconstruction of BOAMP procurement episodes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from typing import Any, Iterable

import numpy as np


EPISODE_VERSION = "boamp_episode_reconstruction_v1.0"
REFERENCE_MAX_SPAN_DAYS = 730


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


class IndexedDisjointSet:
    """Memory-bounded union-find with component-level trusted buyer checks."""

    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)
        self.trusted_buyer: dict[int, str] = {}

    def set_trusted_buyer(self, index: int, siren: str) -> None:
        if siren:
            self.trusted_buyer[index] = siren

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> tuple[bool, str]:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return True, "already_connected"
        left_buyer = self.trusted_buyer.get(left_root, "")
        right_buyer = self.trusted_buyer.get(right_root, "")
        if left_buyer and right_buyer and left_buyer != right_buyer:
            return False, "trusted_buyer_conflict"
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
            left_buyer, right_buyer = right_buyer, left_buyer
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        self.trusted_buyer.pop(right_root, None)
        resolved_buyer = left_buyer or right_buyer
        if resolved_buyer:
            self.trusted_buyer[left_root] = resolved_buyer
        return True, "accepted"

    def roots(self) -> np.ndarray:
        for index in range(len(self.parent)):
            self.parent[index] = self.find(index)
        return self.parent.copy()


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in {None, ""}]
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def parse_json_array(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def trusted_buyer_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_siren = str(left.get("buyer_siren") or "")
    right_siren = str(right.get("buyer_siren") or "")
    return bool(left_siren and right_siren and left_siren != right_siren)


def episode_id(notice_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(notice_ids)))
    return "EP-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def deduplicate_text(values: Iterable[Any]) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return "\n".join(result)


def choose_origin(rows: list[dict[str, Any]]) -> tuple[date | None, str]:
    competition_natures = {"APPEL_OFFRE", "PRE-INFORMATION", "PERIODIQUE"}
    competition_dates = [
        row.get("publication_date") for row in rows
        if row.get("nature") in competition_natures and row.get("publication_date")
    ]
    if competition_dates:
        return min(competition_dates), "earliest_substantive_competition_notice"
    publication_dates = [row.get("publication_date") for row in rows if row.get("publication_date")]
    return (min(publication_dates), "earliest_episode_notice") if publication_dates else (None, "unresolved")


def build_edges(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    by_id = {str(row["idweb"]): row for row in rows}
    proposed: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    contractfolder_groups: dict[str, list[str]] = {}
    for row in rows:
        key = str(row.get("contractfolderid") or "").strip()
        if key:
            contractfolder_groups.setdefault(key, []).append(str(row["idweb"]))
    for key, notice_ids in contractfolder_groups.items():
        anchor = notice_ids[0]
        for candidate in notice_ids[1:]:
            proposed.append({"left": anchor, "right": candidate, "method": "contractfolderid", "evidence": key})

    for row in rows:
        right = str(row["idweb"])
        for left in parse_json_list(row.get("linked_notice_ids_json")):
            if left in by_id and left != right:
                proposed.append({"left": left, "right": right, "method": "explicit_linked_notice", "evidence": left})

    reference_groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        reference = str(row.get("procedure_reference") or "").strip()
        buyer_key = str(row.get("buyer_key") or "").strip()
        if reference and buyer_key:
            reference_groups.setdefault((buyer_key, reference), []).append(str(row["idweb"]))
    for (buyer_key, reference), notice_ids in reference_groups.items():
        sorted_ids = sorted(notice_ids, key=lambda notice_id: by_id[notice_id].get("publication_date") or date.min)
        for left, right in zip(sorted_ids, sorted_ids[1:]):
            left_date = by_id[left].get("publication_date")
            right_date = by_id[right].get("publication_date")
            span = (right_date - left_date).days if left_date and right_date else None
            if span is not None and span <= REFERENCE_MAX_SPAN_DAYS:
                proposed.append({"left": left, "right": right, "method": "exact_reference_same_buyer", "evidence": f"{buyer_key}|{reference}"})
            else:
                counters["rejected_reference_span"] += 1

    priority = {"contractfolderid": 1, "explicit_linked_notice": 2, "exact_reference_same_buyer": 3}
    proposed.sort(key=lambda edge: (priority[edge["method"]], edge["left"], edge["right"]))
    return proposed, counters


def reconstruct(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    by_id = {str(row["idweb"]): row for row in rows}
    dsu = DisjointSet(by_id)
    proposed, counters = build_edges(rows)
    edge_rows: list[dict[str, Any]] = []
    accepted_methods_by_notice: dict[str, set[str]] = {notice_id: set() for notice_id in by_id}
    for edge in proposed:
        left = by_id[edge["left"]]
        right = by_id[edge["right"]]
        accepted = not trusted_buyer_conflict(left, right)
        reason = "accepted" if accepted else "trusted_buyer_conflict"
        if accepted:
            dsu.union(edge["left"], edge["right"])
            accepted_methods_by_notice[edge["left"]].add(edge["method"])
            accepted_methods_by_notice[edge["right"]].add(edge["method"])
            counters[f"accepted_{edge['method']}"] += 1
        else:
            counters[f"rejected_{reason}"] += 1
        edge_rows.append({**edge, "accepted": accepted, "decision_reason": reason})

    components: dict[str, list[str]] = {}
    for notice_id in by_id:
        components.setdefault(dsu.find(notice_id), []).append(notice_id)

    membership: list[dict[str, Any]] = []
    for notice_ids in components.values():
        current_episode_id = episode_id(notice_ids)
        for notice_id in notice_ids:
            methods = sorted(accepted_methods_by_notice[notice_id])
            membership.append({
                "idweb": notice_id,
                "episode_id": current_episode_id,
                "notice_edge_methods_json": json_compact(methods),
                "episode_version": EPISODE_VERSION,
            })
    return membership, edge_rows, dict(counters)


def aggregate_episode(notice_rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    ids = sorted(str(row["idweb"]) for row in notice_rows)
    origin_date, origin_source = choose_origin(notice_rows)
    publication_dates = [row.get("publication_date") for row in notice_rows if row.get("publication_date")]
    last_date = max(publication_dates) if publication_dates else None
    buyer_sirens = sorted({str(row.get("buyer_siren") or "") for row in notice_rows if row.get("buyer_siren")})
    buyer_keys = sorted({str(row.get("buyer_key") or "") for row in notice_rows if row.get("buyer_key")})
    buyer_names = [row.get("buyer_name_raw") for row in notice_rows]
    regions = sorted({str(row.get("buyer_region") or "") for row in notice_rows if row.get("buyer_region")})
    departments = sorted({str(row.get("buyer_department") or "") for row in notice_rows if row.get("buyer_department")})
    all_cpvs = sorted({code for row in notice_rows for code in parse_json_list(row.get("all_cpvs_json"))})
    main_cpvs = [str(row.get("main_cpv") or "") for row in notice_rows if row.get("main_cpv")]
    duration_values = sorted({round(float(row["duration_months"]), 2) for row in notice_rows if row.get("duration_months") is not None})
    duration_conflict = len(duration_values) > 1
    amount_values = [item for row in notice_rows for item in parse_json_array(row.get("amount_candidates_json"))]
    natures = sorted({str(row.get("nature") or "") for row in notice_rows if row.get("nature")})
    notice_types = sorted({str(row.get("type_avis") or "") for row in notice_rows if row.get("type_avis")})
    references = sorted({str(row.get("procedure_reference") or "") for row in notice_rows if row.get("procedure_reference")})
    primary_method = "singleton"
    for candidate in ("contractfolderid", "explicit_linked_notice", "exact_reference_same_buyer"):
        if candidate in methods:
            primary_method = candidate
            break
    buyer_conflict = len(buyer_sirens) > 1
    reference_conflict = len(references) > 1
    quality = "high" if primary_method in {"contractfolderid", "explicit_linked_notice"} and not buyer_conflict else "medium" if primary_method == "exact_reference_same_buyer" and not buyer_conflict else "singleton" if primary_method == "singleton" else "review"

    return {
        "episode_id": episode_id(ids),
        "constituent_notice_ids_json": json_compact(ids),
        "notice_count": len(ids),
        "episode_origin_date": origin_date,
        "episode_origin_date_source": origin_source,
        "episode_last_notice_date": last_date,
        "episode_span_days": (last_date - origin_date).days if origin_date and last_date else None,
        "episode_notice_natures_json": json_compact(natures),
        "episode_notice_types_json": json_compact(notice_types),
        "buyer_siren": buyer_sirens[0] if len(buyer_sirens) == 1 else "",
        "buyer_key": buyer_keys[0] if len(buyer_keys) == 1 else "",
        "buyer_name_raw": deduplicate_text(buyer_names),
        "buyer_name_normalized": deduplicate_text(row.get("buyer_name_normalized") for row in notice_rows),
        "buyer_id_quality": "trusted_siren" if len(buyer_sirens) == 1 else "conflict" if buyer_conflict else "name_only",
        "buyer_department": departments[0] if len(departments) == 1 else "",
        "buyer_region": regions[0] if len(regions) == 1 else "",
        "grand_ouest_flag": any(bool(row.get("grand_ouest_flag")) for row in notice_rows),
        "episode_text": deduplicate_text(row.get("notice_text") for row in notice_rows),
        "main_cpv": Counter(main_cpvs).most_common(1)[0][0] if main_cpvs else "",
        "all_cpvs_json": json_compact(all_cpvs),
        "procedure_references_json": json_compact(references),
        "procedure_type": deduplicate_text(row.get("procedure_libelle") or row.get("type_procedure") for row in notice_rows),
        "framework_flag": any(bool(row.get("framework_flag")) for row in notice_rows),
        "duration_months": duration_values[0] if len(duration_values) == 1 else None,
        "duration_quality": "typed_consensus" if len(duration_values) == 1 else "conflict" if duration_conflict else "unresolved",
        "amount_candidates_json": json_compact(amount_values),
        "episode_reconstruction_method": primary_method,
        "episode_reconstruction_methods_json": json_compact(sorted(set(methods))),
        "episode_reconstruction_quality": quality,
        "buyer_conflict_flag": buyer_conflict,
        "procedure_reference_conflict_flag": reference_conflict,
        "duration_conflict_flag": duration_conflict,
        "impossible_chronology_flag": bool(origin_date and last_date and last_date < origin_date),
        "episode_version": EPISODE_VERSION,
    }
