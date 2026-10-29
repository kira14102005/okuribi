"""Deterministic cross-check rules for the current eight-document prototype."""
from __future__ import annotations

from typing import Any
from world.retrieval.corpus import load_corpus

# These rules are game logic, not model-generated conclusions.
# They identify observable contradictions without revealing the canonical answer.
RULES = [
    {
        "id": "CHECK_TRANSFER_VS_REGISTER",
        "left": "DOC_02",
        "right": "DOC_05",
        "kind": "status_date_conflict",
        "message": "The digital status export says Aoi transferred out, but the paper register still lists Aoi as enrolled.",
    },
    {
        "id": "CHECK_TRANSFER_VS_AUDIT",
        "left": "DOC_02",
        "right": "DOC_07",
        "kind": "timestamp_conflict",
        "message": "The transfer record's displayed date conflicts with the parent commit's later creation date.",
    },
    {
        "id": "CHECK_LANTERN_VS_CHARRED_NOTE",
        "left": "DOC_06",
        "right": "DOC_08",
        "kind": "memorial_conflict",
        "message": "The inventory claims every memorial lantern was released, while the charred note suggests the memorial was unfinished.",
    },
    {
        "id": "CHECK_ANNOUNCEMENT_VS_EQUIPMENT",
        "left": "DOC_01",
        "right": "DOC_03",
        "kind": "timeline_check",
        "message": "The announcement was printed before the festival; the equipment log records preparations on the day before the scheduled event.",
    },
]


def cross_check(
    doc_id_a: str,
    doc_id_b: str,
    corpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    docs = corpus if corpus is not None else load_corpus()
    by_id = {doc["id"]: doc for doc in docs}
    if doc_id_a not in by_id or doc_id_b not in by_id:
        missing = [doc_id for doc_id in (doc_id_a, doc_id_b) if doc_id not in by_id]
        return {"ok": False, "error": "unknown_document_id", "missing_ids": missing}

    requested = {doc_id_a, doc_id_b}
    for rule in RULES:
        if {rule["left"], rule["right"]} == requested:
            return {
                "ok": True,
                "check_id": rule["id"],
                "kind": rule["kind"],
                "document_ids": [doc_id_a, doc_id_b],
                "finding": rule["message"],
            }

    return {
        "ok": True,
        "check_id": None,
        "kind": "no_predefined_rule",
        "document_ids": [doc_id_a, doc_id_b],
        "finding": "No predefined contradiction rule exists for this pair. Do not infer that the records are consistent.",
    }
