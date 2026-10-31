from collections.abc import Callable
from typing import Any

from engine.models import (
    ActionResult,
    CrossCheckAction,
    GameState,
    InspectLocationAction,
    NPCState,
    PerformRitualAction,
    SearchDocAction,
    TalkNPCAction,
)

ArchiveSearch = Callable[[str], list[dict[str, Any]]]


def execute_inspect_location(
    state: GameState,
    action: InspectLocationAction,
    world: dict,
) -> ActionResult:
    location = world["engine_config"]["locations_by_id"][action.location_id]

    state.notebook.discovered_locations.add(action.location_id)

    return ActionResult(
        success=True,
        action_type=action.action_type,
        message=f"You inspect {location['name']}.",
        data={"location_id": action.location_id},
    )


def execute_talk_npc(
    state: GameState,
    action: TalkNPCAction,
    world: dict,
) -> ActionResult:
    npc = world["engine_config"]["npcs_by_id"][action.npc_id]

    npc_state = state.npc_states.setdefault(action.npc_id, NPCState())
    npc_state.interaction_count += 1

    if action.topic:
        npc_state.talked_topics.add(action.topic)

    state.notebook.npcs_questioned.add(action.npc_id)

    # Unlock only clues whose deterministic NPC and topic rules match.
    unlocked = []
    for rule in world["engine_config"]["clue_unlock_rules"]:
        if rule.get("action") != "talk_to_npc":
            continue
        if rule.get("npc_id") != action.npc_id:
            continue

        required_topic = rule.get("topic") or rule.get("required_topic")
        if required_topic and action.topic != required_topic:
            continue

        for clue_id in rule.get("unlock", []):
            if clue_id not in state.discovered_clues:
                state.discovered_clues.add(clue_id)
                state.notebook.list_of_clues.append(clue_id)
                unlocked.append(clue_id)

    return ActionResult(
        success=True,
        action_type=action.action_type,
        message=f"You speak with {npc['name']}.",
        discovered_clues=unlocked,
        data={
            "npc_id": action.npc_id,
            "topic": action.topic,
        },
    )


def execute_search_doc(
    state: GameState,
    action: SearchDocAction,
    world: dict,
    archive_search: ArchiveSearch | None,
) -> ActionResult:
    if archive_search is None:
        raise RuntimeError("Archive MCP search has not been configured.")

    query = action.document_id or action.query
    if not query:
        raise ValueError("Provide a document_id or a search query.")

    documents = archive_search(query)

    # Expected normalized MCP result:
    # [{"doc_id": "DOC_01", "title": "...", "text": "...", "type": "..."}]
    unlocked = []
    mapping = world["engine_config"]["document_clue_map"]

    for document in documents:
        doc_id = document.get("doc_id") or document.get("id")
        if not doc_id:
            continue

        state.notebook.docs_investigated.add(doc_id)

        for clue_id in mapping.get(doc_id, []):
            if clue_id not in state.discovered_clues:
                state.discovered_clues.add(clue_id)
                state.notebook.list_of_clues.append(clue_id)
                unlocked.append(clue_id)

    return ActionResult(
        success=True,
        action_type=action.action_type,
        message=(
            f"Archive search returned {len(documents)} document(s)."
            if documents
            else "The archive search returned no matching documents."
        ),
        discovered_clues=unlocked,
        data={"documents": documents},
    )


def execute_ritual(
    state: GameState,
    action: PerformRitualAction,
    world: dict,
) -> ActionResult:
    solution = world["truth"]["ritual_solution"]

    correct = (
        action.student_id == solution["student_id"]
        and action.offering_id == solution["offering_id"]
    )

    if correct:
        state.phase = "ending"
        state.game_over = True

        return ActionResult(
            success=True,
            action_type=action.action_type,
            message=world["ending"]["correct"]["scene"],
            data={"ending": "correct"},
        )

    # A wrong ritual is a valid attempt with an incorrect verdict.
    # It consumes the action, increases corruption, and resets the loop.
    state.corruption_level += world["engine_config"]["loop"][
        "wrong_ritual_corruption_increase"
    ]
    state.loop_id += 1
    state.ap_remaining = world["engine_config"]["loop"]["starting_action_points"]
    state.current_location = "shrine"
    state.phase = "investigation"
    state.game_over = False
    state.npc_states.clear()
    state.ritual_states.clear() if hasattr(state, "ritual_states") else None

    return ActionResult(
        success=False,
        action_type=action.action_type,
        message=world["ending"]["incorrect"]["scene"],
        data={
            "ending": "incorrect",
            "loop_id": state.loop_id,
            "corruption_level": state.corruption_level,
        },
    )


def execute_cross_check(
    state: GameState,
    action: CrossCheckAction,
    world: dict,
) -> ActionResult:
    required = set(action.document_ids)
    unlocked = []

    for rule in world["engine_config"]["clue_unlock_rules"]:
        if rule.get("action") != "cross_check":
            continue

        rule_docs = set(rule.get("document_ids", []))
        if rule_docs.issubset(required):
            for clue_id in rule.get("unlock", []):
                if clue_id not in state.discovered_clues:
                    state.discovered_clues.add(clue_id)
                    state.notebook.list_of_clues.append(clue_id)
                    unlocked.append(clue_id)

    return ActionResult(
        success=True,
        action_type=action.action_type,
        message="You compare the selected documents.",
        discovered_clues=unlocked,
        data={"document_ids": action.document_ids},
    )
