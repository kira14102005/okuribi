from engine.models import (
    CrossCheckAction,
    GameState,
    InspectLocationAction,
    PerformRitualAction,
    SearchDocAction,
    TalkNPCAction,
)


def action_cost(action, world: dict) -> int:
    """Return the configured AP cost for an action."""
    costs = world["engine_config"]["action_costs"]
    action_type = action.action_type

    if action_type not in costs:
        raise ValueError(f"Unknown action type: {action_type}")

    return costs[action_type]


def validate_move_location_action(state, action, world):
    locations = world["engine_config"]["locations_by_id"]

    if action.location_id not in locations:
        raise ValueError(f"Unknown location: {action.location_id}")

    if action.location_id == state.current_location:
        raise ValueError("You are already at this location.")


def validate_action(state, action, world: dict) -> None:
    """Validate shared preconditions before dispatching an action."""
    if state.game_over or state.phase in {"ending", "game_over"}:
        raise ValueError("The game has already ended.")

    cost = action_cost(action, world)
    if state.ap_remaining < cost:
        raise ValueError(f"Insufficient AP: need {cost}, have {state.ap_remaining}.")


def validate_inspect_location_action(
    state: GameState,
    action: InspectLocationAction,
    world: dict,
) -> None:
    locations = world["engine_config"]["locations_by_id"]

    if action.location_id not in locations:
        raise ValueError(f"Unknown location: {action.location_id}")

    location = locations[action.location_id]
    required_facts = location.get("required_facts", [])

    missing = [
        fact
        for fact in required_facts
        if fact not in state.discovered_facts and fact not in state.discovered_clues
    ]
    if missing:
        raise ValueError(
            f"Missing required facts for {action.location_id}: " + ", ".join(missing)
        )

    if action.location_id in state.notebook.discovered_locations:
        raise ValueError(f"Location '{action.location_id}' has already been inspected.")


def validate_npc_talk_action(
    state: GameState,
    action: TalkNPCAction,
    world: dict,
) -> None:
    npcs = world["engine_config"]["npcs_by_id"]
    npc_locations = world["engine_config"]["npc_location_by_id"]

    if action.npc_id not in npcs:
        raise ValueError(f"Unknown NPC: {action.npc_id}")

    expected_location = npc_locations.get(action.npc_id)
    if expected_location and state.current_location != expected_location:
        raise ValueError(f"Move to '{expected_location}' to talk to {action.npc_id}.")

    npc_state = state.npc_states.get(action.npc_id)

    # Prevent repeating the same named topic, but allow different topics.
    if action.topic and npc_state and action.topic in npc_state.talked_topics:
        raise ValueError(
            f"You have already discussed '{action.topic}' with {action.npc_id}."
        )


def validate_action_search_doc(
    state: GameState,
    action: SearchDocAction,
    world: dict,
) -> None:
    access = world["engine_config"]["archive_access"]

    if state.current_location != access["location_id"]:
        raise ValueError(f"Move to '{access['location_id']}' to search the archive.")

    if access["requires_password"] and not state.library_password_found:
        raise ValueError(
            "The archive terminal is locked. Find the library password first."
        )

    if action.document_id:
        docs = world["engine_config"]["documents_by_id"]
        if action.document_id not in docs:
            raise ValueError(f"Unknown document: {action.document_id}")


def validate_ritual_action(
    state: GameState,
    action: PerformRitualAction,
    world: dict,
) -> None:
    if not world["game_rules"]["ritual"]["available_during_investigation"]:  # noqa: SIM102
        if state.phase == "investigation":
            raise ValueError("The ritual is not available yet.")

    allowed_offerings = {
        item["id"] for item in world["game_rules"]["ritual"]["offerings"]
    }
    if action.offering_id not in allowed_offerings:
        raise ValueError(f"Unknown offering: {action.offering_id}")


def validate_cross_check_action(
    state: GameState,
    action: CrossCheckAction,
    world: dict,
) -> None:
    docs = world["engine_config"]["documents_by_id"]

    unknown = [doc_id for doc_id in action.document_ids if doc_id not in docs]
    if unknown:
        raise ValueError(f"Unknown documents: {', '.join(unknown)}")

    not_found = [
        doc_id
        for doc_id in action.document_ids
        if doc_id not in state.notebook.docs_investigated
    ]
    if not_found:
        raise ValueError(
            "Investigate these documents before cross-checking: " + ", ".join(not_found)
        )
