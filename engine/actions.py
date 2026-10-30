from engine.ap import get_ap_cost
from engine.models import (
    ActionRequest,
    GameMap,
    GameState,
    InspectLocationAction,
    SearchDocAction,
    TalkNPCAction,
)


def validate_action(state: GameState, action: ActionRequest, world: dict) -> None:
    """
    Validate if the action can be performed given the current game state.

    Args:
        state (GameState): The current game state.
        action (ActionRequest): The action to validate.

    Raises:
        ValueError: If the action cannot be performed due to insufficient AP or invalid action type.
    """
    ap_cost = get_ap_cost(action.action_type)
    if state.ap_remaining < ap_cost:
        raise ValueError(
            f"Insufficient AP to perform action '{action.action_type}'. "
            f"Required: {ap_cost}, Available: {state.ap_remaining}"
        )


def validate_inspect_location_action(
    state: GameState, action: InspectLocationAction, world: dict
) -> None:
    """
    Validate if the 'inspect_location' action can be performed given the current game state.

    Args:
        state (GameState): The current game state.
        action (InspectLocationAction): The action to validate.

    Raises:
        ValueError: If the action cannot be performed due to insufficient AP or invalid action type.
    """

    if action.location_id == "school":
        if state.current_day % 7 == 0:
            raise ValueError(
                "School is closed on Sundays. You cannot inspect the school on this day."
            )
        if "NPC_TAKADA" not in state.notebook.npcs_questioned:
            raise ValueError(
                "You need to question NPC_TAKADA before inspecting the school."
            )

    required_loc_facts = world["locations"][action.location_id].required_facts
    if not all(fact in state.notebook.discovered_clues for fact in required_loc_facts):
        missing_facts = [
            fact
            for fact in required_loc_facts
            if fact not in state.notebook.discovered_clues
        ]
        raise ValueError(
            f"Cannot inspect location '{action.location_id}'. Missing required facts: {', '.join(missing_facts)}"
        )

    if action.location_id in state.notebook.discovered_locations:
        raise ValueError(f"Location '{action.location_id}' has already been inspected.")


def validate_npc_talk_action(
    state: GameState, action: TalkNPCAction, world: dict
) -> None:
    """
    Validate if the 'question_npc' action can be performed given the current game state.

    Args:
        state (GameState): The current game state.
        action (TalkNPCAction): The action to validate.

    Raises:
        ValueError: If the action cannot be performed due to insufficient AP or invalid action type.
    """
    talked_npcs = {npc.npc_id for npc in state.notebook.talked_npcs}

    if action.npc_id in talked_npcs:
        raise ValueError(f"NPC '{action.npc_id}' has already been talked to.")

    if (
        action.npc_id in npc_locations
        and state.current_location != npc_locations[action.npc_id]
    ):
        raise ValueError(
            f"Move to location '{npc_locations[action.npc_id]}' to talk to NPC '{action.npc_id}'."
        )


def validate_action_search_doc(
    state: GameState, action: SearchDocAction, world: dict
) -> None:
    """
    Validate if the 'search_doc' action can be performed given the current game state.

    Args:
        state (GameState): The current game state.
        action (SearchDocAction): The action to validate.

    Raises:
        ValueError: If the action cannot be performed due to insufficient AP or invalid action type.
    """
    if state.current_location != "library":
        raise ValueError(
            f"Cannot search for documents in '{state.current_location}'. Move to 'library' to perform this action."
        )


npc_locations = GameMap(
    rules=[
        {"npc_id": "NPC_MIZUKI", "location_id": "student_council_room"},
        {"npc_id": "NPC_SAE", "location_id": "library"},
        {"npc_id": "NPC_REN", "location_id": "computer_lab"},
        {"npc_id": "NPC_HARU", "location_id": "equipment_shed"},
    ]
).npc_to_locations
