from engine.ap import get_ap_cost
from engine.models import (
    GameAction,
    GameMap,
    GameState,
    InspectLocationAction,
    TalkNPCAction,
)


def validate_action(state: GameState, action: GameAction) -> None:
    """
    Validate if the action can be performed given the current game state.

    Args:
        state (GameState): The current game state.
        action (GameAction): The action to validate.

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
    state: GameState, action: InspectLocationAction
) -> None:
    """
    Validate if the 'inspect_location' action can be performed given the current game state.

    Args:
        state (GameState): The current game state.
        action (GameAction): The action to validate.

    Raises:
        ValueError: If the action cannot be performed due to insufficient AP or invalid action type.
    """
    if action.location_id == state.current_location:
        raise ValueError(
            f"Cannot inspect the current location '{state.current_location}'."
        )

    for action.location_id in state.notebook.discovered_locations:
        raise ValueError(f"Location '{action.location_id}' has already been inspected.")


def validate_npc_talk_action(state: GameState, action: TalkNPCAction) -> None:
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

    if action.npc_id in rules and state.current_location != rules[action.npc_id]:
        raise ValueError(
            f"Move to location '{rules[action.npc_id]}' to talk to NPC '{action.npc_id}'."
        )


rules = GameMap(
    rules=[
        {"npc_id": "NPC_MIZUKI", "location_id": "student_council_room"},
        {"npc_id": "NPC_MIZUKI", "location_id": "library"},
        {"npc_id": "NPC_REN", "location_id": "computer_lab"},
        {"npc_id": "NPC_HARU", "location_id": "equipment_shed"},
    ]
).npc_to_locations
