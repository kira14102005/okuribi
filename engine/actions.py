from engine.ap import get_ap_cost
from engine.models import GameAction, GameState, InspectLocationAction, NPCatLocation


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


rules: list[NPCatLocation] = [
    NPCatLocation.model_validate({"npc_id": "NPC_MIZUKI", "location_id": "student_council_room"}),
    NPCatLocation.model_validate({"npc_id": "NPC_REN", "location_id": "computer_lab"}),
    NPCatLocation.model_validate({"npc_id": "NPC_HARU", "location_id": "equipment_shed"}),
    NPCatLocation.model_validate({"npc_id": "NPC_SAE", "location_id": "library"}),
]