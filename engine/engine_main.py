from copy import deepcopy

from engine.actions import validate_action
from engine.ap import get_ap_cost
from engine.execution.inspect import execute_inspect
from engine.execution.talk import execute_talk

HANDLERS = {
    "inspect_location": execute_inspect,
    "question_npc": execute_talk,
    # Add search_doc and perform_ritual here.
}


def process_action(state, action, world):
    # Work on a copy so failed actions don't partially mutate state.
    next_state = deepcopy(state)

    validate_action(next_state, action, world)

    handler = HANDLERS.get(action.action_type)
    if handler is None:
        raise ValueError(f"Unsupported action: {action.action_type}")

    cost = get_ap_cost(action.action_type)

    # Charge AP in exactly one place.
    next_state.ap_remaining -= cost

    result = handler(next_state, action, world)

    result.ap_spent = cost
    return next_state, result
