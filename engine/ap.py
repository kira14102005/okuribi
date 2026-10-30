AP_COSTS = {
    "inspect_location": 1,
    "question_npc": 2,
    "perform_ritual": 3,
}


def get_ap_cost(action_type: str) -> int:
    try:
        return AP_COSTS[action_type]
    except KeyError:
        raise ValueError(f"Unknown action: {action_type}")
