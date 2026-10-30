from engine.models import GameState


def create_game() -> GameState:
    return GameState(
        loop_id=1,
        current_day=1,
        current_location="shrine",
        ap_remaining=6,
    )
