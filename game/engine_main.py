from pydantic import TypeAdapter, ValidationError

from engine.actions import (
    action_cost,
    validate_action,
    validate_action_search_doc,
    validate_cross_check_action,
    validate_inspect_location_action,
    validate_move_location_action,
    validate_npc_talk_action,
    validate_ritual_action,
)
from engine.models import (
    ActionRequest,
    ActionResult,
    CrossCheckAction,
    GameState,
    InspectLocationAction,
    MoveLocationAction,
    PerformRitualAction,
    SearchDocAction,
    TalkNPCAction,
)
from game.execution import (
    ArchiveSearch,
    execute_cross_check,
    execute_inspect_location,
    execute_move_location,
    execute_ritual,
    execute_search_doc,
    execute_talk_npc,
)
from game.mcp_docs_client import search_docs_sync

ACTION_ADAPTER = TypeAdapter(ActionRequest)


class GameEngine:
    def __init__(
        self,
        world: dict,
        archive_search: ArchiveSearch | None = None,
    ):
        self.world = world

    def execute_action(
        self,
        state: GameState,
        action: ActionRequest | dict,
    ) -> tuple[GameState, ActionResult]:
        action_type = (
            action.get("action_type", "invalid")
            if isinstance(action, dict)
            else getattr(action, "action_type", "invalid")
        )

        try:
            parsed_action = ACTION_ADAPTER.validate_python(action)

            # Work on a copy: failed actions must not mutate the live state.
            next_state = state.model_copy(deep=True)

            validate_action(next_state, parsed_action, self.world)
            self._validate_specific(next_state, parsed_action)

            cost = action_cost(parsed_action, self.world)
            next_state.ap_remaining -= cost

            result = self._execute(next_state, parsed_action)
            result.ap_spent = cost

            if next_state.ap_remaining == 0 and not next_state.game_over:
                loop_rules = self.world["engine_config"]["loop"]
                next_state.corruption_level += loop_rules["ap_zero_corruption_increase"]
                next_state.phase = "calling"
                next_state.game_over = True

                result.message += "\n\n" + self.world["ending"]["ap_zero"]["scene"]
                result.data["event"] = "ap_zero"

            return next_state, result

        except (ValidationError, ValueError, RuntimeError, KeyError) as exc:
            return state, ActionResult(
                success=False,
                action_type=str(action_type),
                message=str(exc),
                ap_spent=0,
            )

    def _validate_specific(self, state, action) -> None:
        if isinstance(action, InspectLocationAction):
            validate_inspect_location_action(state, action, self.world)
        elif isinstance(action, TalkNPCAction):
            validate_npc_talk_action(state, action, self.world)
        elif isinstance(action, SearchDocAction):
            validate_action_search_doc(state, action, self.world)
        elif isinstance(action, PerformRitualAction):
            validate_ritual_action(state, action, self.world)
        elif isinstance(action, CrossCheckAction):
            validate_cross_check_action(state, action, self.world)
        elif isinstance(action, MoveLocationAction):
            validate_move_location_action(state, action, self.world)
        else:
            raise TypeError(f"Unsupported action type: {action.action_type}")

    def _execute(self, state, action) -> ActionResult:
        if isinstance(action, InspectLocationAction):
            return execute_inspect_location(state, action, self.world)

        if isinstance(action, TalkNPCAction):
            return execute_talk_npc(state, action, self.world)

        if isinstance(action, SearchDocAction):
            return execute_search_doc(state, action, self.world, self.archive_search)

        if isinstance(action, PerformRitualAction):
            return execute_ritual(state, action, self.world)

        if isinstance(action, CrossCheckAction):
            return execute_cross_check(state, action, self.world)

        if isinstance(action, MoveLocationAction):
            return execute_move_location(state, action, self.world)

        raise ValueError(f"No handler for {action.action_type}")

    def new_game(self) -> GameState:
        """Create a fresh state using the World Bible's starting settings."""
        start = self.world["engine_config"]["starting_state"]
        return GameState(
            loop_id=start["loop_id"],
            current_day=start["current_day"],
            current_location=start["current_location"],
            ap_remaining=start["ap_remaining"],
            phase=start["phase"],
            corruption_level=start["corruption_level"],
        )

    def reset_loop(self, state: GameState) -> GameState:
        """Start another loop while preserving persistent notebook evidence."""
        start = self.world["engine_config"]["starting_state"]
        next_state = state.model_copy(deep=True)

        next_state.loop_id += 1
        next_state.current_day = start["current_day"]
        next_state.current_location = start["current_location"]
        next_state.ap_remaining = start["ap_remaining"]
        next_state.library_password_found = False
        next_state.phase = "investigation"
        next_state.game_over = False
        next_state.npc_states.clear()

        # Keep discovered clues and notebook evidence across loops.
        return next_state
