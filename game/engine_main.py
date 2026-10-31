import asyncio
from copy import deepcopy

from engine.actions import (
    validate_action,
    validate_inspect_location_action,
    validate_npc_talk_action,
)
from engine.ap import get_ap_cost
from engine.models import (
    ActionResult,
    GameAction,
    GameState,
    InspectLocationAction,
    TalkNPCAction,
)
from game.mcp_docs_client import search_docs


class GameEngine:
    """Coordinates deterministic execution of Okuribi game actions."""

    def execute_action(
        self,
        state: GameState,
        action: GameAction,
    ) -> tuple[GameState, ActionResult]:
        # Work on a copy so failed actions don't partially modify live state.
        next_state = deepcopy(state)

        try:
            # 1. General validation, including available AP.
            validate_action(next_state, action)

            # 2. Action-specific validation and execution.
            if isinstance(action, InspectLocationAction):
                result = self._inspect_location(next_state, action)

            elif isinstance(action, TalkNPCAction):
                result = self._talk_npc(next_state, action)

            elif action.action_type == "search_doc":
                result = self._search_doc(next_state, action)

            else:
                raise ValueError(f"Unsupported action: {action.action_type}")

            # 3. Charge AP only after successful execution.
            ap_cost = get_ap_cost(action.action_type)
            next_state.ap_remaining -= ap_cost

            # 4. Return the updated state and action result.
            return next_state, result

        except ValueError as exc:
            # Discard the modified copy if validation/execution fails.
            return state, ActionResult(
                success=False,
                message=str(exc),
                ap_spent=0,
                discovered_clues=[],
            )

    def _inspect_location(
        self,
        state: GameState,
        action: InspectLocationAction,
    ) -> ActionResult:
        validate_inspect_location_action(state, action)

        # MVP behavior: record that the location was investigated.
        state.notebook.discovered_locations.add(action.location_id)

        return ActionResult(
            success=True,
            message=f"You inspect {action.location_id}.",
            ap_spent=get_ap_cost(action.action_type),
            discovered_clues=[],
        )

    def _talk_npc(
        self,
        state: GameState,
        action: TalkNPCAction,
    ) -> ActionResult:
        validate_npc_talk_action(state, action)

        # MVP behavior: record that this NPC was questioned.
        state.notebook.npcs_questioned.add(action.npc_id)

        return ActionResult(
            success=True,
            message=f"You question {action.npc_id}.",
            ap_spent=get_ap_cost(action.action_type),
            discovered_clues=[],
        )

    def _search_doc(
        self,
        state: GameState,
        action: GameAction,
    ) -> ActionResult:
        docs = asyncio.run(search_docs())
        state.notebook.docs_investigated.update(doc.get("title", "") for doc in docs)
        state.discovered_clues.update(doc.get("content", "") for doc in docs)
        return ActionResult(
            success=True,
            message=f"You searched the docs and found {len(docs)} documents.",
            ap_spent=get_ap_cost(action.action_type),
            discovered_clues=[doc.get("content", "") for doc in docs],
        )
