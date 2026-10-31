from __future__ import annotations

import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from engine.models import ActionRequest, ActionResult, GameState
from game.engine_main import ACTION_ADAPTER, WORLD_BIBLE_PATH, GameEngine
from game.llm_context import build_planner_context, create_llm
from game.mcp_docs_client import search_docs_sync
from game.npc_subgraph import build_npc_subgraphs
from game.world_loader import load_world

load_dotenv()


class ActionPlan(BaseModel):
    """Planner output. Only fields relevant to action_type should be populated."""

    needs_clarification: bool = False
    clarification_question: str | None = None

    action_type: (
        Literal[
            "move_location",
            "inspect_location",
            "question_npc",
            "search_doc",
            "cross_check",
            "perform_ritual",
        ]
        | None
    ) = None

    location_id: str | None = None
    npc_id: str | None = None
    topic: str | None = None
    document_id: str | None = None
    query: str | None = None
    document_ids: list[str] | None = None
    student_id: str | None = None
    offering_id: str | None = None


class OkuribiState(TypedDict, total=False):
    user_query: str
    game_state: GameState

    action_plan: dict
    action: ActionRequest
    action_result: ActionResult

    session_id: str
    active_npc_id: str | None
    leave_conversation: bool

    npc_dialogue: str
    response: str
    needs_clarification: bool


EXIT_COMMANDS = {"/back", "/exit", "leave conversation", "stop talking"}


def build_action(plan: ActionPlan) -> ActionRequest:
    """Convert the planner's schema into a validated engine action."""

    if plan.needs_clarification or plan.action_type is None:
        raise ValueError("The player action needs clarification.")

    fields_by_action = {
        "move_location": ["location_id"],
        "inspect_location": ["location_id"],
        "question_npc": ["npc_id", "topic"],
        "search_doc": ["document_id", "query"],
        "cross_check": ["document_ids"],
        "perform_ritual": ["student_id", "offering_id"],
    }

    payload: dict = {"action_type": plan.action_type}
    for field in fields_by_action[plan.action_type]:
        value = getattr(plan, field)
        if value is not None:
            payload[field] = value

    # Action-specific required fields are enforced by the existing Pydantic
    # ActionRequest adapter, so missing npc_id/document_ids/etc. fail here.
    return ACTION_ADAPTER.validate_python(payload)


def _format_planner_context(world: dict, game: GameState, user_query: str) -> str:
    """
    Use the project's context helper, while ensuring the current user request
    is included. Do not add hidden truth, ritual solution, or future clues here.
    """
    context = build_planner_context(world, game)
    if isinstance(context, str):
        safe_context = context
    else:
        safe_context = json.dumps(context, ensure_ascii=False, default=str)

    return f"""
You are the action planner for Okuribi. Convert the player's latest request
into exactly ONE action plan matching the provided structured schema.

Action rules:
- move_location: requires location_id.
- inspect_location: requires location_id.
- question_npc: requires npc_id; topic is optional.
- search_doc: requires document_id or query (or both if appropriate).
- cross_check: requires at least two document_ids.
- perform_ritual: requires student_id and offering_id.
- Populate only fields relevant to the chosen action; leave all other fields null.
- If the player only says something like "talk to NPC_SAE" or "talk to Sae",
  choose question_npc, set npc_id to the matching valid NPC ID, and set topic=null.
  This means enter the conversation without asking a question yet.
- If the player includes an actual question/topic in the same request, set topic
  to that question.
- Never invent IDs. Use only valid IDs in the context.
- If the request is ambiguous or a required target cannot be determined, set
  needs_clarification=true and provide a short clarification_question.
- Do not infer or reveal hidden truth, future clues, or the ritual solution.

Allowed game context:
{safe_context}

Current game state (public operational fields only):
- Current location: {game.current_location}
- Remaining AP: {game.ap_remaining}
- Current phase: {getattr(game, "phase", "unknown")}

Player's latest request:
{user_query}
""".strip()


def _npc_allowed_clues(world: dict, game: GameState, npc_id: str) -> list[dict]:
    """Return only discovered clues that this NPC is allowed to discuss."""
    npc_record = None
    for item in world.get("npcs", []):
        if isinstance(item, dict) and item.get("id") == npc_id:
            npc_record = item
            break

    allowed_ids = set((npc_record or {}).get("allowed_clue_ids", []))
    discovered_ids = set(game.discovered_clues)

    # If the Bible defines an allow-list, enforce it. If not, provide no clue
    # records rather than exposing every discovered clue to an NPC by default.
    permitted_ids = discovered_ids & allowed_ids if allowed_ids else set()

    config = world.get("engine_config", {})
    clues_by_id = config.get("clues_by_id", {})

    if not clues_by_id:
        raw_clues = world.get("clues", [])
        if isinstance(raw_clues, list):
            clues_by_id = {
                clue["id"]: clue
                for clue in raw_clues
                if isinstance(clue, dict) and clue.get("id")
            }
        elif isinstance(raw_clues, dict):
            clues_by_id = raw_clues

    return [
        clues_by_id.get(clue_id, {"id": clue_id}) for clue_id in sorted(permitted_ids)
    ]


def build_workflow(engine: GameEngine, model_name: str, world: dict):
    """
    Build the top-level game workflow.

    Planner and narrator use Gemini as in the supplied implementation.
    NPC dialogue uses the local Ollama model returned by create_llm().
    Each NPC has a separate compiled subgraph and independent in-memory
    checkpoint store. MemorySaver is not durable across process restarts.
    """
    planner_llm = ChatGoogleGenerativeAI(model=model_name)
    planner = planner_llm.with_structured_output(ActionPlan)

    narrator = ChatGoogleGenerativeAI(model=model_name)
    npc_llm = create_llm()
    npc_subgraphs = build_npc_subgraphs(world=world, llm=npc_llm)

    def begin_turn(state: OkuribiState) -> dict:
        return {
            "npc_dialogue": "",
            "leave_conversation": False,
            "needs_clarification": False,
        }

    def plan_action(state: OkuribiState) -> dict:
        game = state["game_state"]
        prompt = _format_planner_context(world, game, state["user_query"])

        try:
            plan = planner.invoke(prompt)
            if not isinstance(plan, ActionPlan):
                plan = ActionPlan.model_validate(plan)

            if plan.needs_clarification or plan.action_type is None:
                return {
                    "action_plan": plan.model_dump(),
                    "needs_clarification": True,
                    "response": (
                        plan.clarification_question or "What would you like to do?"
                    ),
                }

            action = build_action(plan)
            return {
                "action_plan": plan.model_dump(),
                "action": action,
                "needs_clarification": False,
            }

        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            # A malformed/incomplete plan should ask for clarification, not
            # crash the game or reach the engine.
            return {
                "needs_clarification": True,
                "response": (
                    "I couldn't determine that action safely. "
                    "Please specify the target or try phrasing it differently. "
                    f"(Planner validation: {exc})"
                ),
            }

    def execute_action(state: OkuribiState) -> dict:
        next_game, result = engine.execute_action(
            state["game_state"],
            state["action"],
        )
        return {
            "game_state": next_game,
            "action_result": result,
        }

    def enter_npc_conversation(state: OkuribiState) -> dict:
        action = state["action"]
        result = state["action_result"]

        if not result.success:
            return {
                "active_npc_id": None,
                "response": result.message,
            }

        return {
            "active_npc_id": action.npc_id,
            "response": (
                f"You approach {action.npc_id}. "
                "What would you like to ask? (Type /back to leave.)"
            ),
        }

    def npc_turn(state: OkuribiState) -> dict:
        """Treat input as a question for the currently active NPC."""
        query = state["user_query"].strip()
        npc_id = state.get("active_npc_id")

        if query.lower() in EXIT_COMMANDS:
            return {
                "active_npc_id": None,
                "leave_conversation": True,
                "response": "You step away from the conversation. What next?",
            }

        if not npc_id:
            return {
                "leave_conversation": True,
                "response": "No NPC conversation is active. What would you like to do?",
            }

        try:
            action = ACTION_ADAPTER.validate_python(
                {
                    "action_type": "question_npc",
                    "npc_id": npc_id,
                    "topic": query,
                }
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return {
                "leave_conversation": True,
                "response": f"I couldn't process that question: {exc}",
            }

        return {
            "action": action,
            "action_plan": {
                "action_type": "question_npc",
                "npc_id": npc_id,
                "topic": query,
            },
            "leave_conversation": False,
        }

    def talk_to_npc(state: OkuribiState) -> dict:
        """Invoke the selected NPC's own subgraph after engine validation."""
        action = state["action"]
        result = state["action_result"]

        if action.action_type != "question_npc" or not result.success:
            return {"npc_dialogue": ""}

        npc_id = action.npc_id
        npc_graph = npc_subgraphs.get(npc_id)
        if npc_graph is None:
            return {
                "npc_dialogue": (
                    f"{npc_id} is not configured in the World Bible. "
                    "No character response is available."
                ),
                "active_npc_id": npc_id,
            }

        game = state["game_state"]
        clues = _npc_allowed_clues(world, game, npc_id)
        session_id = state.get("session_id", "okuribi-default-session")

        # Every NPC has a separate graph/checkpointer; the thread ID also
        # separates sessions for that NPC.
        subgraph_result = npc_graph.invoke(
            {
                "player_question": action.topic or state["user_query"],
                "revealed_clues": clues,
            },
            config={
                "configurable": {
                    "thread_id": f"{session_id}:{npc_id}",
                }
            },
        )

        return {
            "npc_dialogue": subgraph_result.get(
                "response", "The NPC is silent for a moment."
            ),
            # Keep the conversation active for follow-up questions.
            "active_npc_id": npc_id,
        }

    def narrate_result(state: OkuribiState) -> dict:
        game = state["game_state"]
        result = state["action_result"]
        npc_dialogue = state.get("npc_dialogue", "")

        prompt = f"""
You narrate the outcome of a turn in Okuribi.

Treat the engine result as authoritative. Do not change game rules, invent
discoveries, or claim an action succeeded when success is false.

Current loop: {game.loop_id}
Current corruption: {getattr(game, "corruption_level", 0)}
Remaining AP: {game.ap_remaining}
Current location: {game.current_location}
Discovered clues: {sorted(game.discovered_clues)}

Authoritative action result:
{result.model_dump_json(indent=2)}

NPC dialogue generated for this turn:
{npc_dialogue or "No NPC dialogue was generated for this turn."}

Player's original request:
{state["user_query"]}

Instructions:
- If NPC dialogue is present, preserve its meaning and character voice.
- Do not add claims, clues, or NPC statements that are absent from the source.
- Do not claim a clue was discovered unless the engine result confirms it.
- If the action failed, explain the failure honestly.
- If there is no NPC dialogue, narrate the engine result normally.
- Keep the response concise and atmospheric.
""".strip()

        response = (narrator | StrOutputParser()).invoke(prompt)
        return {"response": response}

    def route_at_start(state: OkuribiState) -> str:
        if state.get("active_npc_id"):
            return "npc_turn"
        return "plan_action"

    def route_after_planning(state: OkuribiState) -> str:
        if state.get("needs_clarification", False):
            return "end"
        return "execute_action"

    def route_after_npc_turn(state: OkuribiState) -> str:
        if state.get("leave_conversation", False):
            return "end"
        return "execute_action"

    def route_after_execution(state: OkuribiState) -> str:
        action = state["action"]
        result = state["action_result"]

        if action.action_type == "question_npc":
            if not result.success:
                return "narrate_result"
            if not action.topic:
                return "enter_npc_conversation"
            return "talk_to_npc"

        return "narrate_result"

    graph = StateGraph(OkuribiState)
    graph.add_node("begin_turn", begin_turn)
    graph.add_node("plan_action", plan_action)
    graph.add_node("execute_action", execute_action)
    graph.add_node("enter_npc_conversation", enter_npc_conversation)
    graph.add_node("npc_turn", npc_turn)
    graph.add_node("talk_to_npc", talk_to_npc)
    graph.add_node("narrate_result", narrate_result)

    graph.add_edge(START, "begin_turn")
    graph.add_conditional_edges(
        "begin_turn",
        route_at_start,
        {
            "plan_action": "plan_action",
            "npc_turn": "npc_turn",
        },
    )
    graph.add_conditional_edges(
        "plan_action",
        route_after_planning,
        {
            "execute_action": "execute_action",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "npc_turn",
        route_after_npc_turn,
        {
            "execute_action": "execute_action",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "execute_action",
        route_after_execution,
        {
            "enter_npc_conversation": "enter_npc_conversation",
            "talk_to_npc": "talk_to_npc",
            "narrate_result": "narrate_result",
        },
    )
    graph.add_edge("enter_npc_conversation", END)
    graph.add_edge("talk_to_npc", "narrate_result")
    graph.add_edge("narrate_result", END)

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    world = load_world(WORLD_BIBLE_PATH)
    engine = GameEngine(world, search_docs_sync)

    game_state = engine.new_game()
    session_id = os.getenv("OKURIBI_SESSION_ID", "okuribi-terminal-demo")

    print("New game state initialized:")
    print(game_state.model_dump_json(indent=2))
    print("\nType /quit to exit. Type /back to leave an NPC conversation.")

    workflow = build_workflow(
        engine,
        model_name=os.getenv("OKURIBI_PLANNER_MODEL", "gemini-3.6-flash"),
        world=world,
    )

    while True:
        user_query = input("\n> ").strip()
        if user_query.lower() in {"/quit", "quit", "exit"}:
            break
        if not user_query:
            continue

        # Keep one stable parent thread ID for this game session. LangGraph
        # restores active_npc_id and other workflow fields from its checkpoint.
        state_dict = workflow.invoke(
            {
                "user_query": user_query,
                "game_state": game_state,
                "session_id": session_id,
            },
            config={"configurable": {"thread_id": session_id}},
        )

        if "game_state" in state_dict:
            game_state = GameState.model_validate(state_dict["game_state"])

        print(f"\n{state_dict.get('response', '')}")
