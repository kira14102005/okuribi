from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from engine.models import ActionRequest, ActionResult, GameState
from game.engine_main import ACTION_ADAPTER, WORLD_BIBLE_PATH, GameEngine
from game.mcp_docs_client import search_docs_sync
from game.world_loader import load_world


class ActionPlan(BaseModel):
    """Structured interpretation of one player turn."""

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
    action: dict
    action_result: dict
    response: str
    needs_clarification: bool


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

    payload = {"action_type": plan.action_type}

    for field in fields_by_action[plan.action_type]:
        value = getattr(plan, field)
        if value is not None:
            payload[field] = value

    return ACTION_ADAPTER.validate_python(payload)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)


def build_workflow(engine: GameEngine, model_name: str):
    planner = llm.with_structured_output(ActionPlan)

    narrator = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.3,
    )

    def plan_action(state: OkuribiState) -> dict:
        game = state["game_state"]

        prompt = f"""
You are the action interpreter for the game Okuribi.

Interpret the player's request as ONE game action.
Return only the requested structured schema.

Allowed action types:
- move_location: requires location_id
- inspect_location: requires location_id
- question_npc: requires npc_id; topic is optional
- search_doc: requires a query or document_id
- cross_check: requires at least two document_ids
- perform_ritual: requires student_id and offering_id

Do not invent IDs. Use only IDs from the World Bible
and game context supplied to you. If essential information
is missing or the request is ambiguous, ask for clarification.

Current loop: {game.loop_id}
Current day: {game.current_day}
Current location: {game.current_location}
Remaining AP: {game.ap_remaining}
Corruption level: {game.corruption_level}
Discovered clues: {sorted(game.discovered_clues)}
Notebook clues: {game.notebook.list_of_clues}

Player request:
{state["user_query"]}
"""
        plan = planner.invoke(prompt)

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
            "action": action.model_dump(),
            "needs_clarification": False,
        }

    def execute_action(state: OkuribiState) -> dict:
        action = state["action"]

        next_game, result = engine.execute_action(
            state["game_state"],
            action,
        )

        return {
            "game_state": next_game,
            "action_result": result.model_dump(),
        }

    def route_after_planning(state: OkuribiState) -> str:
        if state.get("needs_clarification", False):
            return END
        return "execute_action"

    def narrate_result(state: OkuribiState) -> dict:
        game = state["game_state"]
        result = state["action_result"]

        prompt = f"""
You narrate the outcome of a turn in Okuribi.

Treat the engine result as authoritative. Do not change
game rules, invent discoveries, or claim an action succeeded
when success is false.

Current loop: {game.loop_id}
Current corruption: {game.corruption_level}
Remaining AP: {game.ap_remaining}
Current location: {game.current_location}
Discovered clues: {sorted(game.discovered_clues)}

Authoritative action result:
{result}

Player's original request:
{state["user_query"]}

Narrate the result in an atmospheric, concise way.
If the action failed, explain the failure honestly.
"""
        response = narrator.invoke(prompt)

        return {"response": response.content}

    graph = StateGraph(OkuribiState)

    graph.add_node("plan_action", plan_action)
    graph.add_node("execute_action", execute_action)
    graph.add_node("narrate_result", narrate_result)

    graph.add_edge(START, "plan_action")

    graph.add_conditional_edges(
        "plan_action",
        route_after_planning,
    )

    graph.add_edge("execute_action", "narrate_result")
    graph.add_edge("narrate_result", END)

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    world = load_world(WORLD_BIBLE_PATH)
    engine = GameEngine(world, search_docs_sync)

    state = engine.new_game()

    workflow = build_workflow(engine, model_name="gemini-3.8-flash")
    
    while True:
        user_query = input("\nWhat would you like to do? ")
        state_dict = {"user_query": user_query, "game_state": state.model_dump()}

        state_dict = workflow.run(state_dict)

        state = GameState.model_validate(state_dict["game_state"])
        response = state_dict.get("response", "")
        print(f"\n{response}")
