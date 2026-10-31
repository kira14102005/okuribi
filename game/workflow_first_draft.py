from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from engine.models import ActionRequest, ActionResult, GameState
from game.engine_main import ACTION_ADAPTER, WORLD_BIBLE_PATH, GameEngine
from game.llm_context import build_npc_prompt, build_planner_context, create_llm
from game.mcp_docs_client import search_docs_sync
from game.world_loader import load_world

load_dotenv()  # Load environment variables from .env file


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
    action: ActionRequest
    action_result: ActionResult
    npc_dialogue: str
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


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm2 = create_llm()


def talk_as_npc(world, npc_id, player_question, clues):
    prompt = build_npc_prompt(
        world=world,
        npc_id=npc_id,
        player_question=player_question,
        revealed_clues=clues,
    )
    parser = StrOutputParser()
    chain = llm2 | parser
    return chain.invoke(prompt)


def build_workflow(engine: GameEngine, model_name: str, world: dict):
    planner = llm.with_structured_output(ActionPlan)

    narrator = ChatGoogleGenerativeAI(model=model_name)

    def plan_action(state: OkuribiState) -> dict:
        game = state["game_state"]
        prompt = build_planner_context(world, game)
        plan = planner.invoke(prompt)
        # print(f"Planner output: {plan.model_dump_json(indent=2)}")
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

    def execute_action(state: OkuribiState) -> dict:
        action = state["action"]

        next_game, result = engine.execute_action(
            state["game_state"],
            action,
        )

        return {
            "game_state": next_game,
            "action_result": result,
        }

    def talk_to_npc(state: OkuribiState) -> dict:
        """Generate NPC dialogue only after a successful engine action."""

        action = state["action"]
        result = state["action_result"]

        # Never let the NPC respond to an invalid or failed action.
        if action.action_type != "question_npc" or not result.success:
            return {}

        game = state["game_state"]

        # Build context from clues that the engine has actually unlocked.
        # Prefer full clue records from the World Bible when available.
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

        revealed_clues = [
            clues_by_id.get(clue_id, {"id": clue_id})
            for clue_id in sorted(game.discovered_clues)
        ]

        npc_id = action.npc_id
        player_question = state["user_query"]

        dialogue = talk_as_npc(
            world=world,
            npc_id=npc_id,
            player_question=player_question,
            clues=revealed_clues,
        )

        return {"npc_dialogue": dialogue}

    def route_after_planning(state: OkuribiState) -> str:
        if state.get("needs_clarification", False):
            return END
        return "execute_action"

    def route_after_execution(state: OkuribiState) -> str:
        action = state["action"]
        result = state["action_result"]

        if action.action_type == "question_npc" and result.success:
            return "talk_to_npc"

        return "narrate_result"

    def narrate_result(state: OkuribiState) -> dict:
        game = state["game_state"]
        result = state["action_result"]
        npc_dialogue = state.get("npc_dialogue")
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
{result.model_dump_json(indent=2)}

Player's original request:
{state["user_query"]}

NPC dialogue generated for this turn:
{npc_dialogue or "No NPC dialogue was generated for this turn."}

Instructions:
- If NPC dialogue is present, preserve its meaning and character voice.
- Do not invent additional NPC statements or clues.
- Do not claim that a clue was discovered unless the engine result confirms it.
- If the action failed, narrate the failure honestly.
- If NPC dialogue is absent, narrate the engine result normally.
Narrate the result in an atmospheric, concise way.
If the action failed, explain the failure honestly.
"""

        parser = StrOutputParser()
        chain = narrator | parser

        response = chain.invoke(prompt)
        return {"response": response}

    graph = StateGraph(OkuribiState)

    graph.add_node("plan_action", plan_action)
    graph.add_node("execute_action", execute_action)
    graph.add_node("talk_to_npc", talk_to_npc)
    graph.add_node("narrate_result", narrate_result)

    graph.add_edge(START, "plan_action")

    graph.add_conditional_edges(
        "plan_action",
        route_after_planning,
    )
    graph.add_conditional_edges(
        "execute_action",
        route_after_execution,
    )
    graph.add_edge("talk_to_npc", "narrate_result")
    graph.add_edge("narrate_result", END)

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    world = load_world(WORLD_BIBLE_PATH)
    engine = GameEngine(world, search_docs_sync)

    state = engine.new_game()
    print("New game state initialized:")
    print(state.model_dump_json(indent=2))

    workflow = build_workflow(engine, model_name="gemini-3.6-flash", world=world)

    while True:
        user_query = input("\nWhat would you like to do? ")
        state_dict = {"user_query": user_query, "game_state": state}

        state_dict = workflow.invoke(
            state_dict, config={"configurable": {"thread_id": state.loop_id}}
        )

        state = GameState.model_validate(state_dict["game_state"])
        response = state_dict.get("response", "")
        print(f"\n{response}")
