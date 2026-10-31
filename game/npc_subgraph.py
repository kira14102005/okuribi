from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from game.llm_context import build_npc_prompt


class NPCConversationState(TypedDict, total=False):
    player_question: str
    revealed_clues: list[dict[str, Any]]
    dialogue_history: list[dict[str, str]]
    response: str


def build_npc_subgraph(world: dict, npc_id: str, llm):
    """
    Build an isolated conversation subgraph for one NPC.

    Each NPC gets its own graph instance and MemorySaver, so SAE's history
    is not mixed with Mizuki's, Ren's, or Haru's. MemorySaver persists only
    while this Python process is alive; use a SQLite/Postgres checkpointer for
    persistence across restarts.
    """

    def respond(state: NPCConversationState) -> dict:
        question = state["player_question"].strip()
        history = list(state.get("dialogue_history", []))
        clues = state.get("revealed_clues", [])

        prompt = build_npc_prompt(
            world=world,
            npc_id=npc_id,
            player_question=question,
            revealed_clues=clues,
            recent_dialogue=history[-8:],
        )

        response = (llm | StrOutputParser()).invoke(prompt).strip()

        # Save the last few turns in this NPC's own checkpointed state.
        updated_history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]

        return {
            "response": response,
            "dialogue_history": updated_history[-20:],
        }

    graph = StateGraph(NPCConversationState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=MemorySaver())


def build_npc_subgraphs(world: dict, llm) -> dict[str, Any]:
    """
    Build one independently checkpointed subgraph per NPC ID in the Bible.
    """
    subgraphs: dict[str, Any] = {}

    for npc in world.get("npcs", []):
        if not isinstance(npc, dict):
            continue
        npc_id = npc.get("id")
        if npc_id:
            subgraphs[npc_id] = build_npc_subgraph(
                world=world,
                npc_id=npc_id,
                llm=llm,
            )

    return subgraphs
