from langchain_ollama import ChatOllama

MODEL_NAME = "gemma3:1b"


def create_llm():
    return ChatOllama(model=MODEL_NAME, temperature=0, keep_alive="15m")


def get_npc_config(world: dict, npc_id: str) -> dict:
    for npc in world["npcs"]:
        if npc["id"] == npc_id:
            return npc

    raise ValueError(f"Unknown NPC: {npc_id}")


def build_planner_context(world: dict, game_state) -> dict:
    config = world["engine_config"]

    locations = config["locations_by_id"]
    npcs = config["npcs_by_id"]
    documents = config["documents_by_id"]

    return {
        "allowed_action_names": [
            "move_location",
            "inspect_location",
            "question_npc",
            "search_doc",
            "cross_check",
            "perform_ritual",
        ],
        "current_phase": game_state.phase,
        "valid_target_ids": {
            "locations": [
                {"id": key, "name": value.get("name", key)}
                for key, value in locations.items()
            ],
            "npcs": [
                {
                    "id": key,
                    "name": value.get("name", key),
                    "role": value.get("role", "Unknown"),
                }
                for key, value in npcs.items()
            ],
            "documents": [
                {"id": key, "title": value.get("title", key)}
                for key, value in documents.items()
            ],
        },
    }


def build_npc_prompt(
    world: dict,
    npc_id: str,
    player_question: str,
    revealed_clues: list[str],
    recent_dialogue: list[dict] | None = None,
) -> str:
    npc = get_npc_config(world, npc_id)

    # Only the selected NPC's information enters this prompt.
    # The full truth object and other NPCs' secrets stay private.
    persona = {
        "name": npc["name"],
        "role": npc["role"],
        "public_persona": npc["public_persona"],
        "motive": npc["motive"],
        "private_secret": npc["secret"],
        "knows": npc["knows"],
        "does_not_know": npc["does_not_know"],
        "dialogue_style": npc["dialogue_style"],
        "deja_vu_line": npc.get("deja_vu_line"),
    }

    return f"""
You are role-playing {npc["name"]} in Okuribi.

Character profile:
{persona}

Facts already revealed to the player:
{revealed_clues}

Recent dialogue:
{recent_dialogue or []}

Player's question:
{player_question}

Rules:
- Stay in character.
- Do not claim to know facts listed in does_not_know.
- A private secret is not automatically authorized for disclosure.
- Reveal a clue only when the game engine confirms its unlock.
- Do not invent clues, evidence, document contents, or game mechanics.
- Do not reveal another NPC's private information.
- Respond naturally and concisely, in character.
"""
