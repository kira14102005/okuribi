# game/world_loader.py

import json
from pathlib import Path
from typing import Any


def load_world(path: str | Path) -> dict[str, Any]:
    """Load the World Bible and create lookup tables for the engine."""
    with Path(path).open("r", encoding="utf-8") as file:
        world = json.load(file)

    config = world["engine_config"]

    config["locations_by_id"] = {
        location["id"]: location for location in config["locations"]
    }
    config["npcs_by_id"] = {npc["id"]: npc for npc in world["npcs"]}
    config["documents_by_id"] = {
        document["id"]: document for document in world["documents"]
    }
    config["clues_by_id"] = {clue["id"]: clue for clue in world["clues"]}
    config["npc_location_by_id"] = {
        item["npc_id"]: item["location_id"] for item in config["npc_locations"]
    }

    return world
