from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class GameState(BaseModel):
    loop_id: int = Field(
        default=1, description="The current loop ID of the game state."
    )
    current_day: int = Field(
        default=1, description="The current day in the game state."
    )
    current_location: str = Field(
        default="shrine", description="The current location in the game state."
    )
    ap_remaining: int = Field(
        default=3, ge=0, description="The remaining action points in the game state."
    )

    discovered_clues: set[str] = Field(
        default_factory=set,
        description="The set of discovered clues in the game state.",
    )

    notebook_entries: list[str] = Field(
        default_factory=list,
        description="The list of notebook entries in the game state.",
    )


class GameAction(BaseModel):
    action_type: Literal[
        "inspect_location", "question_npc", "perform_ritual", "search_doc"
    ]
    target_id: str


class ActionResult(BaseModel):
    success: bool
    message: str
    ap_spent: int
    discovered_clues: list[str] = Field(
        default_factory=list,
        description="The list of discovered clues as a result of the action.",
    )
