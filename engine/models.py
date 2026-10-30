from collections import defaultdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator


class NoteBook(BaseModel):
    discovered_locations: set[str] = Field(
        default_factory=set,
        description="The set of discovered locations in the notebook.",
    )
    npcs_questioned: set[str] = Field(default_factory=set)
    docs_investigated: set[str] = Field(default_factory=set)


class NPCState(BaseModel):
    talked_topics: set[str] = Field(default_factory=set)
    interaction_count: int = 0


class RitualState(BaseModel):
    completed: bool = False
    completed_steps: set[str] = Field(default_factory=set)


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
        default=10, ge=0, description="The remaining action points in the game state."
    )

    discovered_clues: set[str] = Field(
        default_factory=set,
        description="The set of discovered clues in the game state.",
    )
    discovered_facts: set[str] = Field(default_factory=set)
    npc_states: dict[str, NPCState] = Field(default_factory=dict)
    ritual_states: dict[str, RitualState] = Field(default_factory=dict)
    notebook: NoteBook = Field(default_factory=NoteBook)


class GameAction(BaseModel):
    pass


class InspectLocationAction(GameAction):
    action_type: Literal["inspect_location"] = "inspect_location"
    location_id: str


class TalkNPCAction(GameAction):
    action_type: Literal["question_npc"] = "question_npc"
    npc_id: str
    topic: str | None = None


class SearchDocAction(GameAction):
    action_type: Literal["search_doc"] = "search_doc"
    document_id: str | None = None


class PerformRitualAction(GameAction):
    action_type: Literal["perform_ritual"] = "perform_ritual"
    ritual_id: str


ActionRequest = Annotated[
    InspectLocationAction | TalkNPCAction | SearchDocAction | PerformRitualAction,
    Field(discriminator="action_type"),
]


class ActionResult(BaseModel):
    success: bool
    message: str
    action_type: str
    ap_spent: int = 0
    discovered_clues: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


VALID_LOCATIONS = {
    "shrine",
    "school",
    "library",
    "cafe",
    "equipment_shed",
    "computer_lab",
    "student_council_room",
}


class NPCatLocation(BaseModel):
    npc_id: str
    location_id: str


class GameMap(BaseModel):
    rules: list[NPCatLocation]

    @computed_field
    @property
    def npc_to_locations(self) -> dict[str, str]:
        """Automatically groups locations by npc_id, supporting multiple locations."""
        npc_dict = defaultdict(str)
        for rule in self.rules:
            npc_dict[rule.npc_id] = rule.location_id
        return dict(npc_dict)
