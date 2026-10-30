from typing import Literal

from pydantic import BaseModel, Field, field_validator


class NoteBook(BaseModel):
    discovered_locations: set[str] = Field(
        default_factory=set,
        description="The set of discovered locations in the notebook.",
    )
    list_of_clues: list[str] = Field(
        default_factory=list,
        description="The list of clues in the notebook.",
    )
    npcs_questioned: set[str] = Field(default_factory=set)
    docs_investigated: set[str] = Field(default_factory=set)


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

    notebook: NoteBook = Field(default_factory=NoteBook)


class GameAction(BaseModel):
    action_type: Literal[
        "inspect_location", "question_npc", "perform_ritual", "search_doc"
    ]


class ActionResult(BaseModel):
    success: bool
    message: str
    ap_spent: int
    discovered_clues: list[str] = Field(
        default_factory=list,
        description="The list of discovered clues as a result of the action.",
    )


VALID_LOCATIONS = {
    "shrine",
    "school",
    "library",
    "cafe",
    "equipment_shed",
    "computer_lab",
    "student_council_room",
}


from collections import defaultdict

from pydantic import BaseModel, computed_field


class NPCatLocation(BaseModel):
    npc_id: str
    location_id: str


class GameMap(BaseModel):
    rules: list[NPCatLocation]

    @computed_field
    @property
    def npc_to_locations(self) -> dict[str, list[str]]:
        """Automatically groups locations by npc_id, supporting multiple locations."""
        npc_dict = defaultdict(list)
        for rule in self.rules:
            npc_dict[rule.npc_id].append(rule.location_id)
        return dict(npc_dict)


class InspectLocationAction(GameAction):
    location_id: str

    @field_validator("location_id")
    def validate_location_id(cls, v):
        if v not in VALID_LOCATIONS:
            raise ValueError(
                f"Invalid location_id: {v}. Must be one of {VALID_LOCATIONS}."
            )
        return v

    result: ActionResult = Field(default_factory=lambda: ActionResult())


class TalkNPCAction(GameAction):
    npc_id: str

    result: ActionResult = Field(default_factory=lambda: ActionResult())
