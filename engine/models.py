from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class NoteBook(BaseModel):
    discovered_locations: set[str] = Field(default_factory=set)
    npcs_questioned: set[str] = Field(default_factory=set)
    docs_investigated: set[str] = Field(default_factory=set)
    list_of_clues: list[str] = Field(default_factory=list)


class NPCState(BaseModel):
    talked_topics: set[str] = Field(default_factory=set)
    interaction_count: int = 0


class GameState(BaseModel):
    loop_id: int = 1
    current_day: int = 1
    current_location: str = "shrine"
    ap_remaining: int = Field(default=8, ge=0)

    discovered_clues: set[str] = Field(default_factory=set)
    discovered_facts: set[str] = Field(default_factory=set)

    library_password_found: bool = False
    phase: str = "investigation"
    corruption_level: int = Field(default=0, ge=0)
    game_over: bool = False

    npc_states: dict[str, NPCState] = Field(default_factory=dict)
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


class MoveLocationAction(GameAction):
    action_type: Literal["move_location"] = "move_location"
    location_id: str


class SearchDocAction(GameAction):
    action_type: Literal["search_doc"] = "search_doc"
    document_id: str | None = None
    query: str | None = None


class PerformRitualAction(GameAction):
    action_type: Literal["perform_ritual"] = "perform_ritual"
    student_id: str
    offering_id: str


class CrossCheckAction(GameAction):
    action_type: Literal["cross_check"] = "cross_check"
    document_ids: list[str] = Field(min_length=2)


ActionRequest = Annotated[
    InspectLocationAction
    | TalkNPCAction
    | SearchDocAction
    | PerformRitualAction
    | CrossCheckAction
    | MoveLocationAction,
    Field(discriminator="action_type"),
]


class ActionResult(BaseModel):
    success: bool
    message: str
    action_type: str
    ap_spent: int = 0
    discovered_clues: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
