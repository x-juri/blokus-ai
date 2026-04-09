from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


BOARD_SIZE = 20


class PlayerColor(str, Enum):
    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"
    GREEN = "green"


TURN_ORDER = [
    PlayerColor.BLUE,
    PlayerColor.YELLOW,
    PlayerColor.RED,
    PlayerColor.GREEN,
]


class GameVariant(str, Enum):
    STANDARD_4 = "standard-4"
    PAIRED_2 = "paired-2"
    SHARED_3 = "shared-3"


class Coordinate(BaseModel):
    row: int
    col: int


class Move(BaseModel):
    color: PlayerColor
    piece_id: str
    anchor_cell: Coordinate
    rotation: int = 0
    reflection: bool = False
    is_pass: bool = False


class MoveSuggestion(BaseModel):
    move: Move
    score: float
    rationale: str
    visits: int = 0


AgentId = Literal[
    "heuristic-mcts",
    "policy-mcts",
    "mobility-heuristic",
    "random-legal",
]


class AgentConfig(BaseModel):
    agent_id: AgentId = "heuristic-mcts"
    checkpoint_id: Optional[str] = None
    simulations: Optional[int] = Field(default=None, ge=1, le=4000)
    candidate_limit: Optional[int] = Field(default=None, ge=1, le=256)
    rollout_depth: Optional[int] = Field(default=None, ge=1, le=64)
    exploration_weight: Optional[float] = Field(default=None, gt=0.0, le=4.0)
    root_dirichlet_alpha: Optional[float] = Field(default=None, gt=0.0, le=10.0)
    root_exploration_fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    sampling_temperature: Optional[float] = Field(default=None, gt=0.0, le=4.0)
    sampling_moves: Optional[int] = Field(default=None, ge=0, le=256)
    seed: Optional[int] = None


class GameConfig(BaseModel):
    variant: GameVariant = GameVariant.STANDARD_4
    board_size: int = BOARD_SIZE
    shared_color: Optional[PlayerColor] = None
    top_k_suggestions: int = 5
    mcts_simulations: int = 96
    candidate_limit: int = 24
    rollout_depth: int = 8

    @model_validator(mode="after")
    def validate_variant(self) -> "GameConfig":
        if self.board_size != BOARD_SIZE:
            raise ValueError("The official Blokus board size is fixed at 20.")
        if self.variant == GameVariant.SHARED_3 and self.shared_color is None:
            self.shared_color = PlayerColor.GREEN
        return self


class BoardState(BaseModel):
    variant: GameVariant = GameVariant.STANDARD_4
    active_color: PlayerColor = PlayerColor.BLUE
    board: list[list[Optional[PlayerColor]]] = Field(default_factory=list)
    remaining_pieces_by_color: dict[PlayerColor, list[str]] = Field(default_factory=dict)
    opened_colors: dict[PlayerColor, bool] = Field(default_factory=dict)
    passes_in_row: int = 0
    shared_color: Optional[PlayerColor] = None
    move_history: list[Move] = Field(default_factory=list)
    last_piece_placed_by_color: dict[PlayerColor, Optional[str]] = Field(default_factory=dict)

    @field_validator("board")
    @classmethod
    def validate_board_shape(
        cls, board: list[list[Optional[PlayerColor]]]
    ) -> list[list[Optional[PlayerColor]]]:
        if not board:
            return board
        if len(board) != BOARD_SIZE:
            raise ValueError("Board must contain exactly 20 rows.")
        for row in board:
            if len(row) != BOARD_SIZE:
                raise ValueError("Each board row must contain exactly 20 cells.")
        return board

    @model_validator(mode="after")
    def validate_state(self) -> "BoardState":
        if not self.board:
            self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        if not self.remaining_pieces_by_color:
            self.remaining_pieces_by_color = {color: [] for color in TURN_ORDER}
        if not self.opened_colors:
            self.opened_colors = {color: False for color in TURN_ORDER}
        if not self.last_piece_placed_by_color:
            self.last_piece_placed_by_color = {color: None for color in TURN_ORDER}
        if self.variant == GameVariant.SHARED_3 and self.shared_color is None:
            self.shared_color = PlayerColor.GREEN
        return self


class GameResult(BaseModel):
    scores_by_color: dict[PlayerColor, int]
    group_scores: dict[str, int]
    winner_group: Optional[str] = None
