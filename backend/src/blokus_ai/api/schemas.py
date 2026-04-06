from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from blokus_ai.engine.models import (
    AgentConfig,
    AgentId,
    BoardState,
    GameConfig,
    GameResult,
    GameVariant,
    Move,
    MoveSuggestion,
    PlayerColor,
)


class LegalMovesRequest(BaseModel):
    state: BoardState
    color: Optional[PlayerColor] = None
    piece_id: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=512)


class ApplyMoveRequest(BaseModel):
    state: BoardState
    move: Move


class AgentRequestMixin(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    agent_id: Optional[AgentId] = None
    checkpoint_id: Optional[str] = None
    simulations: Optional[int] = Field(default=None, ge=1, le=4000)
    candidate_limit: Optional[int] = Field(default=None, ge=1, le=256)
    rollout_depth: Optional[int] = Field(default=None, ge=1, le=64)
    exploration_weight: Optional[float] = Field(default=None, gt=0.0, le=4.0)
    seed: Optional[int] = None

    @model_validator(mode="after")
    def merge_agent_overrides(self) -> "AgentRequestMixin":
        overrides = {
            "agent_id": self.agent_id,
            "checkpoint_id": self.checkpoint_id,
            "simulations": self.simulations,
            "candidate_limit": self.candidate_limit,
            "rollout_depth": self.rollout_depth,
            "exploration_weight": self.exploration_weight,
            "seed": self.seed,
        }
        base = self.agent.model_dump()
        for key, value in overrides.items():
            if value is not None:
                base[key] = value
        self.agent = AgentConfig.model_validate(base)
        return self


class SuggestMovesRequest(AgentRequestMixin):
    state: BoardState
    top_k: int = Field(default=5, ge=1, le=20)


class SuggestMovesResponse(BaseModel):
    suggestions: list[MoveSuggestion]


class NewGameRequest(BaseModel):
    config: GameConfig = Field(
        default_factory=lambda: GameConfig(variant=GameVariant.PAIRED_2)
    )


class NewGameResponse(BaseModel):
    state: BoardState


class AiTurnRequest(AgentRequestMixin):
    state: BoardState
    top_k: int = Field(default=3, ge=1, le=12)


class AiTurnResponse(BaseModel):
    move: Optional[Move] = None
    state: BoardState
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    forced_passes: list[Move] = Field(default_factory=list)
    result: Optional[GameResult] = None


class ReplayGameRequest(BaseModel):
    config: GameConfig = Field(
        default_factory=lambda: GameConfig(variant=GameVariant.PAIRED_2)
    )
    player_a_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(agent_id="policy-mcts")
    )
    player_b_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(agent_id="heuristic-mcts")
    )
    seed: int = 7
    max_turns: int = Field(default=512, ge=1, le=1024)


class ReplayGameResponse(BaseModel):
    initial_state: BoardState
    moves: list[Move]
    state_history: list[BoardState]
    result: GameResult
    seed: int
    agent_matchup: dict[str, AgentConfig]
