from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from blokus_ai.engine.models import BoardState, Move, MoveSuggestion, PlayerColor


class LegalMovesRequest(BaseModel):
    state: BoardState
    color: Optional[PlayerColor] = None
    piece_id: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=512)


class ApplyMoveRequest(BaseModel):
    state: BoardState
    move: Move


class SuggestMovesRequest(BaseModel):
    state: BoardState
    top_k: int = Field(default=5, ge=1, le=20)
    simulations: Optional[int] = Field(default=None, ge=1, le=2000)
    candidate_limit: Optional[int] = Field(default=None, ge=1, le=128)
    rollout_depth: Optional[int] = Field(default=None, ge=1, le=32)


class SuggestMovesResponse(BaseModel):
    suggestions: list[MoveSuggestion]
