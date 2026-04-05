from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from blokus_ai.ai.mcts import MCTSAgent
from blokus_ai.api.schemas import (
    ApplyMoveRequest,
    LegalMovesRequest,
    SuggestMovesRequest,
    SuggestMovesResponse,
)
from blokus_ai.engine.game import apply_move, create_initial_state, generate_legal_moves
from blokus_ai.engine.models import GameConfig
from blokus_ai.engine.pieces import PIECE_IDS, PIECE_SIZES, PIECE_TRANSFORMS, RAW_PIECES


app = FastAPI(title="Blokus AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/pieces")
def list_pieces() -> dict[str, list[dict]]:
    pieces = [
        {
            "piece_id": piece_id,
            "size": PIECE_SIZES[piece_id],
            "cells": list(RAW_PIECES[piece_id]),
            "transform_count": len(PIECE_TRANSFORMS[piece_id]),
        }
        for piece_id in PIECE_IDS
    ]
    return {"pieces": pieces}


@app.get("/api/initial-state")
def initial_state() -> dict:
    return create_initial_state(GameConfig()).model_dump(mode="json")


@app.post("/api/legal-moves")
def legal_moves(request: LegalMovesRequest) -> dict[str, list[dict]]:
    moves = generate_legal_moves(
        request.state,
        color=request.color,
        piece_id=request.piece_id,
        max_candidates=request.limit,
    )
    return {"moves": [move.model_dump(mode="json") for move in moves]}


@app.post("/api/apply-move")
def apply_move_endpoint(request: ApplyMoveRequest) -> dict:
    try:
        new_state = apply_move(request.state, request.move)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return new_state.model_dump(mode="json")


@app.post("/api/suggest-moves", response_model=SuggestMovesResponse)
def suggest_moves(request: SuggestMovesRequest) -> SuggestMovesResponse:
    agent = MCTSAgent(
        simulations=request.simulations or 96,
        candidate_limit=request.candidate_limit or 24,
        rollout_depth=request.rollout_depth or 8,
    )
    suggestions = agent.suggest(request.state, top_k=request.top_k)
    return SuggestMovesResponse(suggestions=suggestions)

