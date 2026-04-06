from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from blokus_ai.ai.agents import build_agent
from blokus_ai.api.schemas import (
    AiTurnRequest,
    AiTurnResponse,
    ApplyMoveRequest,
    LegalMovesRequest,
    NewGameRequest,
    NewGameResponse,
    ReplayGameRequest,
    ReplayGameResponse,
    SuggestMovesRequest,
    SuggestMovesResponse,
)
from blokus_ai.engine.game import (
    advance_forced_passes,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    has_legal_move,
    is_terminal,
    owner_group,
    pass_move_for_color,
    result,
)
from blokus_ai.engine.models import BoardState, GameConfig
from blokus_ai.engine.pieces import PIECE_IDS, PIECE_SIZES, PIECE_TRANSFORMS, RAW_PIECES


app = FastAPI(title="Blokus AI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _advance_forced_passes_for_group(state: BoardState, group_name: str) -> tuple[BoardState, list]:
    working_state = state
    forced_passes = []
    while (
        not is_terminal(working_state)
        and owner_group(working_state, working_state.active_color) == group_name
        and not has_legal_move(working_state)
    ):
        forced_pass = pass_move_for_color(working_state.active_color)
        working_state = apply_move(working_state, forced_pass)
        forced_passes.append(forced_pass)
    return working_state, forced_passes


def _serialize_state(state: BoardState) -> dict:
    return state.model_dump(mode="json")


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
    return _serialize_state(create_initial_state(GameConfig()))


@app.post("/api/new-game", response_model=NewGameResponse)
def new_game(request: NewGameRequest) -> NewGameResponse:
    return NewGameResponse(state=create_initial_state(request.config))


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
    return _serialize_state(new_state)


@app.post("/api/suggest-moves", response_model=SuggestMovesResponse)
def suggest_moves(request: SuggestMovesRequest) -> SuggestMovesResponse:
    agent = build_agent(request.agent)
    suggestions = agent.suggest(request.state, top_k=request.top_k)
    return SuggestMovesResponse(suggestions=suggestions)


@app.post("/api/ai-turn", response_model=AiTurnResponse)
def ai_turn(request: AiTurnRequest) -> AiTurnResponse:
    target_group = owner_group(request.state, request.state.active_color)
    working_state, forced_before = _advance_forced_passes_for_group(request.state, target_group)
    if is_terminal(working_state) or owner_group(working_state, working_state.active_color) != target_group:
        return AiTurnResponse(
            move=None,
            state=working_state,
            diagnostics={
                "agent_id": request.agent.agent_id,
                "forced_pass_count": len(forced_before),
                "skipped_due_to_forced_pass": bool(forced_before),
            },
            forced_passes=forced_before,
            result=result(working_state) if is_terminal(working_state) else None,
        )

    agent = build_agent(request.agent)
    decision = agent.select_move(working_state, top_k=request.top_k)
    if decision.chosen_move is None:
        raise HTTPException(status_code=400, detail="Agent could not choose a move.")

    try:
        next_state = apply_move(working_state, decision.chosen_move)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    final_state, forced_after = advance_forced_passes(next_state)
    diagnostics = dict(decision.diagnostics)
    diagnostics["forced_pass_count"] = len(forced_before) + len(forced_after)
    diagnostics["forced_passes_after_move"] = len(forced_after)

    return AiTurnResponse(
        move=decision.chosen_move,
        state=final_state,
        diagnostics=diagnostics,
        forced_passes=forced_before + forced_after,
        result=result(final_state) if is_terminal(final_state) else None,
    )


@app.post("/api/replay-game", response_model=ReplayGameResponse)
def replay_game(request: ReplayGameRequest) -> ReplayGameResponse:
    state = create_initial_state(request.config)
    initial_state = state
    moves = []
    state_history = [state]
    player_a_config = request.player_a_agent.model_copy(
        update={"seed": request.player_a_agent.seed if request.player_a_agent.seed is not None else request.seed}
    )
    player_b_config = request.player_b_agent.model_copy(
        update={
            "seed": request.player_b_agent.seed
            if request.player_b_agent.seed is not None
            else request.seed + 1
        }
    )

    team_agents = {
        "player_a": build_agent(player_a_config),
        "player_b": build_agent(player_b_config),
    }

    for _ in range(request.max_turns):
        if is_terminal(state):
            break

        while not is_terminal(state) and not has_legal_move(state):
            forced_pass = pass_move_for_color(state.active_color)
            state = apply_move(state, forced_pass)
            moves.append(forced_pass)
            state_history.append(state)

        if is_terminal(state):
            break

        active_group = owner_group(state, state.active_color)
        agent = team_agents.get(active_group, team_agents["player_a"])
        decision = agent.select_move(state, top_k=1)
        if decision.chosen_move is None:
            break

        state = apply_move(state, decision.chosen_move)
        moves.append(decision.chosen_move)
        state_history.append(state)

    final_result = result(state)
    return ReplayGameResponse(
        initial_state=initial_state,
        moves=moves,
        state_history=state_history,
        result=final_result,
        seed=request.seed,
        agent_matchup={
            "player_a": player_a_config,
            "player_b": player_b_config,
        },
    )
