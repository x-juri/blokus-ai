from __future__ import annotations

from typing import Optional

from blokus_ai.engine.game import (
    apply_move,
    apply_legal_move_unchecked,
    color_score,
    frontier_corners,
    generate_legal_moves,
    group_map,
    pass_move_for_color,
    owner_group,
    score_margin_for_color,
)
from blokus_ai.engine.models import BoardState, Move, MoveSuggestion, PlayerColor
from blokus_ai.engine.pieces import PIECE_SIZES


def frontier_count(state: BoardState, color: PlayerColor) -> int:
    return len(frontier_corners(state, color))


def center_bias(move: Move) -> float:
    center = 9.5
    return -(
        abs(move.anchor_cell.row - center) + abs(move.anchor_cell.col - center)
    )


def group_score_margin(state: BoardState, root_color: PlayerColor) -> float:
    return score_margin_for_color(state, root_color)


def heuristic_value(state: BoardState, root_color: PlayerColor) -> float:
    root_group = owner_group(state, root_color)
    groups = group_map(state)

    root_frontier = sum(frontier_count(state, color) for color in groups[root_group])
    rival_frontiers = 0
    for group_name, colors in groups.items():
        if group_name == root_group:
            continue
        rival_frontiers += sum(frontier_count(state, color) for color in colors)

    root_score = sum(color_score(state, color) for color in groups[root_group])
    return (
        group_score_margin(state, root_color)
        + 0.35 * root_frontier
        - 0.18 * rival_frontiers
        + 0.2 * root_score
    )


def describe_move(state: BoardState, move: Move) -> str:
    if move.is_pass:
        return f"{move.color.value} has no legal placements and must pass."
    projected = apply_move(state, move)
    return describe_projected_move(state, move, projected)


def describe_projected_move(state: BoardState, move: Move, projected: BoardState) -> str:
    size = PIECE_SIZES[move.piece_id]
    frontier = frontier_count(projected, move.color)
    margin = group_score_margin(projected, move.color)
    return (
        f"Places {move.piece_id} ({size} squares), preserves {frontier} playable corners, "
        f"and projects a margin of {margin:.1f}."
    )


def heuristic_move_score(state: BoardState, move: Move, root_color: PlayerColor) -> float:
    projected = apply_move(state, move)
    return heuristic_projected_move_score(state, move, root_color, projected)


def heuristic_projected_move_score(
    state: BoardState,
    move: Move,
    root_color: PlayerColor,
    projected: BoardState,
) -> float:
    piece_size = PIECE_SIZES[move.piece_id]
    same_group = owner_group(state, move.color) == owner_group(state, root_color)
    direction = 1.0 if same_group else -1.0
    local_frontier = frontier_count(projected, move.color)
    return (
        heuristic_value(projected, root_color)
        + direction * 0.5 * piece_size
        + direction * 0.25 * local_frontier
        + 0.1 * center_bias(move)
    )


def rank_moves(
    state: BoardState,
    color: PlayerColor,
    root_color: PlayerColor,
    top_k: Optional[int] = None,
    include_rationales: bool = True,
) -> list[MoveSuggestion]:
    ranked = []
    for move in generate_legal_moves(state, color):
        projected = apply_legal_move_unchecked(state, move)
        ranked.append(
            MoveSuggestion(
                move=move,
                score=heuristic_projected_move_score(state, move, root_color, projected),
                rationale=describe_projected_move(state, move, projected)
                if include_rationales
                else "",
                visits=0,
            )
        )
    if not ranked and color == state.active_color:
        pass_move = pass_move_for_color(color)
        ranked = [
            MoveSuggestion(
                move=pass_move,
                score=heuristic_value(state, root_color) - 1.0,
                rationale=describe_move(state, pass_move) if include_rationales else "",
                visits=0,
            )
        ]
    ranked.sort(key=lambda suggestion: suggestion.score, reverse=True)
    return ranked if top_k is None else ranked[:top_k]
