from __future__ import annotations

from blokus_ai.engine.game import (
    apply_move,
    color_score,
    create_initial_state,
    generate_legal_moves,
    is_legal_move,
    result,
)
from blokus_ai.engine.models import Coordinate, GameConfig, Move, PlayerColor
from blokus_ai.engine.pieces import PIECE_TRANSFORMS


def test_first_move_must_cover_corner() -> None:
    state = create_initial_state(GameConfig())
    illegal = Move(
        color=PlayerColor.BLUE,
        piece_id="I1",
        anchor_cell=Coordinate(row=1, col=1),
        rotation=0,
        reflection=False,
    )
    assert not is_legal_move(state, illegal)


def test_opening_move_can_cover_assigned_corner() -> None:
    state = create_initial_state(GameConfig())
    legal = Move(
        color=PlayerColor.BLUE,
        piece_id="I1",
        anchor_cell=Coordinate(row=0, col=0),
    )
    assert is_legal_move(state, legal)


def test_same_color_edge_contact_is_forbidden() -> None:
    state = create_initial_state(GameConfig())
    state = apply_move(
        state,
        Move(color=PlayerColor.BLUE, piece_id="I1", anchor_cell=Coordinate(row=0, col=0)),
    )
    state = apply_move(
        state,
        Move(color=PlayerColor.YELLOW, piece_id="I1", anchor_cell=Coordinate(row=0, col=19)),
    )
    state = apply_move(
        state,
        Move(color=PlayerColor.RED, piece_id="I1", anchor_cell=Coordinate(row=19, col=19)),
    )
    state = apply_move(
        state,
        Move(color=PlayerColor.GREEN, piece_id="I1", anchor_cell=Coordinate(row=19, col=0)),
    )

    illegal = Move(
        color=PlayerColor.BLUE,
        piece_id="I2",
        anchor_cell=Coordinate(row=0, col=1),
        rotation=90,
    )
    assert not is_legal_move(state, illegal)


def test_corner_contact_is_required_after_opening() -> None:
    state = create_initial_state(GameConfig())
    state = apply_move(
        state,
        Move(color=PlayerColor.BLUE, piece_id="I1", anchor_cell=Coordinate(row=0, col=0)),
    )
    state = apply_move(
        state,
        Move(color=PlayerColor.YELLOW, piece_id="I1", anchor_cell=Coordinate(row=0, col=19)),
    )
    state = apply_move(
        state,
        Move(color=PlayerColor.RED, piece_id="I1", anchor_cell=Coordinate(row=19, col=19)),
    )
    state = apply_move(
        state,
        Move(color=PlayerColor.GREEN, piece_id="I1", anchor_cell=Coordinate(row=19, col=0)),
    )

    legal = Move(
        color=PlayerColor.BLUE,
        piece_id="I2",
        anchor_cell=Coordinate(row=1, col=1),
        rotation=90,
    )
    assert is_legal_move(state, legal)


def test_piece_transform_catalog_has_expected_symmetries() -> None:
    assert len(PIECE_TRANSFORMS["I1"]) == 1
    assert len(PIECE_TRANSFORMS["I2"]) == 2
    assert len(PIECE_TRANSFORMS["O4"]) == 1
    assert len(PIECE_TRANSFORMS["L5"]) == 8


def test_legal_move_generation_produces_opening_choices() -> None:
    state = create_initial_state(GameConfig())
    blue_moves = generate_legal_moves(state, PlayerColor.BLUE)
    assert blue_moves
    assert all(move.color == PlayerColor.BLUE for move in blue_moves)


def test_endgame_bonus_for_finishing_with_single_square() -> None:
    state = create_initial_state(GameConfig())
    state.remaining_pieces_by_color[PlayerColor.BLUE] = []
    state.last_piece_placed_by_color[PlayerColor.BLUE] = "I1"
    assert color_score(state, PlayerColor.BLUE) == 20


def test_result_groups_standard_scores_by_color() -> None:
    state = create_initial_state(GameConfig())
    summary = result(state)
    assert summary.group_scores["blue"] == -89
    assert summary.group_scores["yellow"] == -89

