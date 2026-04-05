from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Optional

from blokus_ai.engine.models import (
    BOARD_SIZE,
    TURN_ORDER,
    BoardState,
    Coordinate,
    GameConfig,
    GameResult,
    GameVariant,
    Move,
    PlayerColor,
)
from blokus_ai.engine.pieces import PIECE_IDS, PIECE_SIZES, PIECE_TRANSFORMS


CORNER_BY_COLOR = {
    PlayerColor.BLUE: (0, 0),
    PlayerColor.YELLOW: (0, BOARD_SIZE - 1),
    PlayerColor.RED: (BOARD_SIZE - 1, BOARD_SIZE - 1),
    PlayerColor.GREEN: (BOARD_SIZE - 1, 0),
}

ORTHOGONAL_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def create_initial_state(config: Optional[GameConfig] = None) -> BoardState:
    config = config or GameConfig()
    return BoardState(
        variant=config.variant,
        active_color=PlayerColor.BLUE,
        board=[[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)],
        remaining_pieces_by_color={color: list(PIECE_IDS) for color in TURN_ORDER},
        opened_colors={color: False for color in TURN_ORDER},
        passes_in_row=0,
        shared_color=config.shared_color,
        move_history=[],
        last_piece_placed_by_color={color: None for color in TURN_ORDER},
    )


def clone_state(state: BoardState) -> BoardState:
    return BoardState.model_validate(deepcopy(state.model_dump()))


def next_color(color: PlayerColor) -> PlayerColor:
    index = TURN_ORDER.index(color)
    return TURN_ORDER[(index + 1) % len(TURN_ORDER)]


def get_transform(move: Move):
    for transform in PIECE_TRANSFORMS[move.piece_id]:
        if transform.rotation == move.rotation and transform.reflection == move.reflection:
            return transform
    raise ValueError(f"Unknown transform for {move.piece_id} rotation={move.rotation}.")


def placed_cells(move: Move) -> list[tuple[int, int]]:
    transform = get_transform(move)
    anchor = move.anchor_cell
    return [(anchor.row + row, anchor.col + col) for row, col in transform.cells]


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def get_piece_size(piece_id: str) -> int:
    return PIECE_SIZES[piece_id]


def empty_neighbors(state: BoardState, row: int, col: int) -> Iterable[tuple[int, int]]:
    for d_row, d_col in ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS:
        n_row, n_col = row + d_row, col + d_col
        if in_bounds(n_row, n_col) and state.board[n_row][n_col] is None:
            yield (n_row, n_col)


def frontier_corners(state: BoardState, color: PlayerColor) -> set[tuple[int, int]]:
    if not state.opened_colors[color]:
        return {CORNER_BY_COLOR[color]}

    frontiers: set[tuple[int, int]] = set()
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if state.board[row][col] != color:
                continue
            for d_row, d_col in DIAGONAL_DIRECTIONS:
                corner_row, corner_col = row + d_row, col + d_col
                if not in_bounds(corner_row, corner_col):
                    continue
                if state.board[corner_row][corner_col] is not None:
                    continue
                if any(
                    in_bounds(corner_row + o_row, corner_col + o_col)
                    and state.board[corner_row + o_row][corner_col + o_col] == color
                    for o_row, o_col in ORTHOGONAL_DIRECTIONS
                ):
                    continue
                frontiers.add((corner_row, corner_col))
    return frontiers


def is_legal_move(state: BoardState, move: Move) -> bool:
    if move.is_pass:
        return not generate_legal_moves(state, move.color)
    if move.color != state.active_color:
        return False
    if move.piece_id not in state.remaining_pieces_by_color[move.color]:
        return False

    cells = placed_cells(move)
    if not all(in_bounds(row, col) for row, col in cells):
        return False
    if any(state.board[row][col] is not None for row, col in cells):
        return False

    own_corner_contact = False
    occupied = set(cells)
    for row, col in cells:
        for d_row, d_col in ORTHOGONAL_DIRECTIONS:
            n_row, n_col = row + d_row, col + d_col
            if in_bounds(n_row, n_col) and state.board[n_row][n_col] == move.color:
                return False
        for d_row, d_col in DIAGONAL_DIRECTIONS:
            n_row, n_col = row + d_row, col + d_col
            if in_bounds(n_row, n_col) and state.board[n_row][n_col] == move.color:
                own_corner_contact = True

    if not state.opened_colors[move.color]:
        return CORNER_BY_COLOR[move.color] in occupied
    return own_corner_contact


def generate_legal_moves(
    state: BoardState,
    color: Optional[PlayerColor] = None,
    piece_id: Optional[str] = None,
    max_candidates: Optional[int] = None,
) -> list[Move]:
    target_color = color or state.active_color
    remaining = state.remaining_pieces_by_color[target_color]
    if piece_id is not None:
        if piece_id not in remaining:
            return []
        candidate_pieces = [piece_id]
    else:
        candidate_pieces = sorted(remaining, key=lambda item: (-PIECE_SIZES[item], item))

    frontiers = frontier_corners(state, target_color)
    if not frontiers:
        return []

    legal_moves: list[Move] = []
    seen: set[tuple[str, int, bool, int, int]] = set()
    for candidate_piece in candidate_pieces:
        for transform in PIECE_TRANSFORMS[candidate_piece]:
            for frontier_row, frontier_col in frontiers:
                for cell_row, cell_col in transform.cells:
                    anchor_row = frontier_row - cell_row
                    anchor_col = frontier_col - cell_col
                    key = (
                        candidate_piece,
                        transform.rotation,
                        transform.reflection,
                        anchor_row,
                        anchor_col,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    move = Move(
                        color=target_color,
                        piece_id=candidate_piece,
                        anchor_cell=Coordinate(row=anchor_row, col=anchor_col),
                        rotation=transform.rotation,
                        reflection=transform.reflection,
                    )
                    if is_legal_move(state, move):
                        legal_moves.append(move)
                        if max_candidates is not None and len(legal_moves) >= max_candidates:
                            return legal_moves
    return legal_moves


def apply_pass(state: BoardState, color: Optional[PlayerColor] = None) -> BoardState:
    color = color or state.active_color
    if generate_legal_moves(state, color):
        raise ValueError(f"{color.value} has legal moves and may not pass.")
    new_state = clone_state(state)
    new_state.active_color = next_color(color)
    new_state.passes_in_row += 1
    return new_state


def apply_move(state: BoardState, move: Move) -> BoardState:
    if move.is_pass:
        return apply_pass(state, move.color)
    if not is_legal_move(state, move):
        raise ValueError("Illegal move.")

    new_state = clone_state(state)
    for row, col in placed_cells(move):
        new_state.board[row][col] = move.color

    new_state.remaining_pieces_by_color[move.color].remove(move.piece_id)
    new_state.opened_colors[move.color] = True
    new_state.last_piece_placed_by_color[move.color] = move.piece_id
    new_state.move_history.append(move)
    new_state.active_color = next_color(move.color)
    new_state.passes_in_row = 0
    return new_state


def is_terminal(state: BoardState) -> bool:
    if all(not state.remaining_pieces_by_color[color] for color in TURN_ORDER):
        return True
    return state.passes_in_row >= len(TURN_ORDER)


def color_score(state: BoardState, color: PlayerColor) -> int:
    remaining_cells = sum(PIECE_SIZES[piece_id] for piece_id in state.remaining_pieces_by_color[color])
    score = -remaining_cells
    if not state.remaining_pieces_by_color[color]:
        score += 15
        if state.last_piece_placed_by_color[color] == "I1":
            score += 5
    return score


def group_map(state: BoardState) -> dict[str, list[PlayerColor]]:
    if state.variant == GameVariant.PAIRED_2:
        return {
            "player_a": [PlayerColor.BLUE, PlayerColor.RED],
            "player_b": [PlayerColor.YELLOW, PlayerColor.GREEN],
        }
    if state.variant == GameVariant.SHARED_3:
        shared_color = state.shared_color or PlayerColor.GREEN
        return {
            color.value: [color]
            for color in TURN_ORDER
            if color != shared_color
        }
    return {color.value: [color] for color in TURN_ORDER}


def owner_group(state: BoardState, color: PlayerColor) -> str:
    for group_name, colors in group_map(state).items():
        if color in colors:
            return group_name
    return color.value


def result(state: BoardState) -> GameResult:
    scores_by_color = {color: color_score(state, color) for color in TURN_ORDER}
    grouped: dict[str, int] = {}
    for group_name, colors in group_map(state).items():
        grouped[group_name] = sum(scores_by_color[color] for color in colors)

    winner_group: Optional[str] = None
    if grouped:
        best_score = max(grouped.values())
        winners = [group_name for group_name, score in grouped.items() if score == best_score]
        if len(winners) == 1:
            winner_group = winners[0]

    return GameResult(scores_by_color=scores_by_color, group_scores=grouped, winner_group=winner_group)

