from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from blokus_ai.engine.game import PASS_PIECE_ID, legal_moves_or_pass, owner_group
from blokus_ai.engine.models import BOARD_SIZE, BoardState, Move, PlayerColor, TURN_ORDER
from blokus_ai.engine.pieces import PIECE_IDS, PIECE_TRANSFORMS

if TYPE_CHECKING:
    import torch


SPATIAL_CHANNELS = 8
VARIANT_FLAGS = ("standard-4", "paired-2", "shared-3")
TRANSFORM_CATALOG = tuple(
    (piece_id, transform.rotation, transform.reflection)
    for piece_id in PIECE_IDS
    for transform in PIECE_TRANSFORMS[piece_id]
)
TRANSFORM_TO_INDEX = {
    transform_key: index for index, transform_key in enumerate(TRANSFORM_CATALOG)
}
ACTION_SPACE_SIZE = len(TRANSFORM_CATALOG) * BOARD_SIZE * BOARD_SIZE + 1
PASS_ACTION_INDEX = ACTION_SPACE_SIZE - 1
NON_SPATIAL_FEATURES = len(TURN_ORDER) * len(PIECE_IDS) + len(VARIANT_FLAGS) + 1 + len(TURN_ORDER)


@dataclass(frozen=True)
class EncodedState:
    spatial: list[list[list[float]]]
    metadata: list[float]


def encode_action(move: Move) -> int:
    if move.is_pass or move.piece_id == PASS_PIECE_ID:
        return PASS_ACTION_INDEX
    transform_index = TRANSFORM_TO_INDEX[(move.piece_id, move.rotation, move.reflection)]
    anchor_index = move.anchor_cell.row * BOARD_SIZE + move.anchor_cell.col
    return transform_index * BOARD_SIZE * BOARD_SIZE + anchor_index


def decode_action(action_index: int, color: PlayerColor) -> Move:
    from blokus_ai.engine.models import Coordinate

    if action_index == PASS_ACTION_INDEX:
        return Move(
            color=color,
            piece_id=PASS_PIECE_ID,
            anchor_cell=Coordinate(row=0, col=0),
            rotation=0,
            reflection=False,
            is_pass=True,
        )

    transform_index, anchor_index = divmod(action_index, BOARD_SIZE * BOARD_SIZE)
    row, col = divmod(anchor_index, BOARD_SIZE)
    piece_id, rotation, reflection = TRANSFORM_CATALOG[transform_index]
    return Move(
        color=color,
        piece_id=piece_id,
        anchor_cell=Coordinate(row=row, col=col),
        rotation=rotation,
        reflection=reflection,
        is_pass=False,
    )


def legal_action_indices(state: BoardState) -> list[int]:
    return [encode_action(move) for move in legal_moves_or_pass(state)]


def legal_action_mask(state: BoardState) -> list[float]:
    mask = [0.0] * ACTION_SPACE_SIZE
    for action_index in legal_action_indices(state):
        mask[action_index] = 1.0
    return mask


def encode_state(state: BoardState) -> EncodedState:
    spatial = [
        [[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for _ in range(SPATIAL_CHANNELS)
    ]

    for color_index, color in enumerate(TURN_ORDER):
        active_channel = len(TURN_ORDER) + color_index
        active_value = 1.0 if color == state.active_color else 0.0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                spatial[active_channel][row][col] = active_value
                if state.board[row][col] == color:
                    spatial[color_index][row][col] = 1.0

    metadata: list[float] = []
    for color in TURN_ORDER:
        remaining = set(state.remaining_pieces_by_color[color])
        metadata.extend(1.0 if piece_id in remaining else 0.0 for piece_id in PIECE_IDS)
    metadata.extend(1.0 if state.variant.value == variant else 0.0 for variant in VARIANT_FLAGS)
    metadata.append(1.0 if state.variant.value == "paired-2" else 0.0)
    metadata.extend(1.0 if state.shared_color == color else 0.0 for color in TURN_ORDER)

    return EncodedState(spatial=spatial, metadata=metadata)


def encode_state_tensors(state: BoardState) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    encoded = encode_state(state)
    spatial = torch.tensor(encoded.spatial, dtype=torch.float32).unsqueeze(0)
    metadata = torch.tensor(encoded.metadata, dtype=torch.float32).unsqueeze(0)
    return spatial, metadata


def action_priors_from_logits(
    logits: "torch.Tensor",
    legal_indices: list[int],
) -> dict[int, float]:
    import torch

    if not legal_indices:
        return {}

    index_tensor = torch.tensor(legal_indices, dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(0, index_tensor)
    weights = torch.softmax(selected_logits, dim=0)
    return {
        action_index: float(weight)
        for action_index, weight in zip(legal_indices, weights.tolist())
    }


def perspective_sign(state: BoardState, root_color: PlayerColor) -> float:
    return 1.0 if owner_group(state, state.active_color) == owner_group(state, root_color) else -1.0
