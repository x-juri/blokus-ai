from __future__ import annotations

from dataclasses import dataclass


Coordinate2D = tuple[int, int]


RAW_PIECES: dict[str, tuple[Coordinate2D, ...]] = {
    "I1": ((0, 0),),
    "I2": ((0, 0), (0, 1)),
    "I3": ((0, 0), (0, 1), (0, 2)),
    "V3": ((0, 0), (1, 0), (1, 1)),
    "I4": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "O4": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "T4": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "L4": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "Z4": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "I5": ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    "L5": ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    "T5": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    "V5": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    "N5": ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1)),
    "Z5": ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),
    "P5": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    "W5": ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    "U5": ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    "F5": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
    "X5": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "Y5": ((0, 0), (1, 0), (2, 0), (3, 0), (2, 1)),
}


def normalize(cells: tuple[Coordinate2D, ...]) -> tuple[Coordinate2D, ...]:
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    normalized = tuple(sorted((row - min_row, col - min_col) for row, col in cells))
    return normalized


def rotate(cells: tuple[Coordinate2D, ...]) -> tuple[Coordinate2D, ...]:
    return tuple((col, -row) for row, col in cells)


def reflect(cells: tuple[Coordinate2D, ...]) -> tuple[Coordinate2D, ...]:
    return tuple((row, -col) for row, col in cells)


@dataclass(frozen=True)
class PieceTransform:
    piece_id: str
    rotation: int
    reflection: bool
    cells: tuple[Coordinate2D, ...]
    width: int
    height: int
    size: int


def build_transforms() -> dict[str, list[PieceTransform]]:
    transforms: dict[str, list[PieceTransform]] = {}
    for piece_id, base_cells in RAW_PIECES.items():
        seen: set[tuple[Coordinate2D, ...]] = set()
        options: list[PieceTransform] = []
        for reflection in (False, True):
            transformed = reflect(base_cells) if reflection else base_cells
            for rotation in range(4):
                if rotation:
                    transformed = rotate(transformed)
                normalized = normalize(transformed)
                if normalized in seen:
                    continue
                seen.add(normalized)
                width = max(col for _, col in normalized) + 1
                height = max(row for row, _ in normalized) + 1
                options.append(
                    PieceTransform(
                        piece_id=piece_id,
                        rotation=rotation * 90,
                        reflection=reflection,
                        cells=normalized,
                        width=width,
                        height=height,
                        size=len(normalized),
                    )
                )
        transforms[piece_id] = options
    return transforms


PIECE_TRANSFORMS = build_transforms()
PIECE_IDS = tuple(RAW_PIECES.keys())
PIECE_SIZES = {piece_id: len(cells) for piece_id, cells in RAW_PIECES.items()}

