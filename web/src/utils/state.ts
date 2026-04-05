import { fallbackPieces, playerColors } from "../constants/pieces";
import type { BoardState, PlayerColor } from "../types/blokus";

export function createEmptyBoard(): (PlayerColor | null)[][] {
  return Array.from({ length: 20 }, () => Array.from({ length: 20 }, () => null));
}

export function createInitialState(): BoardState {
  const allPieces = fallbackPieces.map((piece) => piece.piece_id);
  return {
    variant: "standard-4",
    active_color: "blue",
    board: createEmptyBoard(),
    remaining_pieces_by_color: {
      blue: [...allPieces],
      yellow: [...allPieces],
      red: [...allPieces],
      green: [...allPieces]
    },
    opened_colors: {
      blue: false,
      yellow: false,
      red: false,
      green: false
    },
    passes_in_row: 0,
    shared_color: null,
    move_history: [],
    last_piece_placed_by_color: {
      blue: null,
      yellow: null,
      red: null,
      green: null
    }
  };
}

export function syncOpenedColors(board: (PlayerColor | null)[][]): Record<PlayerColor, boolean> {
  return {
    blue: board.some((row) => row.includes("blue")),
    yellow: board.some((row) => row.includes("yellow")),
    red: board.some((row) => row.includes("red")),
    green: board.some((row) => row.includes("green"))
  };
}

export function countPlacedCells(board: (PlayerColor | null)[][]): Record<PlayerColor, number> {
  const counts: Record<PlayerColor, number> = {
    blue: 0,
    yellow: 0,
    red: 0,
    green: 0
  };
  for (const row of board) {
    for (const cell of row) {
      if (cell) {
        counts[cell] += 1;
      }
    }
  }
  return counts;
}

export function rotateActiveColor(color: PlayerColor): PlayerColor {
  const index = playerColors.indexOf(color);
  return playerColors[(index + 1) % playerColors.length];
}

