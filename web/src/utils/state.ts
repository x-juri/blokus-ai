import { fallbackPieces, playerColors } from "../constants/pieces";
import type { BoardState, GameResult, PlayerColor, TeamId } from "../types/blokus";

const pieceSizeMap = Object.fromEntries(fallbackPieces.map((piece) => [piece.piece_id, piece.size]));

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

export function teamForColor(color: PlayerColor): TeamId {
  return color === "blue" || color === "red" ? "player_a" : "player_b";
}

export function humanControlsColor(humanSide: TeamId, color: PlayerColor): boolean {
  return teamForColor(color) === humanSide;
}

export function isTerminalState(state: BoardState): boolean {
  const allPiecesExhausted = playerColors.every(
    (color) => state.remaining_pieces_by_color[color].length === 0
  );
  return allPiecesExhausted || state.passes_in_row >= playerColors.length;
}

export function colorScore(state: BoardState, color: PlayerColor): number {
  const remainingPenalty = state.remaining_pieces_by_color[color].reduce(
    (sum, pieceId) => sum + (pieceSizeMap[pieceId] ?? 0),
    0
  );
  let score = -remainingPenalty;
  if (state.remaining_pieces_by_color[color].length === 0) {
    score += 15;
    if (state.last_piece_placed_by_color[color] === "I1") {
      score += 5;
    }
  }
  return score;
}

export function summarizeGame(state: BoardState): GameResult {
  const scores_by_color = {
    blue: colorScore(state, "blue"),
    yellow: colorScore(state, "yellow"),
    red: colorScore(state, "red"),
    green: colorScore(state, "green")
  };

  const group_scores: Record<string, number> =
    state.variant === "paired-2"
      ? {
          player_a: scores_by_color.blue + scores_by_color.red,
          player_b: scores_by_color.yellow + scores_by_color.green
        }
      : {
          blue: scores_by_color.blue,
          yellow: scores_by_color.yellow,
          red: scores_by_color.red,
          green: scores_by_color.green
        };

  const entries = Object.entries(group_scores);
  const topScore = Math.max(...entries.map(([, score]) => score));
  const winners = entries.filter(([, score]) => score === topScore);

  return {
    scores_by_color,
    group_scores,
    winner_group: winners.length === 1 ? winners[0][0] : null
  };
}
