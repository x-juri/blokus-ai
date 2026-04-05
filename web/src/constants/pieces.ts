import type { PieceDescriptor, PlayerColor } from "../types/blokus";

export const playerColors: PlayerColor[] = ["blue", "yellow", "red", "green"];

export const fallbackPieces: PieceDescriptor[] = [
  { piece_id: "I1", size: 1, cells: [[0, 0]], transform_count: 1 },
  { piece_id: "I2", size: 2, cells: [[0, 0], [0, 1]], transform_count: 2 },
  { piece_id: "I3", size: 3, cells: [[0, 0], [0, 1], [0, 2]], transform_count: 2 },
  { piece_id: "V3", size: 3, cells: [[0, 0], [1, 0], [1, 1]], transform_count: 4 },
  { piece_id: "I4", size: 4, cells: [[0, 0], [0, 1], [0, 2], [0, 3]], transform_count: 2 },
  { piece_id: "O4", size: 4, cells: [[0, 0], [0, 1], [1, 0], [1, 1]], transform_count: 1 },
  { piece_id: "T4", size: 4, cells: [[0, 0], [0, 1], [0, 2], [1, 1]], transform_count: 4 },
  { piece_id: "L4", size: 4, cells: [[0, 0], [1, 0], [2, 0], [2, 1]], transform_count: 8 },
  { piece_id: "Z4", size: 4, cells: [[0, 0], [0, 1], [1, 1], [1, 2]], transform_count: 4 },
  { piece_id: "I5", size: 5, cells: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]], transform_count: 2 },
  { piece_id: "L5", size: 5, cells: [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]], transform_count: 8 },
  { piece_id: "T5", size: 5, cells: [[0, 0], [0, 1], [0, 2], [1, 1], [2, 1]], transform_count: 4 },
  { piece_id: "V5", size: 5, cells: [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], transform_count: 4 },
  { piece_id: "N5", size: 5, cells: [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1]], transform_count: 8 },
  { piece_id: "Z5", size: 5, cells: [[0, 0], [1, 0], [1, 1], [1, 2], [2, 2]], transform_count: 4 },
  { piece_id: "P5", size: 5, cells: [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]], transform_count: 8 },
  { piece_id: "W5", size: 5, cells: [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]], transform_count: 4 },
  { piece_id: "U5", size: 5, cells: [[0, 0], [0, 2], [1, 0], [1, 1], [1, 2]], transform_count: 4 },
  { piece_id: "F5", size: 5, cells: [[0, 1], [1, 0], [1, 1], [1, 2], [2, 0]], transform_count: 8 },
  { piece_id: "X5", size: 5, cells: [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]], transform_count: 1 },
  { piece_id: "Y5", size: 5, cells: [[0, 0], [1, 0], [2, 0], [3, 0], [2, 1]], transform_count: 8 }
];
