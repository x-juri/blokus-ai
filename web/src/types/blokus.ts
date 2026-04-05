export type PlayerColor = "blue" | "yellow" | "red" | "green";
export type GameVariant = "standard-4" | "paired-2" | "shared-3";

export interface Coordinate {
  row: number;
  col: number;
}

export interface Move {
  color: PlayerColor;
  piece_id: string;
  anchor_cell: Coordinate;
  rotation: number;
  reflection: boolean;
  is_pass?: boolean;
}

export interface MoveSuggestion {
  move: Move;
  score: number;
  rationale: string;
  visits: number;
}

export interface BoardState {
  variant: GameVariant;
  active_color: PlayerColor;
  board: (PlayerColor | null)[][];
  remaining_pieces_by_color: Record<PlayerColor, string[]>;
  opened_colors: Record<PlayerColor, boolean>;
  passes_in_row: number;
  shared_color: PlayerColor | null;
  move_history: Move[];
  last_piece_placed_by_color: Record<PlayerColor, string | null>;
}

export interface PieceDescriptor {
  piece_id: string;
  size: number;
  cells: [number, number][];
  transform_count: number;
}

export interface PlacementSelection {
  color: PlayerColor;
  pieceId: string;
  rotation: number;
  reflection: boolean;
}
