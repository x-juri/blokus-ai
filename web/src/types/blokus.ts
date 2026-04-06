export type PlayerColor = "blue" | "yellow" | "red" | "green";
export type GameVariant = "standard-4" | "paired-2" | "shared-3";
export type AgentId = "heuristic-mcts" | "policy-mcts" | "mobility-heuristic" | "random-legal";
export type TeamId = "player_a" | "player_b";
export type AppMode = "workbench" | "play";

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

export interface AgentConfig {
  agent_id: AgentId;
  checkpoint_id?: string | null;
  simulations?: number | null;
  candidate_limit?: number | null;
  rollout_depth?: number | null;
  exploration_weight?: number | null;
  seed?: number | null;
}

export interface GameConfig {
  variant: GameVariant;
  board_size?: number;
  shared_color?: PlayerColor | null;
  top_k_suggestions?: number;
  mcts_simulations?: number;
  candidate_limit?: number;
  rollout_depth?: number;
}

export interface GameResult {
  scores_by_color: Record<PlayerColor, number>;
  group_scores: Record<string, number>;
  winner_group: string | null;
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

export interface AiTurnResponse {
  move: Move | null;
  state: BoardState;
  diagnostics: Record<string, unknown>;
  forced_passes: Move[];
  result: GameResult | null;
}

export interface ReplayGameResponse {
  initial_state: BoardState;
  moves: Move[];
  state_history: BoardState[];
  result: GameResult;
  seed: number;
  agent_matchup: Record<TeamId, AgentConfig>;
}
