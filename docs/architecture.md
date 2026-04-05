# Architecture

## Backend

- `engine/` owns the immutable game state, piece catalogue, move generation, scoring, and variant behavior.
- `ai/` owns evaluation heuristics, baseline agents, and the MCTS suggester.
- `training/` owns self-play trace generation, dataset loading, and lightweight PyTorch model scaffolding.
- `api/` exposes the engine and AI through FastAPI.

## Frontend

- Board editing state mirrors the backend `BoardState` schema.
- Suggestions come from the backend and can be previewed or applied directly.
- Piece availability and active color stay explicit so users can reconstruct arbitrary midgame states.

## Search strategy

- Rank legal moves with a fast heuristic before expansion.
- Use progressive widening so each node explores only the best-ranked candidates first.
- Use a short rollout with heuristic policies for opponents, then finish with a static evaluation when the depth budget ends.

