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
- For `policy-mcts`, blend heuristic and learned priors at the root candidate stage and blend heuristic and learned leaf values.
- Search and ranking use an internal unchecked apply path only after a move has already been generated as legal.
- Candidate ranking skips rationale generation; rationales are produced only for returned suggestions.
- Policy/value checkpoints are cached per process and invalidated when checkpoint file metadata changes.

## Training-time exploration

- Offline self-play can inject Dirichlet noise into root priors to diversify openings without changing live play behavior.
- Early self-play moves can be sampled from root visit counts instead of always taking the argmax action.
- Each self-play game derives its own seed so the same run is reproducible while still producing distinct trajectories.
- Benchmark evaluation uses seeded diversified opening plies rather than replaying one deterministic initial game.
