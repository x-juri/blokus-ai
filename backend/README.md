# Blokus AI Backend

Python backend for the Blokus AI project.

It contains:

- the exact Blokus rules engine
- baseline agents, heuristic MCTS, and policy-guided MCTS
- FastAPI endpoints for legal moves, move application, suggestions, paired-2 play turns, and replay generation
- offline paired-2 RL commands:
  - `uv run blokus-self-play`
  - `uv run blokus-train`
  - `uv run blokus-phase1`
  - `uv run blokus-benchmark`

The offline RL path now uses seeded self-play exploration:

- per-game derived seeds for reproducible trace generation
- root Dirichlet noise during self-play MCTS
- visit-distribution sampling for early self-play plies
- seeded diversified openings for evaluation benchmarks

Performance-oriented backend behavior:

- generated legal moves are applied through an internal unchecked path inside search and ranking
- pass checks use early-exit legal move generation
- MCTS ranking avoids building human-readable rationales until final suggestions
- loaded policy/value checkpoints are cached and reloaded when their file metadata changes

For the full project overview, web app instructions, and architecture notes, see the repository root `README.md`.
