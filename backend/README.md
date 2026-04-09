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

For the full project overview, web app instructions, and architecture notes, see the repository root `README.md`.
