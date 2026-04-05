from __future__ import annotations

from blokus_ai.ai.benchmark import run_benchmark


def test_benchmark_runner_returns_rows_for_all_baselines() -> None:
    rows = run_benchmark(games=1, plies=2, mcts_simulations=4, candidate_limit=4, rollout_depth=2)
    names = [row.agent for row in rows]
    assert "random-legal" in names
    assert "largest-piece-greedy" in names
    assert "mobility-heuristic" in names
    assert "progressive-mcts" in names
