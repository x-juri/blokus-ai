from __future__ import annotations

from blokus_ai.ai.benchmark import run_paired_tournament
from blokus_ai.engine.models import AgentConfig


def test_paired_tournament_returns_seat_swapped_rows() -> None:
    rows = run_paired_tournament(
        AgentConfig(agent_id="heuristic-mcts", simulations=4, candidate_limit=4, rollout_depth=2),
        AgentConfig(agent_id="random-legal", seed=7),
        games=1,
        max_turns=6,
    )
    assert len(rows) == 2
    assert all("heuristic-mcts" in row.matchup for row in rows)
