from __future__ import annotations

from blokus_ai.ai.benchmark import build_seeded_opening_state, run_paired_tournament
from blokus_ai.engine.models import AgentConfig, GameConfig, GameVariant


def test_paired_tournament_returns_seat_swapped_rows() -> None:
    rows = run_paired_tournament(
        AgentConfig(agent_id="heuristic-mcts", simulations=4, candidate_limit=4, rollout_depth=2),
        AgentConfig(agent_id="random-legal", seed=7),
        games=1,
        max_turns=6,
    )
    assert len(rows) == 2
    assert all("heuristic-mcts" in row.matchup for row in rows)


def test_paired_tournament_progress_callback_reports_each_game() -> None:
    progress_calls: list[tuple[str, int, int]] = []
    run_paired_tournament(
        AgentConfig(agent_id="random-legal", seed=1),
        AgentConfig(agent_id="random-legal", seed=2),
        games=1,
        max_turns=2,
        progress_callback=lambda seat_name, game_index, total_games: progress_calls.append(
            (seat_name, game_index, total_games)
        ),
    )
    assert progress_calls == [("player_a", 1, 1), ("player_b", 1, 1)]


def test_seeded_opening_state_is_reproducible_and_diversified() -> None:
    state_a = build_seeded_opening_state(
        config=GameConfig(variant=GameVariant.PAIRED_2),
        opening_seed=11,
        opening_plies=4,
    )
    state_b = build_seeded_opening_state(
        config=GameConfig(variant=GameVariant.PAIRED_2),
        opening_seed=11,
        opening_plies=4,
    )
    state_c = build_seeded_opening_state(
        config=GameConfig(variant=GameVariant.PAIRED_2),
        opening_seed=12,
        opening_plies=4,
    )

    assert state_a.move_history == state_b.move_history
    assert len(state_a.move_history) == 4
    assert state_a.move_history != state_c.move_history
