from __future__ import annotations

from blokus_ai.engine.game import (
    advance_forced_passes,
    create_initial_state,
    group_map,
    owner_group,
    result,
    team_of_color,
    team_to_move,
)
from blokus_ai.engine.models import GameConfig, GameVariant, PlayerColor


def test_two_player_variant_groups_opposite_colors() -> None:
    state = create_initial_state(GameConfig(variant=GameVariant.PAIRED_2))
    groups = group_map(state)
    assert groups["player_a"] == [PlayerColor.BLUE, PlayerColor.RED]
    assert groups["player_b"] == [PlayerColor.YELLOW, PlayerColor.GREEN]
    assert owner_group(state, PlayerColor.RED) == "player_a"
    assert team_of_color(state, PlayerColor.YELLOW) == "player_b"
    assert team_to_move(state) == "player_a"


def test_three_player_variant_ignores_shared_color_in_group_scores() -> None:
    state = create_initial_state(
        GameConfig(variant=GameVariant.SHARED_3, shared_color=PlayerColor.GREEN)
    )
    summary = result(state)
    assert "green" not in summary.group_scores
    assert summary.scores_by_color[PlayerColor.GREEN] == -89


def test_advance_forced_passes_records_pass_moves() -> None:
    state = create_initial_state(GameConfig(variant=GameVariant.PAIRED_2))
    state.remaining_pieces_by_color[PlayerColor.BLUE] = []
    advanced, forced_passes = advance_forced_passes(state)
    assert len(forced_passes) == 1
    assert forced_passes[0].is_pass
    assert advanced.active_color == PlayerColor.YELLOW
    assert advanced.move_history[-1].is_pass
