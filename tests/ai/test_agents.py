from __future__ import annotations

import random

import pytest

from blokus_ai.ai.agents import LargestPieceGreedyAgent, MobilityHeuristicAgent, RandomLegalAgent
from blokus_ai.ai.evaluation import rank_moves
from blokus_ai.ai.mcts import (
    MCTSAgent,
    _apply_root_dirichlet_noise,
    _blend_candidate_entries,
    _sample_action_index_from_visit_counts,
)
from blokus_ai.engine.game import create_initial_state, generate_legal_moves
from blokus_ai.engine.models import GameConfig
from blokus_ai.training.encoding import encode_action
from blokus_ai.engine.pieces import PIECE_SIZES


def test_random_agent_returns_legal_suggestions() -> None:
    state = create_initial_state(GameConfig())
    suggestions = RandomLegalAgent(seed=7).suggest(state, top_k=3)
    assert len(suggestions) == 3


def test_random_agent_uses_stateful_rng_between_calls() -> None:
    state = create_initial_state(GameConfig())
    agent = RandomLegalAgent(seed=7)
    first = [suggestion.move for suggestion in agent.suggest(state, top_k=3)]
    second = [suggestion.move for suggestion in agent.suggest(state, top_k=3)]
    assert first != second


def test_greedy_agent_prefers_large_opening_piece() -> None:
    state = create_initial_state(GameConfig())
    suggestion = LargestPieceGreedyAgent().suggest(state, top_k=1)[0]
    assert PIECE_SIZES[suggestion.move.piece_id] == 5


def test_heuristic_ranking_returns_descending_scores() -> None:
    state = create_initial_state(GameConfig())
    suggestions = rank_moves(state, state.active_color, state.active_color, top_k=5)
    assert suggestions[0].score >= suggestions[-1].score


def test_mcts_returns_ranked_move_suggestions() -> None:
    state = create_initial_state(GameConfig())
    suggestions = MCTSAgent(simulations=16, candidate_limit=12, rollout_depth=4).suggest(state, top_k=3)
    assert suggestions
    assert suggestions[0].visits > 0


def test_mobility_agent_returns_at_least_one_suggestion() -> None:
    state = create_initial_state(GameConfig())
    suggestions = MobilityHeuristicAgent().suggest(state, top_k=2)
    assert len(suggestions) == 2


def test_blended_policy_candidates_preserve_heuristic_floor() -> None:
    state = create_initial_state(GameConfig())
    legal_moves = generate_legal_moves(state)
    heuristic_entries = [(legal_moves[0], 0.8), (legal_moves[1], 0.2)]
    policy_entries = [(legal_moves[2], 0.9), (legal_moves[3], 0.1)]

    blended = _blend_candidate_entries(
        heuristic_entries=heuristic_entries,
        policy_entries=policy_entries,
        candidate_limit=3,
        heuristic_floor_fraction=0.5,
        heuristic_prior_weight=0.65,
        policy_prior_weight=0.35,
    )

    blended_actions = {encode_action(move) for move, _ in blended}
    assert encode_action(legal_moves[0]) in blended_actions
    assert encode_action(legal_moves[1]) in blended_actions
    assert sum(prior for _, prior in blended) == pytest.approx(1.0)


def test_root_dirichlet_noise_is_reproducible_and_changes_priors() -> None:
    state = create_initial_state(GameConfig())
    entries = MCTSAgent(simulations=4, candidate_limit=4, rollout_depth=1)._candidate_entries(
        state,
        state.active_color,
    )

    noisy_a = _apply_root_dirichlet_noise(entries, alpha=0.3, exploration_fraction=0.25, rng=random.Random(7))
    noisy_b = _apply_root_dirichlet_noise(entries, alpha=0.3, exploration_fraction=0.25, rng=random.Random(7))

    assert [prior for _, prior in noisy_a] == pytest.approx([prior for _, prior in noisy_b])
    assert [prior for _, prior in noisy_a] != pytest.approx([prior for _, prior in entries])
    assert sum(prior for _, prior in noisy_a) == pytest.approx(1.0)


def test_visit_sampling_with_low_temperature_picks_argmax() -> None:
    sampled = _sample_action_index_from_visit_counts(
        {11: 9, 12: 3, 13: 1},
        temperature=1e-9,
        rng=random.Random(7),
    )
    assert sampled == 11


def test_mcts_select_move_reports_self_play_sampling_and_root_noise() -> None:
    state = create_initial_state(GameConfig())
    agent = MCTSAgent(
        simulations=8,
        candidate_limit=4,
        rollout_depth=1,
        seed=7,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    )
    decision = agent.select_move(
        state,
        top_k=1,
        sample_from_visit_distribution=True,
        sampling_temperature=1.0,
        add_root_dirichlet_noise=True,
    )

    assert decision.chosen_move is not None
    assert decision.diagnostics["sampled_from_visit_distribution"] is True
    assert decision.diagnostics["root_dirichlet_noise_applied"] is True
    assert decision.diagnostics["selected_action_index"] in decision.diagnostics["visit_counts_by_action"]
