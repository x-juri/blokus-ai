from __future__ import annotations

from blokus_ai.ai.agents import LargestPieceGreedyAgent, MobilityHeuristicAgent, RandomLegalAgent
from blokus_ai.ai.evaluation import rank_moves
from blokus_ai.ai.mcts import MCTSAgent
from blokus_ai.engine.game import create_initial_state
from blokus_ai.engine.models import GameConfig
from blokus_ai.engine.pieces import PIECE_SIZES


def test_random_agent_returns_legal_suggestions() -> None:
    state = create_initial_state(GameConfig())
    suggestions = RandomLegalAgent(seed=7).suggest(state, top_k=3)
    assert len(suggestions) == 3


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

