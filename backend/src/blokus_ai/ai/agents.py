from __future__ import annotations

import random
from dataclasses import dataclass

from blokus_ai.ai.evaluation import rank_moves
from blokus_ai.engine.game import generate_legal_moves
from blokus_ai.engine.models import BoardState, MoveSuggestion, PlayerColor
from blokus_ai.engine.pieces import PIECE_SIZES


class BaseAgent:
    name = "base"

    def suggest(self, state: BoardState, top_k: int = 1) -> list[MoveSuggestion]:
        raise NotImplementedError


@dataclass
class RandomLegalAgent(BaseAgent):
    seed: int = 0
    name: str = "random-legal"

    def suggest(self, state: BoardState, top_k: int = 1) -> list[MoveSuggestion]:
        rng = random.Random(self.seed)
        moves = generate_legal_moves(state, state.active_color)
        rng.shuffle(moves)
        return [
            MoveSuggestion(move=move, score=0.0, rationale="Random legal baseline.", visits=0)
            for move in moves[:top_k]
        ]


@dataclass
class LargestPieceGreedyAgent(BaseAgent):
    name: str = "largest-piece-greedy"

    def suggest(self, state: BoardState, top_k: int = 1) -> list[MoveSuggestion]:
        ranked = rank_moves(state, state.active_color, state.active_color)
        ranked.sort(
            key=lambda suggestion: (PIECE_SIZES[suggestion.move.piece_id], suggestion.score),
            reverse=True,
        )
        for suggestion in ranked[:top_k]:
            suggestion.rationale = "Greedy size-first baseline with heuristic tie-breaks."
        return ranked[:top_k]


@dataclass
class MobilityHeuristicAgent(BaseAgent):
    name: str = "mobility-heuristic"

    def suggest(self, state: BoardState, top_k: int = 1) -> list[MoveSuggestion]:
        ranked = rank_moves(state, state.active_color, state.active_color)
        for suggestion in ranked[:top_k]:
            suggestion.rationale = "Heuristic baseline optimized for mobility, corners, and score."
        return ranked[:top_k]

