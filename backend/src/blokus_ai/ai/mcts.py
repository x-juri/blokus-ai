from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from blokus_ai.ai.evaluation import describe_move, group_score_margin, heuristic_value, rank_moves
from blokus_ai.engine.game import apply_move, apply_pass, generate_legal_moves, is_terminal
from blokus_ai.engine.models import BoardState, Move, MoveSuggestion, PlayerColor


@dataclass
class SearchNode:
    state: BoardState
    root_color: PlayerColor
    incoming_move: Optional[Move] = None
    visits: int = 0
    total_value: float = 0.0
    children: dict[str, "SearchNode"] = field(default_factory=dict)
    ranked_candidates: list[Move] = field(default_factory=list)
    expanded_count: int = 0

    def candidate_budget(self, base: int) -> int:
        return min(len(self.ranked_candidates), max(1, base + int(math.sqrt(max(self.visits, 1)))))

    def can_expand(self, base: int) -> bool:
        return self.expanded_count < self.candidate_budget(base)


class MCTSAgent:
    name = "progressive-mcts"

    def __init__(
        self,
        simulations: int = 96,
        candidate_limit: int = 24,
        rollout_depth: int = 8,
        exploration_weight: float = 1.15,
    ) -> None:
        self.simulations = simulations
        self.candidate_limit = candidate_limit
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight

    def suggest(self, state: BoardState, top_k: int = 5) -> list[MoveSuggestion]:
        legal_moves = generate_legal_moves(state, state.active_color)
        if not legal_moves:
            return []

        root = SearchNode(
            state=state,
            root_color=state.active_color,
            ranked_candidates=[
                suggestion.move
                for suggestion in rank_moves(
                    state,
                    state.active_color,
                    state.active_color,
                    top_k=self.candidate_limit,
                )
            ],
        )

        for _ in range(self.simulations):
            path = [root]
            node = root
            while True:
                if is_terminal(node.state):
                    value = group_score_margin(node.state, root.root_color)
                    break
                if not node.ranked_candidates:
                    node.ranked_candidates = [
                        suggestion.move
                        for suggestion in rank_moves(
                            node.state,
                            node.state.active_color,
                            root.root_color,
                            top_k=self.candidate_limit,
                        )
                    ]
                if node.can_expand(base=4):
                    move = node.ranked_candidates[node.expanded_count]
                    node.expanded_count += 1
                    child_state = apply_move(node.state, move)
                    child_key = self._move_key(move)
                    child = SearchNode(state=child_state, root_color=root.root_color, incoming_move=move)
                    node.children[child_key] = child
                    path.append(child)
                    value = self._rollout(child.state, root.root_color)
                    break
                if node.children:
                    node = self._select_child(node)
                    path.append(node)
                    continue
                value = self._rollout(node.state, root.root_color)
                break

            for visited in path:
                visited.visits += 1
                visited.total_value += value

        suggestions: list[MoveSuggestion] = []
        for child in root.children.values():
            if child.incoming_move is None or child.visits == 0:
                continue
            average_value = child.total_value / child.visits
            rationale = describe_move(state, child.incoming_move)
            rationale += f" Search value {average_value:.2f} over {child.visits} visits."
            suggestions.append(
                MoveSuggestion(
                    move=child.incoming_move,
                    score=average_value,
                    rationale=rationale,
                    visits=child.visits,
                )
            )
        suggestions.sort(key=lambda suggestion: (suggestion.score, suggestion.visits), reverse=True)
        return suggestions[:top_k]

    def _move_key(self, move: Move) -> str:
        return (
            f"{move.piece_id}:{move.rotation}:{int(move.reflection)}:"
            f"{move.anchor_cell.row}:{move.anchor_cell.col}"
        )

    def _select_child(self, node: SearchNode) -> SearchNode:
        assert node.children
        total_visits = max(node.visits, 1)

        def uct(child: SearchNode) -> float:
            exploitation = child.total_value / max(child.visits, 1)
            exploration = self.exploration_weight * math.sqrt(
                math.log(total_visits + 1) / max(child.visits, 1)
            )
            return exploitation + exploration

        return max(node.children.values(), key=uct)

    def _rollout(self, state: BoardState, root_color: PlayerColor) -> float:
        rollout_state = state
        for _ in range(self.rollout_depth):
            if is_terminal(rollout_state):
                return group_score_margin(rollout_state, root_color)
            ranked = rank_moves(
                rollout_state,
                rollout_state.active_color,
                root_color,
                top_k=6,
            )
            if not ranked:
                rollout_state = apply_pass(rollout_state, rollout_state.active_color)
                continue
            rollout_state = apply_move(rollout_state, ranked[0].move)
        return heuristic_value(rollout_state, root_color)

