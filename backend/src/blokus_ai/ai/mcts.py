from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from blokus_ai.ai.evaluation import (
    describe_move,
    heuristic_value,
    rank_moves,
)
from blokus_ai.ai.types import AgentDecision
from blokus_ai.engine.game import (
    MAX_ABS_TEAM_MARGIN,
    apply_move,
    is_terminal,
    normalized_score_margin_for_color,
)
from blokus_ai.engine.models import BoardState, GameVariant, Move, MoveSuggestion, PlayerColor
from blokus_ai.training.encoding import (
    action_priors_from_logits,
    decode_action,
    encode_action,
    encode_state_tensors,
    legal_action_indices,
    perspective_sign,
)
from blokus_ai.training.model import load_policy_value_checkpoint


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


@dataclass
class SearchEdge:
    move: Move
    prior: float
    child: "SearchNode"


@dataclass
class SearchNode:
    state: BoardState
    root_color: PlayerColor
    incoming_move: Optional[Move] = None
    visits: int = 0
    total_value: float = 0.0
    children: dict[str, SearchEdge] = field(default_factory=dict)
    candidate_entries: list[tuple[Move, float]] = field(default_factory=list)
    expanded_count: int = 0

    def candidate_budget(self, base: int) -> int:
        return min(len(self.candidate_entries), max(1, base + int(math.sqrt(max(self.visits, 1)))))

    def can_expand(self, base: int) -> bool:
        return self.expanded_count < self.candidate_budget(base)


@dataclass
class SearchSummary:
    suggestions: list[MoveSuggestion]
    visit_counts_by_action: dict[int, int]
    root_value: float
    chosen_move: Optional[Move]
    diagnostics: dict[str, Any]


class MCTSAgent:
    name = "heuristic-mcts"

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

    def search(self, state: BoardState, top_k: int = 5) -> SearchSummary:
        root = SearchNode(state=state, root_color=state.active_color)
        root.candidate_entries = self._candidate_entries(state, root.root_color)
        if not root.candidate_entries:
            return SearchSummary(
                suggestions=[],
                visit_counts_by_action={},
                root_value=0.0,
                chosen_move=None,
                diagnostics={"agent_id": self.name},
            )

        for _ in range(self.simulations):
            path = [root]
            node = root
            while True:
                if is_terminal(node.state):
                    value = normalized_score_margin_for_color(node.state, root.root_color)
                    break

                if not node.candidate_entries:
                    node.candidate_entries = self._candidate_entries(node.state, root.root_color)

                if node.can_expand(base=4):
                    move, prior = node.candidate_entries[node.expanded_count]
                    node.expanded_count += 1
                    child_state = apply_move(node.state, move)
                    child = SearchNode(
                        state=child_state,
                        root_color=root.root_color,
                        incoming_move=move,
                    )
                    node.children[self._move_key(move)] = SearchEdge(move=move, prior=prior, child=child)
                    path.append(child)
                    value = self._rollout(child.state, root.root_color)
                    break

                if node.children:
                    selected_edge = self._select_child(node)
                    node = selected_edge.child
                    path.append(node)
                    continue

                value = self._rollout(node.state, root.root_color)
                break

            for visited in path:
                visited.visits += 1
                visited.total_value += value

        suggestions: list[MoveSuggestion] = []
        visit_counts_by_action: dict[int, int] = {}
        for edge in root.children.values():
            child = edge.child
            if child.incoming_move is None or child.visits == 0:
                continue
            average_value = child.total_value / child.visits
            visit_counts_by_action[encode_action(edge.move)] = child.visits
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

        suggestions.sort(key=lambda suggestion: (suggestion.visits, suggestion.score), reverse=True)
        chosen_move = suggestions[0].move if suggestions else None
        root_value = root.total_value / max(root.visits, 1)
        return SearchSummary(
            suggestions=suggestions[:top_k],
            visit_counts_by_action=visit_counts_by_action,
            root_value=root_value,
            chosen_move=chosen_move,
            diagnostics={
                "agent_id": self.name,
                "simulations": self.simulations,
                "expanded_children": len(root.children),
                "root_value": root_value,
            },
        )

    def suggest(self, state: BoardState, top_k: int = 5) -> list[MoveSuggestion]:
        return self.search(state, top_k=top_k).suggestions

    def select_move(self, state: BoardState, top_k: int = 1) -> AgentDecision:
        summary = self.search(state, top_k=max(top_k, 3))
        if summary.chosen_move is not None:
            summary.diagnostics["selected_action_index"] = encode_action(summary.chosen_move)
        summary.diagnostics["visit_counts_by_action"] = summary.visit_counts_by_action
        return AgentDecision(
            chosen_move=summary.chosen_move,
            suggestions=summary.suggestions[:top_k],
            diagnostics=summary.diagnostics,
        )

    def _move_key(self, move: Move) -> str:
        return (
            f"{move.piece_id}:{move.rotation}:{int(move.reflection)}:"
            f"{move.anchor_cell.row}:{move.anchor_cell.col}:{int(move.is_pass)}"
        )

    def _candidate_entries(
        self,
        state: BoardState,
        root_color: PlayerColor,
    ) -> list[tuple[Move, float]]:
        ranked = rank_moves(
            state,
            state.active_color,
            root_color,
            top_k=self.candidate_limit,
        )
        priors = _softmax([suggestion.score for suggestion in ranked])
        return [
            (suggestion.move, prior)
            for suggestion, prior in zip(ranked, priors)
        ]

    def _select_child(self, node: SearchNode) -> SearchEdge:
        assert node.children
        total_visits = max(node.visits, 1)

        def puct(edge: SearchEdge) -> float:
            child = edge.child
            exploitation = child.total_value / max(child.visits, 1)
            exploration = self.exploration_weight * edge.prior * math.sqrt(total_visits) / (
                1 + child.visits
            )
            return exploitation + exploration

        return max(node.children.values(), key=puct)

    def _rollout(self, state: BoardState, root_color: PlayerColor) -> float:
        rollout_state = state
        for _ in range(self.rollout_depth):
            if is_terminal(rollout_state):
                return normalized_score_margin_for_color(rollout_state, root_color)
            candidates = self._candidate_entries(rollout_state, root_color)
            if not candidates:
                break
            rollout_state = apply_move(rollout_state, candidates[0][0])
        return self._evaluate_leaf(rollout_state, root_color)

    def _evaluate_leaf(self, state: BoardState, root_color: PlayerColor) -> float:
        if is_terminal(state):
            return normalized_score_margin_for_color(state, root_color)
        scaled = heuristic_value(state, root_color) / MAX_ABS_TEAM_MARGIN
        return max(-1.0, min(1.0, scaled))


class PolicyGuidedMCTSAgent(MCTSAgent):
    name = "policy-mcts"

    def __init__(
        self,
        simulations: int = 96,
        candidate_limit: int = 24,
        rollout_depth: int = 8,
        exploration_weight: float = 1.15,
        checkpoint_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            simulations=simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
            exploration_weight=exploration_weight,
        )
        self.checkpoint_id = checkpoint_id
        self.loaded_checkpoint = load_policy_value_checkpoint(checkpoint_id)
        self.fallback_agent = MCTSAgent(
            simulations=simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
            exploration_weight=exploration_weight,
        )

    def search(self, state: BoardState, top_k: int = 5) -> SearchSummary:
        if self.loaded_checkpoint is None or state.variant != GameVariant.PAIRED_2:
            summary = self.fallback_agent.search(state, top_k=top_k)
            summary.diagnostics["requested_agent"] = self.name
            summary.diagnostics["checkpoint_id"] = self.checkpoint_id
            summary.diagnostics["fallback_agent"] = self.fallback_agent.name
            return summary

        summary = super().search(state, top_k=top_k)
        summary.diagnostics["checkpoint_id"] = self.loaded_checkpoint.checkpoint_id
        return summary

    def _candidate_entries(
        self,
        state: BoardState,
        root_color: PlayerColor,
    ) -> list[tuple[Move, float]]:
        if self.loaded_checkpoint is None or state.variant != GameVariant.PAIRED_2:
            return self.fallback_agent._candidate_entries(state, root_color)

        import torch

        legal_indices = legal_action_indices(state)
        if not legal_indices:
            return []

        spatial_inputs, metadata_inputs = encode_state_tensors(state)
        with torch.no_grad():
            policy_logits, _ = self.loaded_checkpoint.model(spatial_inputs, metadata_inputs)
        priors_by_action = action_priors_from_logits(policy_logits[0], legal_indices)
        ranked_actions = sorted(
            priors_by_action.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.candidate_limit]
        return [
            (decode_action(action_index, state.active_color), prior)
            for action_index, prior in ranked_actions
        ]

    def _evaluate_leaf(self, state: BoardState, root_color: PlayerColor) -> float:
        if self.loaded_checkpoint is None or state.variant != GameVariant.PAIRED_2:
            return self.fallback_agent._evaluate_leaf(state, root_color)
        if is_terminal(state):
            return normalized_score_margin_for_color(state, root_color)

        import torch

        spatial_inputs, metadata_inputs = encode_state_tensors(state)
        with torch.no_grad():
            _, values = self.loaded_checkpoint.model(spatial_inputs, metadata_inputs)
        return float(values.item()) * perspective_sign(state, root_color)
