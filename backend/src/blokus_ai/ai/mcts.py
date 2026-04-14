from __future__ import annotations

import math
import random
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
    apply_legal_move_unchecked,
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


def _sample_dirichlet(alpha: float, size: int, rng: random.Random) -> list[float]:
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(samples)
    if total <= 0.0:
        return [1.0 / size] * size
    return [value / total for value in samples]


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _move_key(move: Move) -> str:
    return (
        f"{move.piece_id}:{move.rotation}:{int(move.reflection)}:"
        f"{move.anchor_cell.row}:{move.anchor_cell.col}:{int(move.is_pass)}"
    )


def _normalize_candidate_entries(entries: list[tuple[Move, float]]) -> list[tuple[Move, float]]:
    if not entries:
        return []
    total = sum(max(0.0, prior) for _, prior in entries)
    if total <= 0.0:
        uniform = 1.0 / len(entries)
        return [(move, uniform) for move, _ in entries]
    return [(move, max(0.0, prior) / total) for move, prior in entries]


def _apply_root_dirichlet_noise(
    entries: list[tuple[Move, float]],
    alpha: float,
    exploration_fraction: float,
    rng: random.Random,
) -> list[tuple[Move, float]]:
    normalized_entries = _normalize_candidate_entries(entries)
    if (
        not normalized_entries
        or alpha <= 0.0
        or exploration_fraction <= 0.0
    ):
        return normalized_entries

    dirichlet_noise = _sample_dirichlet(alpha, len(normalized_entries), rng)
    blended_entries = [
        (
            move,
            (1.0 - exploration_fraction) * prior + exploration_fraction * noise,
        )
        for (move, prior), noise in zip(normalized_entries, dirichlet_noise)
    ]
    return _normalize_candidate_entries(blended_entries)


def _sample_action_index_from_visit_counts(
    visit_counts_by_action: dict[int, int],
    temperature: float,
    rng: random.Random,
) -> Optional[int]:
    if not visit_counts_by_action:
        return None
    if temperature <= 1e-6:
        return max(
            visit_counts_by_action.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

    weighted_actions = []
    total_weight = 0.0
    exponent = 1.0 / temperature
    for action_index, visits in visit_counts_by_action.items():
        weight = float(visits) ** exponent
        if weight <= 0.0:
            continue
        weighted_actions.append((action_index, weight))
        total_weight += weight

    if total_weight <= 0.0:
        return max(
            visit_counts_by_action.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

    threshold = rng.random() * total_weight
    cumulative = 0.0
    for action_index, weight in weighted_actions:
        cumulative += weight
        if cumulative >= threshold:
            return action_index
    return weighted_actions[-1][0]


def _blend_candidate_entries(
    heuristic_entries: list[tuple[Move, float]],
    policy_entries: list[tuple[Move, float]],
    candidate_limit: int,
    heuristic_floor_fraction: float,
    heuristic_prior_weight: float,
    policy_prior_weight: float,
) -> list[tuple[Move, float]]:
    if candidate_limit <= 0:
        return []
    if not heuristic_entries:
        return _normalize_candidate_entries(policy_entries[:candidate_limit])
    if not policy_entries:
        return _normalize_candidate_entries(heuristic_entries[:candidate_limit])

    heuristic_subset = _normalize_candidate_entries(heuristic_entries[:candidate_limit])
    policy_subset = _normalize_candidate_entries(policy_entries[:candidate_limit])
    heuristic_lookup = {_move_key(move): (move, prior) for move, prior in heuristic_subset}
    policy_lookup = {_move_key(move): (move, prior) for move, prior in policy_subset}

    combined_scores: dict[str, float] = {}
    for move_key in set(heuristic_lookup) | set(policy_lookup):
        heuristic_prior = heuristic_lookup.get(move_key, (None, 0.0))[1]
        policy_prior = policy_lookup.get(move_key, (None, 0.0))[1]
        combined_scores[move_key] = (
            heuristic_prior_weight * heuristic_prior
            + policy_prior_weight * policy_prior
        )

    heuristic_floor = min(
        len(heuristic_subset),
        max(1, math.ceil(candidate_limit * heuristic_floor_fraction)),
    )
    selected_keys = [_move_key(move) for move, _ in heuristic_subset[:heuristic_floor]]

    for move_key, _ in sorted(
        combined_scores.items(),
        key=lambda item: (
            item[1],
            heuristic_lookup.get(item[0], (None, 0.0))[1],
            policy_lookup.get(item[0], (None, 0.0))[1],
        ),
        reverse=True,
    ):
        if move_key in selected_keys:
            continue
        selected_keys.append(move_key)
        if len(selected_keys) >= candidate_limit:
            break

    for entries in (heuristic_subset, policy_subset):
        for move, _ in entries:
            move_key = _move_key(move)
            if move_key in selected_keys:
                continue
            selected_keys.append(move_key)
            if len(selected_keys) >= candidate_limit:
                break
        if len(selected_keys) >= candidate_limit:
            break

    blended_entries = [
        (
            heuristic_lookup[move_key][0] if move_key in heuristic_lookup else policy_lookup[move_key][0],
            combined_scores.get(move_key, 0.0),
        )
        for move_key in selected_keys[:candidate_limit]
    ]
    return _normalize_candidate_entries(blended_entries)


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
        seed: Optional[int] = None,
        root_dirichlet_alpha: float = 0.0,
        root_exploration_fraction: float = 0.0,
    ) -> None:
        self.simulations = simulations
        self.candidate_limit = candidate_limit
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.seed = seed
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.rng = random.Random(seed)

    def search(
        self,
        state: BoardState,
        top_k: int = 5,
        add_root_dirichlet_noise: bool = False,
    ) -> SearchSummary:
        root = SearchNode(state=state, root_color=state.active_color)
        root.candidate_entries = self._candidate_entries(state, root.root_color)
        if add_root_dirichlet_noise:
            root.candidate_entries = _apply_root_dirichlet_noise(
                root.candidate_entries,
                alpha=self.root_dirichlet_alpha,
                exploration_fraction=self.root_exploration_fraction,
                rng=self.rng,
            )
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
                    child_state = apply_legal_move_unchecked(node.state, move)
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
                "root_dirichlet_noise_applied": add_root_dirichlet_noise,
                "root_dirichlet_alpha": self.root_dirichlet_alpha if add_root_dirichlet_noise else 0.0,
                "root_exploration_fraction": (
                    self.root_exploration_fraction if add_root_dirichlet_noise else 0.0
                ),
            },
        )

    def suggest(self, state: BoardState, top_k: int = 5) -> list[MoveSuggestion]:
        return self.search(state, top_k=top_k).suggestions

    def select_move(
        self,
        state: BoardState,
        top_k: int = 1,
        sample_from_visit_distribution: bool = False,
        sampling_temperature: Optional[float] = None,
        add_root_dirichlet_noise: bool = False,
    ) -> AgentDecision:
        summary = self.search(
            state,
            top_k=max(top_k, 3),
            add_root_dirichlet_noise=add_root_dirichlet_noise,
        )
        sampled_action_index = None
        if sample_from_visit_distribution and summary.visit_counts_by_action:
            sampled_action_index = _sample_action_index_from_visit_counts(
                summary.visit_counts_by_action,
                temperature=sampling_temperature or 1.0,
                rng=self.rng,
            )
            if sampled_action_index is not None:
                summary.chosen_move = decode_action(sampled_action_index, state.active_color)
        if summary.chosen_move is not None:
            summary.diagnostics["selected_action_index"] = encode_action(summary.chosen_move)
        if sampled_action_index is not None:
            summary.diagnostics["sampled_action_index"] = sampled_action_index
            summary.diagnostics["sampled_from_visit_distribution"] = True
            summary.diagnostics["sampling_temperature"] = sampling_temperature or 1.0
        else:
            summary.diagnostics["sampled_from_visit_distribution"] = False
        summary.diagnostics["visit_counts_by_action"] = summary.visit_counts_by_action
        return AgentDecision(
            chosen_move=summary.chosen_move,
            suggestions=summary.suggestions[:top_k],
            diagnostics=summary.diagnostics,
        )

    def _move_key(self, move: Move) -> str:
        return _move_key(move)

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
            include_rationales=False,
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
            rollout_state = apply_legal_move_unchecked(rollout_state, candidates[0][0])
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
        seed: Optional[int] = None,
        root_dirichlet_alpha: float = 0.0,
        root_exploration_fraction: float = 0.0,
    ) -> None:
        super().__init__(
            simulations=simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
            exploration_weight=exploration_weight,
            seed=seed,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
        )
        self.checkpoint_id = checkpoint_id
        self.loaded_checkpoint = load_policy_value_checkpoint(checkpoint_id)
        self.heuristic_floor_fraction = 0.5
        self.heuristic_prior_weight = 0.65
        self.policy_prior_weight = 0.35
        self.heuristic_value_weight = 0.7
        self.model_value_weight = 0.3
        self.fallback_agent = MCTSAgent(
            simulations=simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
            exploration_weight=exploration_weight,
            seed=seed,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
        )

    def search(
        self,
        state: BoardState,
        top_k: int = 5,
        add_root_dirichlet_noise: bool = False,
    ) -> SearchSummary:
        if self.loaded_checkpoint is None or state.variant != GameVariant.PAIRED_2:
            summary = self.fallback_agent.search(
                state,
                top_k=top_k,
                add_root_dirichlet_noise=add_root_dirichlet_noise,
            )
            summary.diagnostics["requested_agent"] = self.name
            summary.diagnostics["checkpoint_id"] = self.checkpoint_id
            summary.diagnostics["fallback_agent"] = self.fallback_agent.name
            return summary

        summary = super().search(
            state,
            top_k=top_k,
            add_root_dirichlet_noise=add_root_dirichlet_noise,
        )
        summary.diagnostics["checkpoint_id"] = self.loaded_checkpoint.checkpoint_id
        summary.diagnostics["candidate_strategy"] = "hybrid-policy-heuristic"
        summary.diagnostics["heuristic_prior_weight"] = self.heuristic_prior_weight
        summary.diagnostics["policy_prior_weight"] = self.policy_prior_weight
        summary.diagnostics["heuristic_floor_fraction"] = self.heuristic_floor_fraction
        summary.diagnostics["heuristic_value_weight"] = self.heuristic_value_weight
        summary.diagnostics["model_value_weight"] = self.model_value_weight
        return summary

    def _candidate_entries(
        self,
        state: BoardState,
        root_color: PlayerColor,
    ) -> list[tuple[Move, float]]:
        if self.loaded_checkpoint is None or state.variant != GameVariant.PAIRED_2:
            return self.fallback_agent._candidate_entries(state, root_color)

        import torch

        heuristic_entries = self.fallback_agent._candidate_entries(state, root_color)
        legal_indices = legal_action_indices(state)
        if not legal_indices:
            return heuristic_entries

        spatial_inputs, metadata_inputs = encode_state_tensors(state)
        with torch.no_grad():
            policy_logits, _ = self.loaded_checkpoint.model(spatial_inputs, metadata_inputs)
        priors_by_action = action_priors_from_logits(policy_logits[0], legal_indices)
        policy_entries = [
            (decode_action(action_index, state.active_color), prior)
            for action_index, prior in sorted(
            priors_by_action.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.candidate_limit]
        ]
        return _blend_candidate_entries(
            heuristic_entries=heuristic_entries,
            policy_entries=policy_entries,
            candidate_limit=self.candidate_limit,
            heuristic_floor_fraction=self.heuristic_floor_fraction,
            heuristic_prior_weight=self.heuristic_prior_weight,
            policy_prior_weight=self.policy_prior_weight,
        )

    def _evaluate_leaf(self, state: BoardState, root_color: PlayerColor) -> float:
        if self.loaded_checkpoint is None or state.variant != GameVariant.PAIRED_2:
            return self.fallback_agent._evaluate_leaf(state, root_color)
        if is_terminal(state):
            return normalized_score_margin_for_color(state, root_color)

        import torch

        heuristic_estimate = self.fallback_agent._evaluate_leaf(state, root_color)
        spatial_inputs, metadata_inputs = encode_state_tensors(state)
        with torch.no_grad():
            _, values = self.loaded_checkpoint.model(spatial_inputs, metadata_inputs)
        model_estimate = float(values.item()) * perspective_sign(state, root_color)
        return _clamp_unit(
            self.heuristic_value_weight * heuristic_estimate
            + self.model_value_weight * model_estimate
        )
