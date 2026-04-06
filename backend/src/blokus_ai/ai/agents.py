from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from blokus_ai.ai.evaluation import rank_moves
from blokus_ai.ai.types import AgentDecision
from blokus_ai.engine.game import legal_moves_or_pass
from blokus_ai.engine.models import AgentConfig, BoardState, MoveSuggestion
from blokus_ai.engine.pieces import PIECE_SIZES
from blokus_ai.training.encoding import decode_action, encode_action, encode_state_tensors, legal_action_indices
from blokus_ai.training.model import load_policy_value_checkpoint


class BaseAgent:
    name = "base"

    def suggest(self, state: BoardState, top_k: int = 1) -> list[MoveSuggestion]:
        raise NotImplementedError

    def select_move(self, state: BoardState, top_k: int = 1) -> AgentDecision:
        suggestions = self.suggest(state, top_k=top_k)
        chosen_move = suggestions[0].move if suggestions else None
        return AgentDecision(
            chosen_move=chosen_move,
            suggestions=suggestions,
            diagnostics={"agent_id": self.name},
        )


@dataclass
class RandomLegalAgent(BaseAgent):
    seed: int = 0
    name: str = "random-legal"

    def suggest(self, state: BoardState, top_k: int = 1) -> list[MoveSuggestion]:
        rng = random.Random(self.seed)
        moves = legal_moves_or_pass(state)
        rng.shuffle(moves)
        suggestions = []
        for move in moves[:top_k]:
            rationale = "Random legal baseline." if not move.is_pass else "No legal move available."
            suggestions.append(MoveSuggestion(move=move, score=0.0, rationale=rationale, visits=0))
        return suggestions


@dataclass
class LargestPieceGreedyAgent(BaseAgent):
    name: str = "largest-piece-greedy"

    def suggest(self, state: BoardState, top_k: int = 1) -> list[MoveSuggestion]:
        ranked = rank_moves(state, state.active_color, state.active_color)
        ranked.sort(
            key=lambda suggestion: (PIECE_SIZES.get(suggestion.move.piece_id, 0), suggestion.score),
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


@dataclass
class PolicyValueAgent(BaseAgent):
    checkpoint_id: Optional[str] = None
    fallback_agent: BaseAgent = field(default_factory=MobilityHeuristicAgent)
    name: str = "policy-value"

    def __post_init__(self) -> None:
        self.loaded_checkpoint = load_policy_value_checkpoint(self.checkpoint_id)

    def suggest(self, state: BoardState, top_k: int = 1) -> list[MoveSuggestion]:
        if self.loaded_checkpoint is None:
            return self.fallback_agent.suggest(state, top_k=top_k)

        import torch

        legal_indices = legal_action_indices(state)
        if not legal_indices:
            return []

        model = self.loaded_checkpoint.model
        spatial_inputs, metadata_inputs = encode_state_tensors(state)
        with torch.no_grad():
            policy_logits, values = model(spatial_inputs, metadata_inputs)
        logits = policy_logits[0]
        indexed_scores = [
            (action_index, float(logits[action_index]))
            for action_index in legal_indices
        ]
        indexed_scores.sort(key=lambda item: item[1], reverse=True)
        policy_weights = torch.softmax(
            torch.tensor([score for _, score in indexed_scores], dtype=torch.float32),
            dim=0,
        ).tolist()

        suggestions: list[MoveSuggestion] = []
        scalar_value = float(values.item())
        for (action_index, _), probability in zip(indexed_scores[:top_k], policy_weights[:top_k]):
            move = decode_action(action_index, state.active_color)
            suggestions.append(
                MoveSuggestion(
                    move=move,
                    score=probability,
                    rationale=(
                        f"Checkpoint {self.loaded_checkpoint.checkpoint_id} policy prior "
                        f"with value estimate {scalar_value:.2f}."
                    ),
                    visits=0,
                )
            )
        return suggestions

    def select_move(self, state: BoardState, top_k: int = 1) -> AgentDecision:
        suggestions = self.suggest(state, top_k=top_k)
        diagnostics = {
            "agent_id": self.name,
            "checkpoint_id": self.loaded_checkpoint.checkpoint_id if self.loaded_checkpoint else None,
            "fallback_agent": None if self.loaded_checkpoint else self.fallback_agent.name,
        }
        if suggestions:
            diagnostics["selected_action_index"] = encode_action(suggestions[0].move)
        return AgentDecision(
            chosen_move=suggestions[0].move if suggestions else None,
            suggestions=suggestions,
            diagnostics=diagnostics,
        )


def build_agent(agent_config: Optional[AgentConfig] = None) -> BaseAgent:
    config = agent_config or AgentConfig()
    if config.agent_id == "random-legal":
        return RandomLegalAgent(seed=config.seed or 0)
    if config.agent_id == "mobility-heuristic":
        return MobilityHeuristicAgent()
    if config.agent_id == "policy-mcts":
        from blokus_ai.ai.mcts import PolicyGuidedMCTSAgent

        return PolicyGuidedMCTSAgent(
            simulations=config.simulations or 96,
            candidate_limit=config.candidate_limit or 24,
            rollout_depth=config.rollout_depth or 8,
            exploration_weight=config.exploration_weight or 1.15,
            checkpoint_id=config.checkpoint_id,
        )

    from blokus_ai.ai.mcts import MCTSAgent

    return MCTSAgent(
        simulations=config.simulations or 96,
        candidate_limit=config.candidate_limit or 24,
        rollout_depth=config.rollout_depth or 8,
        exploration_weight=config.exploration_weight or 1.15,
    )
