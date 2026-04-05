from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from blokus_ai.ai.agents import LargestPieceGreedyAgent, MobilityHeuristicAgent, RandomLegalAgent
from blokus_ai.ai.mcts import MCTSAgent
from blokus_ai.engine.game import apply_move, apply_pass, create_initial_state, is_terminal, result
from blokus_ai.engine.models import GameConfig


@dataclass
class BenchmarkRow:
    agent: str
    average_margin: float
    average_group_score: float


def play_partial_game(agent, plies: int) -> tuple[float, float]:
    state = create_initial_state(GameConfig())
    for _ in range(plies):
        if is_terminal(state):
            break
        suggestions = agent.suggest(state, top_k=1)
        if not suggestions:
            state = apply_pass(state)
            continue
        state = apply_move(state, suggestions[0].move)

    summary = result(state)
    blue_score = summary.group_scores.get("blue", summary.group_scores.get("player_a", 0))
    rivals = [
        score
        for group_name, score in summary.group_scores.items()
        if group_name not in {"blue", "player_a"}
    ]
    margin = blue_score - max(rivals, default=0)
    return margin, blue_score


def run_benchmark(
    games: int = 4,
    plies: int = 12,
    mcts_simulations: int = 24,
    candidate_limit: int = 12,
    rollout_depth: int = 4,
) -> list[BenchmarkRow]:
    agents = [
        RandomLegalAgent(seed=11),
        LargestPieceGreedyAgent(),
        MobilityHeuristicAgent(),
        MCTSAgent(
            simulations=mcts_simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
        ),
    ]
    rows: list[BenchmarkRow] = []
    for agent in agents:
        margins: list[float] = []
        scores: list[float] = []
        for _ in range(games):
            margin, score = play_partial_game(agent, plies)
            margins.append(margin)
            scores.append(score)
        rows.append(
            BenchmarkRow(
                agent=agent.name,
                average_margin=sum(margins) / len(margins),
                average_group_score=sum(scores) / len(scores),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight Blokus AI benchmark games.")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--plies", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = run_benchmark(games=args.games, plies=args.plies)
    if args.json:
        print(json.dumps([row.__dict__ for row in rows], indent=2))
        return

    print("agent\taverage_margin\taverage_group_score")
    for row in rows:
        print(f"{row.agent}\t{row.average_margin:.2f}\t{row.average_group_score:.2f}")


if __name__ == "__main__":
    main()
