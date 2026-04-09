from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Callable, Optional

from blokus_ai.ai.agents import build_agent
from blokus_ai.engine.game import (
    apply_move,
    create_initial_state,
    has_legal_move,
    is_terminal,
    legal_moves_or_pass,
    owner_group,
    pass_move_for_color,
    result,
    score_margin_for_color,
)
from blokus_ai.engine.models import AgentConfig, GameConfig, GameVariant, PlayerColor


@dataclass
class TournamentRow:
    matchup: str
    games: int
    average_margin: float
    wins: int


def _derive_seed(base_seed: Optional[int], game_index: int, salt: int = 0) -> Optional[int]:
    if base_seed is None:
        return None
    return base_seed + salt * 10_000 + game_index


def _config_with_seed(agent_config: AgentConfig, seed: Optional[int]) -> AgentConfig:
    if seed is None:
        return agent_config
    return agent_config.model_copy(update={"seed": seed})


def build_seeded_opening_state(
    config: GameConfig | None = None,
    opening_seed: Optional[int] = None,
    opening_plies: int = 0,
):
    state = create_initial_state(config or GameConfig(variant=GameVariant.PAIRED_2))
    if opening_seed is None or opening_plies <= 0:
        return state

    opening_rng = random.Random(opening_seed)
    for _ in range(opening_plies):
        if is_terminal(state):
            break
        moves = legal_moves_or_pass(state)
        if not moves:
            break
        state = apply_move(state, opening_rng.choice(moves))
    return state


def play_game(
    player_a_agent_config: AgentConfig,
    player_b_agent_config: AgentConfig,
    config: GameConfig | None = None,
    max_turns: int = 512,
    opening_seed: Optional[int] = None,
    opening_plies: int = 0,
) -> tuple[float, str | None]:
    config = config or GameConfig(variant=GameVariant.PAIRED_2)
    state = build_seeded_opening_state(
        config=config,
        opening_seed=opening_seed,
        opening_plies=opening_plies,
    )
    agents = {
        "player_a": build_agent(player_a_agent_config),
        "player_b": build_agent(player_b_agent_config),
    }

    for _ in range(max_turns):
        if is_terminal(state):
            break
        if not has_legal_move(state):
            state = apply_move(state, pass_move_for_color(state.active_color))
            continue

        group_name = owner_group(state, state.active_color)
        decision = agents[group_name].select_move(state, top_k=1)
        if decision.chosen_move is None:
            break
        state = apply_move(state, decision.chosen_move)

    margin = score_margin_for_color(state, PlayerColor.BLUE)
    return margin, result(state).winner_group


def run_paired_tournament(
    agent_one: AgentConfig,
    agent_two: AgentConfig,
    games: int = 4,
    max_turns: int = 512,
    base_seed: int = 7,
    opening_plies: int = 4,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[TournamentRow]:
    rows: list[TournamentRow] = []
    for seat_name in ("player_a", "player_b"):
        margins: list[float] = []
        wins = 0
        for game_index in range(games):
            if progress_callback is not None:
                progress_callback(seat_name, game_index + 1, games)
            agent_one_seed = _derive_seed(base_seed, game_index, salt=1)
            agent_two_seed = _derive_seed(base_seed, game_index, salt=2)
            opening_seed = _derive_seed(base_seed, game_index, salt=3)
            if seat_name == "player_a":
                player_a_config = _config_with_seed(agent_one, agent_one_seed)
                player_b_config = _config_with_seed(agent_two, agent_two_seed)
            else:
                player_a_config = _config_with_seed(agent_two, agent_two_seed)
                player_b_config = _config_with_seed(agent_one, agent_one_seed)
            margin, winner = play_game(
                player_a_config,
                player_b_config,
                max_turns=max_turns,
                opening_seed=opening_seed,
                opening_plies=opening_plies,
            )
            normalized_margin = margin if seat_name == "player_a" else -margin
            margins.append(normalized_margin)
            if winner == seat_name:
                wins += 1
        rows.append(
            TournamentRow(
                matchup=f"{agent_one.agent_id} vs {agent_two.agent_id} ({seat_name})",
                games=games,
                average_margin=sum(margins) / len(margins),
                wins=wins,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired-2 Blokus tournaments.")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=160)
    parser.add_argument(
        "--agent-one",
        choices=["heuristic-mcts", "policy-mcts", "mobility-heuristic", "random-legal"],
        default="heuristic-mcts",
    )
    parser.add_argument(
        "--agent-two",
        choices=["heuristic-mcts", "policy-mcts", "mobility-heuristic", "random-legal"],
        default="mobility-heuristic",
    )
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=6)
    parser.add_argument("--rollout-depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print(
        "[evaluation] starting paired tournament "
        f"{args.agent_one} vs {args.agent_two} "
        f"(games={args.games}, max_turns={args.max_turns}, simulations={args.simulations}, "
        f"candidate_limit={args.candidate_limit}, rollout_depth={args.rollout_depth}, "
        f"seed={args.seed}, opening_plies={args.opening_plies})",
        flush=True,
    )

    rows = run_paired_tournament(
        AgentConfig(
            agent_id=args.agent_one,
            checkpoint_id=args.checkpoint_id,
            simulations=args.simulations,
            candidate_limit=args.candidate_limit,
            rollout_depth=args.rollout_depth,
        ),
        AgentConfig(
            agent_id=args.agent_two,
            checkpoint_id=args.checkpoint_id,
            simulations=args.simulations,
            candidate_limit=args.candidate_limit,
            rollout_depth=args.rollout_depth,
        ),
        games=args.games,
        max_turns=args.max_turns,
        base_seed=args.seed,
        opening_plies=args.opening_plies,
        progress_callback=lambda seat_name, game_index, total_games: print(
            f"[evaluation] {seat_name} game {game_index}/{total_games}",
            flush=True,
        ),
    )
    if args.json:
        print(json.dumps([row.__dict__ for row in rows], indent=2))
        return

    print("matchup\tgames\taverage_margin\twins")
    for row in rows:
        print(f"{row.matchup}\t{row.games}\t{row.average_margin:.2f}\t{row.wins}")


if __name__ == "__main__":
    main()
