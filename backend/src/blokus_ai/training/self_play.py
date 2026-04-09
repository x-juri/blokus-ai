from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from blokus_ai.ai.agents import build_agent
from blokus_ai.ai.mcts import MCTSAgent
from blokus_ai.engine.game import (
    apply_move,
    create_initial_state,
    has_legal_move,
    is_terminal,
    normalized_score_margin_for_color,
    pass_move_for_color,
    result,
)
from blokus_ai.engine.models import AgentConfig, GameConfig, GameVariant
from blokus_ai.training.encoding import PASS_ACTION_INDEX, encode_action, legal_action_indices


def _derive_game_seed(base_seed: Optional[int], game_index: int) -> Optional[int]:
    if base_seed is None:
        return None
    return base_seed + game_index


def _finalize_trace(trace: list[dict], final_state) -> Iterable[dict]:
    summary = result(final_state)
    for record in trace:
        perspective_color = record["perspective_color"]
        record["final_group_scores"] = summary.group_scores
        record["final_value_target"] = normalized_score_margin_for_color(
            final_state,
            perspective_color,
        )
        yield record


def generate_self_play_records(
    games: int,
    output_path: Union[str, Path],
    config: Optional[GameConfig] = None,
    agent_config: Optional[AgentConfig] = None,
    sampling_moves: Optional[int] = None,
    sampling_temperature: Optional[float] = None,
    progress_every: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    config = config or GameConfig(variant=GameVariant.PAIRED_2)
    agent_config = agent_config or AgentConfig(agent_id="heuristic-mcts", simulations=64)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    effective_progress_every = progress_every if games >= progress_every else 1

    with output_file.open("w", encoding="utf-8") as handle:
        for game_index in range(games):
            game_seed = _derive_game_seed(agent_config.seed, game_index)
            game_agent_config = agent_config.model_copy(update={"seed": game_seed})
            agent = build_agent(game_agent_config)
            state = create_initial_state(config)
            trace: list[dict] = []
            ply_index = 0
            while not is_terminal(state):
                if not has_legal_move(state):
                    pass_move = pass_move_for_color(state.active_color)
                    trace.append(
                        {
                            "game_index": game_index,
                            "ply_index": ply_index,
                            "agent_id": game_agent_config.agent_id,
                            "game_seed": game_seed,
                            "state": state.model_dump(mode="json"),
                            "perspective_color": state.active_color,
                            "legal_action_indices": [PASS_ACTION_INDEX],
                            "visit_counts_by_action": {str(PASS_ACTION_INDEX): 1},
                            "chosen_action_index": PASS_ACTION_INDEX,
                            "move": pass_move.model_dump(mode="json"),
                        }
                    )
                    state = apply_move(state, pass_move)
                    ply_index += 1
                    continue

                if isinstance(agent, MCTSAgent):
                    should_sample = (
                        sampling_moves is not None
                        and sampling_moves > 0
                        and ply_index < sampling_moves
                    )
                    decision = agent.select_move(
                        state,
                        top_k=3,
                        sample_from_visit_distribution=should_sample,
                        sampling_temperature=sampling_temperature,
                        add_root_dirichlet_noise=bool(
                            (game_agent_config.root_dirichlet_alpha or 0.0) > 0.0
                            and (game_agent_config.root_exploration_fraction or 0.0) > 0.0
                        ),
                    )
                else:
                    decision = agent.select_move(state, top_k=3)
                if decision.chosen_move is None:
                    break

                legal_indices = legal_action_indices(state)
                chosen_action_index = encode_action(decision.chosen_move)
                trace.append(
                    {
                        "game_index": game_index,
                        "ply_index": ply_index,
                        "agent_id": game_agent_config.agent_id,
                        "game_seed": game_seed,
                        "state": state.model_dump(mode="json"),
                        "perspective_color": state.active_color,
                        "legal_action_indices": legal_indices,
                        "visit_counts_by_action": {
                            str(action_index): visits
                            for action_index, visits in decision.diagnostics.get(
                                "visit_counts_by_action",
                                {chosen_action_index: 1},
                            ).items()
                        },
                        "chosen_action_index": chosen_action_index,
                        "move": decision.chosen_move.model_dump(mode="json"),
                        "diagnostics": decision.diagnostics,
                    }
                )
                state = apply_move(state, decision.chosen_move)
                ply_index += 1

            for record in _finalize_trace(trace, state):
                record["perspective_color"] = record["perspective_color"].value
                handle.write(json.dumps(record) + "\n")

            completed_games = game_index + 1
            if (
                progress_callback is not None
                and effective_progress_every > 0
                and (
                    completed_games % effective_progress_every == 0
                    or completed_games == games
                )
            ):
                progress_callback(completed_games, games)

    return output_file


def build_self_play_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paired-2 Blokus self-play traces.")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/self_play/paired2-bootstrap.jsonl"),
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in GameVariant],
        default=GameVariant.PAIRED_2.value,
    )
    parser.add_argument("--agent-id", default="heuristic-mcts")
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--rollout-depth", type=int, default=3)
    parser.add_argument("--exploration-weight", type=float, default=1.15)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-moves", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main() -> None:
    parser = build_self_play_arg_parser()
    args = parser.parse_args()
    print(
        "[self-play] starting "
        f"{args.games} games with {args.agent_id} "
        f"(simulations={args.simulations}, candidate_limit={args.candidate_limit}, "
        f"rollout_depth={args.rollout_depth}, root_dirichlet_alpha={args.root_dirichlet_alpha}, "
        f"root_exploration_fraction={args.root_exploration_fraction}, "
        f"sampling_temperature={args.sampling_temperature}, sampling_moves={args.sampling_moves})",
        flush=True,
    )
    output_path = generate_self_play_records(
        games=args.games,
        output_path=args.output,
        config=GameConfig(variant=GameVariant(args.variant)),
        agent_config=AgentConfig(
            agent_id=args.agent_id,
            checkpoint_id=args.checkpoint_id,
            simulations=args.simulations,
            candidate_limit=args.candidate_limit,
            rollout_depth=args.rollout_depth,
            exploration_weight=args.exploration_weight,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            root_exploration_fraction=args.root_exploration_fraction,
            sampling_temperature=args.sampling_temperature,
            sampling_moves=args.sampling_moves,
            seed=args.seed,
        ),
        sampling_moves=args.sampling_moves,
        sampling_temperature=args.sampling_temperature,
        progress_every=args.progress_every,
        progress_callback=lambda completed, total: print(
            f"[self-play] completed {completed}/{total} games",
            flush=True,
        ),
    )
    print(
        json.dumps(
            {
                "games": args.games,
                "output": str(output_path),
                "agent_id": args.agent_id,
                "variant": args.variant,
                "root_dirichlet_alpha": args.root_dirichlet_alpha,
                "root_exploration_fraction": args.root_exploration_fraction,
                "sampling_temperature": args.sampling_temperature,
                "sampling_moves": args.sampling_moves,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
