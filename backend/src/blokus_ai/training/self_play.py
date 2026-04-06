from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from blokus_ai.ai.agents import build_agent
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
    progress_every: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    config = config or GameConfig(variant=GameVariant.PAIRED_2)
    agent_config = agent_config or AgentConfig(agent_id="heuristic-mcts", simulations=64)
    agent = build_agent(agent_config)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for game_index in range(games):
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
                            "agent_id": agent_config.agent_id,
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

                decision = agent.select_move(state, top_k=3)
                if decision.chosen_move is None:
                    break

                legal_indices = legal_action_indices(state)
                chosen_action_index = encode_action(decision.chosen_move)
                trace.append(
                    {
                        "game_index": game_index,
                        "ply_index": ply_index,
                        "agent_id": agent_config.agent_id,
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
                and progress_every > 0
                and (completed_games % progress_every == 0 or completed_games == games)
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main() -> None:
    parser = build_self_play_arg_parser()
    args = parser.parse_args()
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
            seed=args.seed,
        ),
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
