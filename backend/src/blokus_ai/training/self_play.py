from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

from blokus_ai.ai.agents import build_agent
from blokus_ai.engine.game import apply_move, create_initial_state, has_legal_move, is_terminal, normalized_score_margin_for_color, pass_move_for_color, result
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
) -> None:
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
