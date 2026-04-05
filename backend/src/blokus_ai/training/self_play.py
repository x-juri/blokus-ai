from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from blokus_ai.ai.mcts import MCTSAgent
from blokus_ai.engine.game import apply_move, apply_pass, create_initial_state, is_terminal, result
from blokus_ai.engine.models import GameConfig


def generate_self_play_records(
    games: int,
    output_path: Union[str, Path],
    config: Optional[GameConfig] = None,
    simulations: int = 64,
) -> None:
    config = config or GameConfig()
    agent = MCTSAgent(simulations=simulations)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for game_index in range(games):
            state = create_initial_state(config)
            trace: list[dict] = []
            while not is_terminal(state):
                suggestions = agent.suggest(state, top_k=3)
                if not suggestions:
                    state = apply_pass(state)
                    continue
                best = suggestions[0]
                trace.append(
                    {
                        "game_index": game_index,
                        "state": state.model_dump(mode="json"),
                        "move": best.move.model_dump(mode="json"),
                        "score": best.score,
                        "visits": best.visits,
                    }
                )
                state = apply_move(state, best.move)

            summary = result(state)
            for record in trace:
                record["final_group_scores"] = summary.group_scores
                handle.write(json.dumps(record) + "\n")
