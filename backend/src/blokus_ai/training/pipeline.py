from __future__ import annotations

import argparse
import json
from pathlib import Path

from blokus_ai.ai.benchmark import run_paired_tournament
from blokus_ai.engine.models import AgentConfig, GameConfig, GameVariant
from blokus_ai.training.self_play import generate_self_play_records
from blokus_ai.training.train import train_policy_value_network


def run_phase_one_bootstrap(
    checkpoint_id: str,
    games: int = 512,
    simulations: int = 32,
    candidate_limit: int = 12,
    rollout_depth: int = 3,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    evaluation_games: int = 12,
) -> dict:
    artifacts_root = Path("artifacts")
    self_play_path = artifacts_root / "self_play" / f"{checkpoint_id}.jsonl"
    report_path = artifacts_root / "reports" / f"{checkpoint_id}.json"

    generate_self_play_records(
        games=games,
        output_path=self_play_path,
        config=GameConfig(variant=GameVariant.PAIRED_2),
        agent_config=AgentConfig(
            agent_id="heuristic-mcts",
            simulations=simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
        ),
        progress_every=10,
        progress_callback=lambda completed, total: print(
            f"[phase1:self-play] completed {completed}/{total} games",
            flush=True,
        ),
    )

    print("[phase1:train] starting checkpoint training", flush=True)
    training_report = train_policy_value_network(
        records_path=self_play_path,
        checkpoint_id=checkpoint_id,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        report_path=report_path,
    )
    print(
        f"[phase1:train] finished checkpoint {checkpoint_id} with mean loss "
        f"{training_report['mean_loss']:.4f}",
        flush=True,
    )

    tournament_rows = run_paired_tournament(
        AgentConfig(
            agent_id="policy-mcts",
            checkpoint_id=checkpoint_id,
            simulations=simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
        ),
        AgentConfig(
            agent_id="heuristic-mcts",
            simulations=simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
        ),
        games=evaluation_games,
        max_turns=160,
        progress_callback=lambda seat_name, game_index, total_games: print(
            f"[phase1:evaluation] {seat_name} game {game_index}/{total_games}",
            flush=True,
        ),
    )

    summary = {
        "checkpoint_id": checkpoint_id,
        "self_play_path": str(self_play_path),
        "training_report": training_report,
        "evaluation": [row.__dict__ for row in tournament_rows],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_phase_one_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 paired-2 RL bootstrap workflow.")
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--rollout-depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluation-games", type=int, default=12)
    return parser


def main() -> None:
    parser = build_phase_one_arg_parser()
    args = parser.parse_args()
    summary = run_phase_one_bootstrap(
        checkpoint_id=args.checkpoint_id,
        games=args.games,
        simulations=args.simulations,
        candidate_limit=args.candidate_limit,
        rollout_depth=args.rollout_depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_games=args.evaluation_games,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
