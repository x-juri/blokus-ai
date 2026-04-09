from __future__ import annotations

import argparse
import json
from pathlib import Path

from blokus_ai.ai.benchmark import run_paired_tournament
from blokus_ai.engine.models import AgentConfig, AgentId, GameConfig, GameVariant
from blokus_ai.training.self_play import generate_self_play_records
from blokus_ai.training.train import train_policy_value_network


def run_phase_one_bootstrap(
    checkpoint_id: str,
    games: int = 512,
    self_play_agent_id: AgentId = "mobility-heuristic",
    simulations: int = 8,
    candidate_limit: int = 6,
    rollout_depth: int = 1,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    evaluation_games: int = 12,
    seed: int = 7,
    evaluation_opening_plies: int = 4,
    root_dirichlet_alpha: float = 0.3,
    root_exploration_fraction: float = 0.25,
    sampling_temperature: float = 1.0,
    sampling_moves: int = 16,
    progress_every: int = 10,
) -> dict:
    artifacts_root = Path("artifacts")
    self_play_path = artifacts_root / "self_play" / f"{checkpoint_id}.jsonl"
    report_path = artifacts_root / "reports" / f"{checkpoint_id}.json"

    print(
        "[phase1:self-play] starting "
        f"{games} games with {self_play_agent_id} "
        f"(simulations={simulations}, candidate_limit={candidate_limit}, rollout_depth={rollout_depth}, "
        f"root_dirichlet_alpha={root_dirichlet_alpha}, "
        f"root_exploration_fraction={root_exploration_fraction}, "
        f"sampling_temperature={sampling_temperature}, sampling_moves={sampling_moves})",
        flush=True,
    )
    generate_self_play_records(
        games=games,
        output_path=self_play_path,
        config=GameConfig(variant=GameVariant.PAIRED_2),
        agent_config=AgentConfig(
            agent_id=self_play_agent_id,
            simulations=simulations,
            candidate_limit=candidate_limit,
            rollout_depth=rollout_depth,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
            sampling_temperature=sampling_temperature,
            sampling_moves=sampling_moves,
            seed=seed,
        ),
        sampling_moves=sampling_moves,
        sampling_temperature=sampling_temperature,
        progress_every=progress_every,
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
        seed=seed,
        report_path=report_path,
    )
    print(
        f"[phase1:train] finished checkpoint {checkpoint_id} with "
        f"train total {training_report['train_total_loss']:.4f}"
        + (
            ""
            if training_report["validation_total_loss"] is None
            else f", validation total {training_report['validation_total_loss']:.4f}"
        ),
        flush=True,
    )

    print(
        "[phase1:evaluation] starting "
        f"{evaluation_games} games per seat with policy-mcts vs heuristic-mcts "
        f"(simulations={simulations}, candidate_limit={candidate_limit}, rollout_depth={rollout_depth}, "
        f"seed={seed}, opening_plies={evaluation_opening_plies})",
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
        base_seed=seed,
        opening_plies=evaluation_opening_plies,
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
    parser.add_argument(
        "--self-play-agent-id",
        choices=["heuristic-mcts", "mobility-heuristic", "random-legal"],
        default="mobility-heuristic",
    )
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=6)
    parser.add_argument("--rollout-depth", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluation-games", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluation-opening-plies", type=int, default=4)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-moves", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main() -> None:
    parser = build_phase_one_arg_parser()
    args = parser.parse_args()
    summary = run_phase_one_bootstrap(
        checkpoint_id=args.checkpoint_id,
        games=args.games,
        self_play_agent_id=args.self_play_agent_id,
        simulations=args.simulations,
        candidate_limit=args.candidate_limit,
        rollout_depth=args.rollout_depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_games=args.evaluation_games,
        seed=args.seed,
        evaluation_opening_plies=args.evaluation_opening_plies,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        sampling_temperature=args.sampling_temperature,
        sampling_moves=args.sampling_moves,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
