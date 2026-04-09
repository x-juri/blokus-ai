from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from blokus_ai.engine.game import create_initial_state, generate_legal_moves
from blokus_ai.engine.models import AgentConfig, GameConfig, GameVariant
from blokus_ai.training.encoding import (
    ACTION_SPACE_SIZE,
    PASS_ACTION_INDEX,
    decode_action,
    encode_action,
    encode_state_tensors,
)
from blokus_ai.training.model import (
    build_policy_value_network,
    load_policy_value_checkpoint,
    save_policy_value_checkpoint,
)
from blokus_ai.training.self_play import generate_self_play_records
from blokus_ai.training.train import train_policy_value_network


def test_action_encoding_round_trip_for_opening_move() -> None:
    state = create_initial_state(GameConfig(variant=GameVariant.PAIRED_2))
    move = generate_legal_moves(state)[0]
    action_index = encode_action(move)
    assert ACTION_SPACE_SIZE == 36401
    assert action_index != PASS_ACTION_INDEX
    assert decode_action(action_index, move.color) == move


def test_policy_value_network_outputs_expected_shapes() -> None:
    if sys.version_info < (3, 12):
        pytest.skip("Torch-backed training tests require the project Python 3.12+ environment.")
    state = create_initial_state(GameConfig(variant=GameVariant.PAIRED_2))
    model = build_policy_value_network()
    spatial_inputs, metadata_inputs = encode_state_tensors(state)
    policy_logits, values = model(spatial_inputs, metadata_inputs)
    assert policy_logits.shape == (1, ACTION_SPACE_SIZE)
    assert values.shape == (1,)


def test_checkpoint_save_and_load_round_trip() -> None:
    if sys.version_info < (3, 12):
        pytest.skip("Torch-backed checkpoint tests require the project Python 3.12+ environment.")
    checkpoint_id = "unit-test-policy"
    model = build_policy_value_network()
    save_policy_value_checkpoint(model, checkpoint_id, metadata={"source": "test"})
    loaded = load_policy_value_checkpoint(checkpoint_id)
    assert loaded is not None
    assert loaded.checkpoint_id == checkpoint_id
    assert loaded.metadata["source"] == "test"


def test_self_play_progress_callback_reports_final_batch(tmp_path: Path) -> None:
    progress_calls: list[tuple[int, int]] = []
    output_path = tmp_path / "smoke.jsonl"
    generate_self_play_records(
        games=3,
        output_path=output_path,
        config=GameConfig(variant=GameVariant.PAIRED_2),
        agent_config=AgentConfig(agent_id="random-legal", seed=1),
        progress_every=2,
        progress_callback=lambda completed, total: progress_calls.append((completed, total)),
    )
    assert progress_calls == [(2, 3), (3, 3)]


def test_self_play_records_include_per_game_seed(tmp_path: Path) -> None:
    output_path = tmp_path / "seeded-self-play.jsonl"
    generate_self_play_records(
        games=2,
        output_path=output_path,
        config=GameConfig(variant=GameVariant.PAIRED_2),
        agent_config=AgentConfig(agent_id="random-legal", seed=41),
    )

    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    game_seeds = {
        (record["game_index"], record["game_seed"])
        for record in (json.loads(line) for line in lines)
    }
    assert game_seeds == {(0, 41), (1, 42)}


def test_self_play_records_capture_sampling_diagnostics(tmp_path: Path) -> None:
    output_path = tmp_path / "sampled-self-play.jsonl"
    generate_self_play_records(
        games=1,
        output_path=output_path,
        config=GameConfig(variant=GameVariant.PAIRED_2),
        agent_config=AgentConfig(
            agent_id="heuristic-mcts",
            simulations=4,
            candidate_limit=4,
            rollout_depth=1,
            seed=5,
            root_dirichlet_alpha=0.3,
            root_exploration_fraction=0.25,
        ),
        sampling_moves=2,
        sampling_temperature=1.0,
    )

    first_record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_record["game_seed"] == 5
    assert first_record["diagnostics"]["sampled_from_visit_distribution"] is True
    assert first_record["diagnostics"]["root_dirichlet_noise_applied"] is True


def test_training_report_includes_validation_diagnostics(tmp_path: Path) -> None:
    if sys.version_info < (3, 12):
        pytest.skip("Torch-backed training tests require the project Python 3.12+ environment.")

    output_path = tmp_path / "training.jsonl"
    report_path = tmp_path / "training-report.json"
    generate_self_play_records(
        games=2,
        output_path=output_path,
        config=GameConfig(variant=GameVariant.PAIRED_2),
        agent_config=AgentConfig(agent_id="random-legal", seed=5),
    )

    report = train_policy_value_network(
        records_path=output_path,
        checkpoint_id="unit-test-diagnostics",
        epochs=1,
        batch_size=8,
        validation_split=0.2,
        seed=11,
        report_path=report_path,
    )

    assert report["records"] > 0
    assert report["train_records"] + report["validation_records"] == report["records"]
    assert report["train_policy_loss"] >= 0.0
    assert report["train_value_loss"] >= 0.0
    assert report["train_total_loss"] >= 0.0
    assert report["validation_total_loss"] is not None
    assert report["best_epoch"] == 1
    assert len(report["epoch_metrics"]) == 1
    assert report_path.exists()
