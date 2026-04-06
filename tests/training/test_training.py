from __future__ import annotations

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
