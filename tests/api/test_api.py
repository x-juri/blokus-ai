from __future__ import annotations

from fastapi.testclient import TestClient

from blokus_ai.api.main import app
from blokus_ai.engine.game import create_initial_state
from blokus_ai.engine.models import GameConfig, GameVariant


client = TestClient(app)


def test_pieces_endpoint_returns_catalog() -> None:
    response = client.get("/api/pieces")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["pieces"]) == 21


def test_legal_moves_endpoint_returns_opening_moves() -> None:
    state = create_initial_state(GameConfig())
    response = client.post("/api/legal-moves", json={"state": state.model_dump(mode="json")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["moves"]


def test_apply_move_endpoint_rejects_illegal_moves() -> None:
    state = create_initial_state(GameConfig())
    move = {
        "color": "blue",
        "piece_id": "I1",
        "anchor_cell": {"row": 5, "col": 5},
        "rotation": 0,
        "reflection": False,
        "is_pass": False,
    }
    response = client.post(
        "/api/apply-move",
        json={"state": state.model_dump(mode="json"), "move": move},
    )
    assert response.status_code == 400


def test_suggest_moves_endpoint_returns_ranked_suggestions() -> None:
    state = create_initial_state(GameConfig())
    response = client.post(
        "/api/suggest-moves",
        json={
            "state": state.model_dump(mode="json"),
            "top_k": 3,
            "simulations": 12,
            "candidate_limit": 8,
            "rollout_depth": 3,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["suggestions"]) == 3


def test_new_game_endpoint_returns_paired_two_state() -> None:
    response = client.post(
        "/api/new-game",
        json={"config": {"variant": "paired-2"}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["variant"] == "paired-2"


def test_ai_turn_endpoint_advances_paired_two_game() -> None:
    state = create_initial_state(GameConfig(variant=GameVariant.PAIRED_2))
    response = client.post(
        "/api/ai-turn",
        json={
            "state": state.model_dump(mode="json"),
            "agent_id": "heuristic-mcts",
            "simulations": 8,
            "candidate_limit": 6,
            "rollout_depth": 3,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["move_history"]
    assert payload["move"] is not None


def test_replay_game_endpoint_returns_state_history() -> None:
    response = client.post(
        "/api/replay-game",
        json={
            "config": {"variant": "paired-2"},
            "player_a_agent": {"agent_id": "random-legal", "seed": 3},
            "player_b_agent": {"agent_id": "random-legal", "seed": 7},
            "max_turns": 8,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["state_history"]
    assert len(payload["state_history"]) >= 1


def test_suggest_moves_accepts_agent_overrides() -> None:
    state = create_initial_state(GameConfig())
    response = client.post(
        "/api/suggest-moves",
        json={
            "state": state.model_dump(mode="json"),
            "top_k": 1,
            "agent_id": "random-legal",
            "seed": 13,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["suggestions"][0]["rationale"] == "Random legal baseline."
