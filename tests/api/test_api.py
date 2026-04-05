from __future__ import annotations

from fastapi.testclient import TestClient

from blokus_ai.api.main import app
from blokus_ai.engine.game import create_initial_state
from blokus_ai.engine.models import GameConfig


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

