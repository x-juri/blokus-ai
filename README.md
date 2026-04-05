# Blokus AI

Search-first Blokus AI with an exact rules engine, baseline agents, a CPU-friendly MCTS suggester, and a local web app for editing positions and reviewing recommended moves.

## Project goals

- Model the official Mattel Blokus rules exactly, including scoring and the published 2-, 3-, and 4-player variants.
- Provide a practical "what should I play next?" assistant for standard four-color Blokus.
- Build a clean foundation for later policy/value learning from self-play and search traces.

## Repository layout

```text
backend/
  pyproject.toml
  src/blokus_ai/
    api/
    ai/
    engine/
    training/
web/
  package.json
  src/
tests/
docs/
.github/workflows/
```

## Stack

- Python 3.12 target with `uv`, FastAPI, Pydantic, PyTorch, pytest, and Ruff
- React + TypeScript + Vite for the local web app

## Quick start

### Backend

```bash
cd backend
uv sync
uv run pytest ../tests
uv run uvicorn blokus_ai.api.main:app --reload
```

If `uv` is not installed, a standard virtual environment also works:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest ../tests
uvicorn blokus_ai.api.main:app --reload
```

### Web app

```bash
cd web
npm install
npm run dev
```

The frontend expects the backend at `http://127.0.0.1:8000` by default.

## API

- `GET /api/pieces`
- `POST /api/legal-moves`
- `POST /api/apply-move`
- `POST /api/suggest-moves`

## Current AI pipeline

1. Exact legal-move generation with precomputed polyomino transforms.
2. Baseline agents:
   - random legal
   - largest-piece greedy
   - mobility/blocking heuristic
3. Progressive-widening MCTS with heuristic candidate pruning.
4. Training scaffolds for self-play trace export and policy/value learning.

## Notes on official variants

- Standard 4-player play uses one color per player.
- 2-player play keeps the normal color order and assigns blue/red against yellow/green.
- 3-player play keeps the normal color order and uses one shared color whose final score is ignored.

## Documentation

- [Architecture](docs/architecture.md)
- [Training roadmap](docs/training-roadmap.md)

