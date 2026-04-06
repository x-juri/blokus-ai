# Blokus AI

Blokus AI is a local-first Blokus workbench and paired-team game runner with an exact rules engine, heuristic and MCTS agents, and a Phase 1 reinforcement-learning workflow built around offline self-play and checkpointed policy/value training.

## Current scope

- Exact modeling of the official Blokus rules, scoring, and published `standard-4`, `paired-2`, and `shared-3` variants
- Workbench mode for reconstructing positions and requesting next-move suggestions
- Play mode for local `paired-2` human-vs-AI games where one player controls `blue/red` or `yellow/green`
- AI-vs-AI replay generation with playback controls
- Offline RL bootstrap workflow for paired-2 self-play, supervised checkpoint training, and checkpoint-vs-baseline evaluation

## Repository layout

```text
backend/
  pyproject.toml
  src/blokus_ai/
    api/
    ai/
    engine/
    training/
  artifacts/
web/
  package.json
  src/
tests/
docs/
.github/workflows/
```

## Stack

- Python 3.12+ target with `uv`, FastAPI, Pydantic, NumPy, PyTorch, pytest, and Ruff
- React + TypeScript + Vite for the local web app

## Quick start

### Backend

```bash
cd backend
uv sync --extra dev
uv run pytest ../tests -q
uv run uvicorn blokus_ai.api.main:app --reload
```

### Web app

```bash
cd web
npm install
npm run dev
```

The frontend expects the backend at `http://127.0.0.1:8000` by default.

## App modes

### Workbench

- edit arbitrary positions with manual piece placement
- inspect inventories per color
- request top move suggestions for the active color

### Play

- start a `paired-2` game as `player_a` (`blue/red`) or `player_b` (`yellow/green`)
- place pieces manually on human turns
- let the backend drive AI turns and forced passes
- choose a live AI preset:
  - `Fast`: low-latency heuristic MCTS for responsive play
  - `Balanced`: uses the latest policy checkpoint when available, otherwise falls back to heuristic MCTS
  - `Strong`: higher local search budget, best for slower deliberate play
- generate AI-vs-AI replays with separate replay presets
- both live play and replay default to the `Fast` preset for laptop-friendly response times

## API

- `GET /health`
- `GET /api/pieces`
- `GET /api/initial-state`
- `POST /api/new-game`
- `POST /api/legal-moves`
- `POST /api/apply-move`
- `POST /api/suggest-moves`
- `POST /api/ai-turn`
- `POST /api/replay-game`

## AI pipeline

### Search agents

- `random-legal`
- `mobility-heuristic`
- `heuristic-mcts`
- `policy-mcts`

`policy-mcts` automatically falls back to heuristic search when no checkpoint is available.

### Phase 1 learning model

The RL system does not train during human games. Learning is currently offline:

1. Generate paired-2 self-play traces.
2. Train a policy/value checkpoint from those sparse visit targets.
3. Evaluate that checkpoint against heuristic MCTS.
4. Use the latest checkpoint in `Balanced` and `Strong` play presets.

## Phase 1 training workflow

Run these commands from `backend/`:

### 1. Export self-play data

```bash
uv run blokus-self-play \
  --games 1000 \
  --output artifacts/self_play/paired2-bootstrap.jsonl \
  --agent-id heuristic-mcts \
  --simulations 8 \
  --candidate-limit 6 \
  --rollout-depth 1
```

This now prints an immediate start banner plus progress every 10 completed self-play games by
default. You can change that with `--progress-every`.

### 2. Train a checkpoint

```bash
uv run blokus-train \
  --records artifacts/self_play/paired2-bootstrap.jsonl \
  --checkpoint-id paired2-bootstrap-v1 \
  --epochs 3 \
  --batch-size 16 \
  --report artifacts/reports/paired2-bootstrap-v1.json
```

### 3. Evaluate it against the heuristic baseline

```bash
uv run blokus-benchmark \
  --agent-one policy-mcts \
  --agent-two heuristic-mcts \
  --checkpoint-id paired2-bootstrap-v1 \
  --games 20 \
  --max-turns 160 \
  --simulations 8 \
  --candidate-limit 6 \
  --rollout-depth 1 \
  --json
```

The benchmark command now prints an immediate start banner plus evaluation progress for each
seat-swapped game as it starts.

### 4. Or run the whole Phase 1 bootstrap in one command

```bash
uv run blokus-phase1 \
  --checkpoint-id paired2-bootstrap-v1 \
  --games 1000 \
  --self-play-agent-id mobility-heuristic \
  --epochs 3 \
  --evaluation-games 20
```

`blokus-phase1` streams stage progress:
- self-play start banner immediately
- self-play progress every 10 games
- training start/finish
- evaluation start banner immediately
- evaluation progress for each game

For a quick smoke run, start smaller:

```bash
uv run blokus-phase1 \
  --checkpoint-id smoke-v1 \
  --games 20 \
  --self-play-agent-id mobility-heuristic \
  --epochs 1 \
  --evaluation-games 2
```

Artifacts are written under `backend/artifacts/`:

- `self_play/`: exported JSONL traces
- `checkpoints/`: trained policy/value checkpoints
- `reports/`: training and evaluation summaries

## Notes on official variants

- Standard 4-player play uses one color per player.
- 2-player play keeps the normal color order and assigns blue/red against yellow/green.
- 3-player play keeps the normal color order and uses one shared color whose final score is ignored.

## Documentation

- [Architecture](docs/architecture.md)
- [Training roadmap](docs/training-roadmap.md)
