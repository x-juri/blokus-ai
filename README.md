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
When a checkpoint is present, it still keeps a heuristic candidate floor and blends heuristic
and model leaf values so weak checkpoints do not completely override the baseline search.

### Phase 1 learning model

The RL system does not train during human games. Learning is currently offline:

1. Generate paired-2 self-play traces.
2. Train a policy/value checkpoint from those sparse visit targets.
3. Evaluate that checkpoint against heuristic MCTS on a fixed seeded opening suite.
4. Use the latest checkpoint in `Balanced` and `Strong` play presets.

Offline self-play now adds two AlphaZero-style exploration mechanisms by default:
- root Dirichlet noise in MCTS priors
- visit-distribution sampling for the early game instead of always taking the top visit count

These are enabled only for training self-play, not for live play or benchmark evaluation.

Training reports now include separate train and validation policy, value, and total losses. That
is a more reliable signal than `mean_loss` alone when deciding whether a checkpoint is worth using.

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
  --rollout-depth 1 \
  --seed 7 \
  --root-dirichlet-alpha 0.3 \
  --root-exploration-fraction 0.25 \
  --sampling-temperature 1.0 \
  --sampling-moves 16
```

This now prints an immediate start banner plus progress every 10 completed self-play games by
default. You can change that with `--progress-every`.

If you pass `--seed`, self-play derives a reproducible per-game seed and stores it in the JSONL
records. That gives you distinct but repeatable game traces instead of one deterministic trajectory
repeated across the whole export.

### 2. Train a checkpoint

```bash
uv run blokus-train \
  --records artifacts/self_play/paired2-bootstrap.jsonl \
  --checkpoint-id paired2-bootstrap-v1 \
  --epochs 3 \
  --batch-size 16 \
  --validation-split 0.1 \
  --report artifacts/reports/paired2-bootstrap-v1.json
```

The JSON report now includes:
- `train_policy_loss`, `train_value_loss`, `train_total_loss`
- `validation_policy_loss`, `validation_value_loss`, `validation_total_loss`
- `best_epoch`
- `epoch_metrics`

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
  --seed 7 \
  --opening-plies 4 \
  --json
```

The benchmark command now prints an immediate start banner plus evaluation progress for each
seat-swapped game as it starts. It also uses a fixed seeded opening suite so repeated "games"
are genuinely distinct while still remaining reproducible across runs.

### 4. Or run the whole Phase 1 bootstrap in one command

```bash
uv run blokus-phase1 \
  --checkpoint-id paired2-bootstrap-v1 \
  --games 1000 \
  --self-play-agent-id mobility-heuristic \
  --seed 7 \
  --evaluation-opening-plies 4 \
  --root-dirichlet-alpha 0.3 \
  --root-exploration-fraction 0.25 \
  --sampling-temperature 1.0 \
  --sampling-moves 16 \
  --epochs 3 \
  --evaluation-games 20
```

`blokus-phase1` streams stage progress:
- self-play start banner immediately
- self-play progress every 10 games
- training start/finish
- evaluation start banner immediately
- evaluation progress for each game

The evaluation stage now compares agents on seeded diversified openings by default instead of
replaying the exact same deterministic start position every time.

The self-play stage now also derives a reproducible per-game seed and uses that seed for root
noise and early-game visit sampling, so exported traces are both more diverse and repeatable.

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
