# Training Roadmap

## Phase 1

- Status: implemented as an offline paired-2 bootstrap workflow.
- Generate self-play traces from the heuristic MCTS agent.
- Persist game states, sparse root-visit targets, chosen actions, and final team-margin targets.
- Train a lightweight policy/value checkpoint from those traces.
- Evaluate the checkpoint against heuristic MCTS before promoting it into live play presets.
- Use seeded per-game self-play, root Dirichlet noise, and early-game visit sampling to diversify traces.
- Use seeded diversified opening suites for benchmark evaluation.

### Commands

```bash
cd backend
uv run blokus-self-play \
  --games 1000 \
  --output artifacts/self_play/paired2-bootstrap.jsonl \
  --agent-id heuristic-mcts \
  --seed 7 \
  --root-dirichlet-alpha 0.3 \
  --root-exploration-fraction 0.25 \
  --sampling-temperature 1.0 \
  --sampling-moves 16
uv run blokus-train --records artifacts/self_play/paired2-bootstrap.jsonl --checkpoint-id paired2-bootstrap-v1
uv run blokus-benchmark \
  --agent-one policy-mcts \
  --agent-two heuristic-mcts \
  --checkpoint-id paired2-bootstrap-v1 \
  --seed 7 \
  --opening-plies 4 \
  --json
```

## Phase 2

- Train a lightweight policy/value network on exported traces.
- Use the policy head to improve root priors and progressive widening order.
- Use the value head to replace or blend with the rollout evaluator.
- Add stronger checkpoint management and promote only checkpoints that beat the baseline at equal budget.

## Phase 3

- Run guided self-play with the learned model inside MCTS.
- Compare model-guided search against the heuristic-only baseline in tournament scripts.
- Consider batch inference, transposition caching, and lighter internal search-state representations for speed.
