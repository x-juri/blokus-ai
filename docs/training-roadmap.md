# Training Roadmap

## Phase 1

- Generate self-play traces from the heuristic and MCTS agents.
- Persist game states, move priors, chosen actions, and final score margins.

## Phase 2

- Train a lightweight policy/value network on exported traces.
- Use the policy head to improve root priors and progressive widening order.
- Use the value head to replace or blend with the rollout evaluator.

## Phase 3

- Run guided self-play with the learned model inside MCTS.
- Compare model-guided search against the heuristic-only baseline in tournament scripts.

