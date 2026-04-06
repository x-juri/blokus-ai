import type { AgentConfig, AiPresetId, TeamId } from "../types/blokus";

export const aiPresetLabels: Record<AiPresetId, string> = {
  fast: "Fast",
  balanced: "Balanced",
  strong: "Strong"
};

export const aiPresetDescriptions: Record<AiPresetId, string> = {
  fast: "Low-latency heuristic search for responsive local play.",
  balanced: "Uses the latest policy checkpoint when available with a moderate search budget.",
  strong: "Highest local search budget; best for deliberate turns and replays."
};

export function buildLiveAgentConfig(preset: AiPresetId): AgentConfig {
  if (preset === "fast") {
    return {
      agent_id: "heuristic-mcts",
      simulations: 8,
      candidate_limit: 6,
      rollout_depth: 1
    };
  }

  if (preset === "balanced") {
    return {
      agent_id: "policy-mcts",
      simulations: 12,
      candidate_limit: 8,
      rollout_depth: 1
    };
  }

  return {
    agent_id: "policy-mcts",
    simulations: 24,
    candidate_limit: 10,
    rollout_depth: 2
  };
}

export function buildReplayAgents(preset: AiPresetId): Record<TeamId, AgentConfig> {
  if (preset === "fast") {
    return {
      player_a: buildLiveAgentConfig("fast"),
      player_b: {
        agent_id: "heuristic-mcts",
        simulations: 8,
        candidate_limit: 6,
        rollout_depth: 1
      }
    };
  }

  if (preset === "balanced") {
    return {
      player_a: buildLiveAgentConfig("balanced"),
      player_b: {
        agent_id: "heuristic-mcts",
        simulations: 12,
        candidate_limit: 8,
        rollout_depth: 1
      }
    };
  }

  return {
    player_a: buildLiveAgentConfig("strong"),
    player_b: {
      agent_id: "heuristic-mcts",
      simulations: 20,
      candidate_limit: 10,
      rollout_depth: 2
    }
  };
}
