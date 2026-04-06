import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";

import App from "./App";

const emptyBoard = Array.from({ length: 20 }, () => Array.from({ length: 20 }, () => null));

const baseState = {
  variant: "standard-4",
  active_color: "blue",
  board: emptyBoard,
  remaining_pieces_by_color: {
    blue: ["I1"],
    yellow: ["I1"],
    red: ["I1"],
    green: ["I1"]
  },
  opened_colors: {
    blue: false,
    yellow: false,
    red: false,
    green: false
  },
  passes_in_row: 0,
  shared_color: null,
  move_history: [],
  last_piece_placed_by_color: {
    blue: null,
    yellow: null,
    red: null,
    green: null
  }
};

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/initial-state")) {
        return new Response(JSON.stringify(baseState));
      }
      if (url.includes("/api/new-game")) {
        return new Response(
          JSON.stringify({
            state: {
              ...baseState,
              variant: "paired-2"
            }
          })
        );
      }
      if (url.includes("/api/pieces")) {
        return new Response(
          JSON.stringify({
            pieces: [{ piece_id: "I1", size: 1, cells: [[0, 0]], transform_count: 1 }]
          })
        );
      }
      if (url.includes("/api/replay-game")) {
        return new Response(
          JSON.stringify({
            initial_state: {
              ...baseState,
              variant: "paired-2"
            },
            moves: [],
            state_history: [
              {
                ...baseState,
                variant: "paired-2"
              }
            ],
            result: {
              scores_by_color: {
                blue: 0,
                yellow: 0,
                red: 0,
                green: 0
              },
              group_scores: {
                player_a: 0,
                player_b: 0
              },
              winner_group: null
            },
            seed: 7,
            agent_matchup: {
              player_a: { agent_id: "policy-mcts" },
              player_b: { agent_id: "heuristic-mcts" }
            }
          })
        );
      }
      return new Response(JSON.stringify({ suggestions: [] }));
    }) as typeof fetch
  );
});

test("renders the main workbench title", async () => {
  render(<App />);
  expect(await screen.findByText("Move Suggestion Workbench")).toBeInTheDocument();
  expect(screen.getByText("Search Setup")).toBeInTheDocument();
  expect(screen.getByText("Placement tool")).toBeInTheDocument();
});

test("switches to play mode and starts a paired-two game", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "Play" }));
  expect(await screen.findByText("Human Vs AI")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Play blue/red" }));
  expect(await screen.findByText("Active Pieces")).toBeInTheDocument();
  expect(screen.getByText("AI Vs AI")).toBeInTheDocument();
});
