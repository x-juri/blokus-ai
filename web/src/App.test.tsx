import { render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";

import App from "./App";

vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
  const url = String(input);
  if (url.includes("/api/initial-state")) {
    return new Response(
      JSON.stringify({
        variant: "standard-4",
        active_color: "blue",
        board: Array.from({ length: 20 }, () => Array.from({ length: 20 }, () => null)),
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
      })
    );
  }
  if (url.includes("/api/pieces")) {
    return new Response(JSON.stringify({ pieces: [{ piece_id: "I1", size: 1, cells: [[0, 0]], transform_count: 1 }] }));
  }
  return new Response(JSON.stringify({ suggestions: [] }));
}) as typeof fetch);

test("renders the main workbench title", async () => {
  render(<App />);
  expect(await screen.findByText("Move Suggestion Workbench")).toBeInTheDocument();
  expect(screen.getByText("Search Setup")).toBeInTheDocument();
  expect(screen.getByText("Placement tool")).toBeInTheDocument();
});
