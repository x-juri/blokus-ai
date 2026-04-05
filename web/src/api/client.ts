import { fallbackPieces } from "../constants/pieces";
import type { BoardState, Move, MoveSuggestion, PieceDescriptor } from "../types/blokus";
import { createInitialState } from "../utils/state";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json"
    },
    ...options
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchInitialState(): Promise<BoardState> {
  try {
    return await requestJson<BoardState>("/api/initial-state");
  } catch {
    return createInitialState();
  }
}

export async function fetchPieces(): Promise<PieceDescriptor[]> {
  try {
    const payload = await requestJson<{ pieces: PieceDescriptor[] }>("/api/pieces");
    return payload.pieces;
  } catch {
    return fallbackPieces;
  }
}

export async function applyMove(state: BoardState, move: Move): Promise<BoardState> {
  return requestJson<BoardState>("/api/apply-move", {
    method: "POST",
    body: JSON.stringify({ state, move })
  });
}

export async function suggestMoves(
  state: BoardState,
  topK: number,
  simulations: number,
  candidateLimit: number
): Promise<MoveSuggestion[]> {
  const payload = await requestJson<{ suggestions: MoveSuggestion[] }>("/api/suggest-moves", {
    method: "POST",
    body: JSON.stringify({
      state,
      top_k: topK,
      simulations,
      candidate_limit: candidateLimit
    })
  });
  return payload.suggestions;
}

