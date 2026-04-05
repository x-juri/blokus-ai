import { startTransition, useEffect, useState, useDeferredValue } from "react";

import { applyMove, fetchInitialState, fetchPieces, suggestMoves } from "./api/client";
import { BoardGrid } from "./components/BoardGrid";
import { ControlsPanel } from "./components/ControlsPanel";
import { PieceInventoryPanel } from "./components/PieceInventoryPanel";
import { SuggestionsPanel } from "./components/SuggestionsPanel";
import type { BoardState, Move, MoveSuggestion, PieceDescriptor, PlayerColor } from "./types/blokus";
import { createInitialState, syncOpenedColors } from "./utils/state";

function App() {
  const [boardState, setBoardState] = useState<BoardState>(createInitialState);
  const [pieces, setPieces] = useState<PieceDescriptor[]>([]);
  const [suggestions, setSuggestions] = useState<MoveSuggestion[]>([]);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const [selectedPaintColor, setSelectedPaintColor] = useState<PlayerColor | null>("blue");
  const [topK, setTopK] = useState(5);
  const [simulations, setSimulations] = useState(96);
  const [candidateLimit, setCandidateLimit] = useState(24);
  const [statusMessage, setStatusMessage] = useState("Load a board state or start from the empty official setup.");
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const deferredSuggestions = useDeferredValue(suggestions);
  const highlightedCells = new Set<string>();
  const selectedSuggestion = deferredSuggestions[selectedSuggestionIndex];
  if (selectedSuggestion && !selectedSuggestion.move.is_pass) {
    highlightedCells.add(
      `${selectedSuggestion.move.anchor_cell.row}:${selectedSuggestion.move.anchor_cell.col}`
    );
  }

  useEffect(() => {
    void (async () => {
      const [initialState, pieceCatalog] = await Promise.all([fetchInitialState(), fetchPieces()]);
      setBoardState(initialState);
      setPieces(pieceCatalog);
    })();
  }, []);

  function updateBoardState(nextState: BoardState) {
    setBoardState(nextState);
  }

  function handleCellClick(row: number, col: number) {
    const nextBoard = boardState.board.map((boardRow) => [...boardRow]);
    const current = nextBoard[row][col];
    nextBoard[row][col] = current === selectedPaintColor ? null : selectedPaintColor;
    const openedColors = syncOpenedColors(nextBoard);
    startTransition(() => {
      setBoardState({
        ...boardState,
        board: nextBoard,
        opened_colors: openedColors
      });
    });
  }

  function togglePiece(color: PlayerColor, pieceId: string) {
    const existing = boardState.remaining_pieces_by_color[color];
    const nextPieces = existing.includes(pieceId)
      ? existing.filter((item) => item !== pieceId)
      : [...existing, pieceId].sort();

    setBoardState({
      ...boardState,
      remaining_pieces_by_color: {
        ...boardState.remaining_pieces_by_color,
        [color]: nextPieces
      }
    });
  }

  async function handleSuggest() {
    setLoading(true);
    setErrorMessage("");
    setStatusMessage(`Searching ${topK} moves for ${boardState.active_color}...`);
    try {
      const ranked = await suggestMoves(boardState, topK, simulations, candidateLimit);
      setSuggestions(ranked);
      setSelectedSuggestionIndex(0);
      setStatusMessage(
        ranked.length
          ? `Found ${ranked.length} ranked moves for ${boardState.active_color}.`
          : `${boardState.active_color} has no legal moves in this position.`
      );
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Suggestion request failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleApplySuggestion(index: number) {
    const suggestion = suggestions[index];
    if (!suggestion) {
      return;
    }
    try {
      const nextState = await applyMove(boardState, suggestion.move);
      setBoardState(nextState);
      setSuggestions([]);
      setStatusMessage(`Applied ${suggestion.move.piece_id} for ${suggestion.move.color}.`);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not apply move.");
    }
  }

  async function handlePass() {
    const passMove: Move = {
      color: boardState.active_color,
      piece_id: "PASS",
      anchor_cell: { row: 0, col: 0 },
      rotation: 0,
      reflection: false,
      is_pass: true
    };
    try {
      const nextState = await applyMove(boardState, passMove);
      setBoardState(nextState);
      setSuggestions([]);
      setStatusMessage(`Passed turn for ${passMove.color}.`);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Pass failed.");
    }
  }

  function handleReset() {
    startTransition(() => {
      setBoardState(createInitialState());
      setSuggestions([]);
      setErrorMessage("");
      setStatusMessage("Reset to the official empty board.");
    });
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Blokus AI</p>
          <h1>Move Suggestion Workbench</h1>
        </div>
        <p className="hero-copy">
          Exact rules, search-first move ranking, and a board editor for reconstructing real games.
        </p>
      </header>

      <main className="layout-grid">
        <BoardGrid
          board={boardState.board}
          selectedPaintColor={selectedPaintColor}
          onCellClick={handleCellClick}
          highlightedCells={highlightedCells}
        />

        <div className="sidebar-stack">
          <ControlsPanel
            boardState={boardState}
            topK={topK}
            simulations={simulations}
            candidateLimit={candidateLimit}
            statusMessage={statusMessage}
            errorMessage={errorMessage}
            loading={loading}
            selectedPaintColor={selectedPaintColor}
            onStateChange={updateBoardState}
            onTopKChange={setTopK}
            onSimulationsChange={setSimulations}
            onCandidateLimitChange={setCandidateLimit}
            onPaintColorChange={setSelectedPaintColor}
            onSuggest={handleSuggest}
            onReset={handleReset}
            onPass={handlePass}
          />

          <SuggestionsPanel
            suggestions={deferredSuggestions}
            selectedIndex={selectedSuggestionIndex}
            onSelect={setSelectedSuggestionIndex}
            onApply={handleApplySuggestion}
          />
        </div>
      </main>

      <PieceInventoryPanel boardState={boardState} pieces={pieces} onTogglePiece={togglePiece} />
    </div>
  );
}

export default App;

