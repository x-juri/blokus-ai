import { startTransition, useEffect, useState, useDeferredValue } from "react";

import { applyMove, fetchInitialState, fetchPieces, suggestMoves } from "./api/client";
import { BoardGrid } from "./components/BoardGrid";
import { ControlsPanel } from "./components/ControlsPanel";
import { PieceInventoryPanel } from "./components/PieceInventoryPanel";
import { SuggestionsPanel } from "./components/SuggestionsPanel";
import type {
  BoardState,
  Coordinate,
  Move,
  MoveSuggestion,
  PieceDescriptor,
  PlacementSelection,
  PlayerColor
} from "./types/blokus";
import {
  coordinatesToSet,
  isPlacementValid,
  pieceCellsForId,
  placePieceAtAnchor
} from "./utils/pieces";
import { createInitialState, syncOpenedColors } from "./utils/state";

function App() {
  const [boardState, setBoardState] = useState<BoardState>(createInitialState);
  const [pieces, setPieces] = useState<PieceDescriptor[]>([]);
  const [suggestions, setSuggestions] = useState<MoveSuggestion[]>([]);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const [selectedPlacement, setSelectedPlacement] = useState<PlacementSelection | null>(null);
  const [hoveredCell, setHoveredCell] = useState<Coordinate | null>(null);
  const [topK, setTopK] = useState(5);
  const [simulations, setSimulations] = useState(96);
  const [candidateLimit, setCandidateLimit] = useState(24);
  const [statusMessage, setStatusMessage] = useState("Load a board state or start from the empty official setup.");
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const deferredSuggestions = useDeferredValue(suggestions);
  const selectedSuggestion = deferredSuggestions[selectedSuggestionIndex];
  const suggestionCells =
    selectedSuggestion && !selectedSuggestion.move.is_pass
      ? coordinatesToSet(
          placePieceAtAnchor(
            selectedSuggestion.move.anchor_cell,
            pieceCellsForId(
              pieces,
              selectedSuggestion.move.piece_id,
              selectedSuggestion.move.rotation,
              selectedSuggestion.move.reflection
            )
          )
        )
      : new Set<string>();

  const selectedPieceCells = selectedPlacement
    ? pieceCellsForId(
        pieces,
        selectedPlacement.pieceId,
        selectedPlacement.rotation,
        selectedPlacement.reflection
      )
    : [];

  const previewPlacement =
    selectedPlacement && hoveredCell
      ? placePieceAtAnchor(hoveredCell, selectedPieceCells)
      : [];
  const previewCells = coordinatesToSet(previewPlacement);
  const previewInvalid =
    previewPlacement.length > 0 && !isPlacementValid(boardState.board, previewPlacement);

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
    if (!selectedPlacement) {
      if (!boardState.board[row][col]) {
        setStatusMessage("Select a remaining piece to place it, or click an occupied cell to erase it.");
        return;
      }
      const nextBoard = boardState.board.map((boardRow) => [...boardRow]);
      nextBoard[row][col] = null;
      setErrorMessage("");
      setBoardState({
        ...boardState,
        board: nextBoard,
        opened_colors: syncOpenedColors(nextBoard)
      });
      setSuggestions([]);
      setStatusMessage("Erased one cell. If needed, restore the corresponding piece in the inventory.");
      return;
    }

    const placement = placePieceAtAnchor({ row, col }, selectedPieceCells);
    if (!isPlacementValid(boardState.board, placement)) {
      setErrorMessage("That piece does not fit there. Keep it in bounds and off occupied cells.");
      return;
    }

    const nextBoard = boardState.board.map((boardRow) => [...boardRow]);
    for (const cell of placement) {
      nextBoard[cell.row][cell.col] = selectedPlacement.color;
    }

    const remainingForColor = boardState.remaining_pieces_by_color[selectedPlacement.color].filter(
      (pieceId) => pieceId !== selectedPlacement.pieceId
    );

    setErrorMessage("");
    setSuggestions([]);
    setBoardState({
      ...boardState,
      board: nextBoard,
      remaining_pieces_by_color: {
        ...boardState.remaining_pieces_by_color,
        [selectedPlacement.color]: remainingForColor
      },
      opened_colors: syncOpenedColors(nextBoard),
      last_piece_placed_by_color: {
        ...boardState.last_piece_placed_by_color,
        [selectedPlacement.color]: selectedPlacement.pieceId
      }
    });
    setSelectedPlacement(null);
    setStatusMessage(
      `Placed ${selectedPlacement.pieceId} for ${selectedPlacement.color} at ${row}, ${col}.`
    );
  }

  function selectPiece(color: PlayerColor, pieceId: string) {
    const shouldClear =
      selectedPlacement?.color === color && selectedPlacement.pieceId === pieceId;
    setSelectedPlacement(
      shouldClear
        ? null
        : {
            color,
            pieceId,
            rotation: 0,
            reflection: false
          }
    );
    setErrorMessage("");
    setStatusMessage(
      shouldClear
        ? "Selection cleared. Click an occupied cell to erase it or choose another piece."
        : `Selected ${pieceId} for ${color}. Click the board to place it.`
    );
  }

  function restorePiece(color: PlayerColor, pieceId: string) {
    const existing = boardState.remaining_pieces_by_color[color];
    if (existing.includes(pieceId)) {
      return;
    }
    const nextPieces = [...existing, pieceId].sort();
    setBoardState({
      ...boardState,
      remaining_pieces_by_color: {
        ...boardState.remaining_pieces_by_color,
        [color]: nextPieces
      }
    });
    setStatusMessage(`Returned ${pieceId} to ${color}'s remaining inventory.`);
  }

  function rotateSelection() {
    setSelectedPlacement((currentSelection) =>
      currentSelection
        ? {
            ...currentSelection,
            rotation: (currentSelection.rotation + 90) % 360
          }
        : null
    );
  }

  function flipSelection() {
    setSelectedPlacement((currentSelection) =>
      currentSelection
        ? {
            ...currentSelection,
            reflection: !currentSelection.reflection
          }
        : null
    );
  }

  function clearSelection() {
    setSelectedPlacement(null);
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
      setSelectedPlacement(null);
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
      setSelectedPlacement(null);
      setStatusMessage(`Passed turn for ${passMove.color}.`);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Pass failed.");
    }
  }

  function handleReset() {
    startTransition(() => {
      setBoardState(createInitialState());
      setSuggestions([]);
      setSelectedPlacement(null);
      setHoveredCell(null);
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
          selectedPlacement={selectedPlacement}
          onCellClick={handleCellClick}
          onCellHover={(row, col) => setHoveredCell({ row, col })}
          onBoardLeave={() => setHoveredCell(null)}
          suggestionCells={suggestionCells}
          previewCells={previewCells}
          previewInvalid={previewInvalid}
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
            selectedPlacement={selectedPlacement}
            selectedPieceCells={selectedPieceCells}
            onStateChange={updateBoardState}
            onTopKChange={setTopK}
            onSimulationsChange={setSimulations}
            onCandidateLimitChange={setCandidateLimit}
            onRotateSelection={rotateSelection}
            onFlipSelection={flipSelection}
            onClearSelection={clearSelection}
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

      <PieceInventoryPanel
        boardState={boardState}
        pieces={pieces}
        selectedPlacement={
          selectedPlacement
            ? { color: selectedPlacement.color, pieceId: selectedPlacement.pieceId }
            : null
        }
        onSelectPiece={selectPiece}
        onRestorePiece={restorePiece}
      />
    </div>
  );
}

export default App;
