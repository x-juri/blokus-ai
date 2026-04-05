import type { PlacementSelection, PlayerColor } from "../types/blokus";

interface BoardGridProps {
  board: (PlayerColor | null)[][];
  selectedPlacement: PlacementSelection | null;
  onCellClick: (row: number, col: number) => void;
  onCellHover: (row: number, col: number) => void;
  onBoardLeave: () => void;
  suggestionCells: Set<string>;
  previewCells: Set<string>;
  previewInvalid: boolean;
}

export function BoardGrid({
  board,
  selectedPlacement,
  onCellClick,
  onCellHover,
  onBoardLeave,
  suggestionCells,
  previewCells,
  previewInvalid
}: BoardGridProps) {
  const previewLabel = selectedPlacement
    ? `${selectedPlacement.color} ${selectedPlacement.pieceId}, ${selectedPlacement.rotation}°, ${
        selectedPlacement.reflection ? "flipped" : "normal"
      }`
    : "erase mode";

  return (
    <div className="board-shell">
      <div className="board-header">
        <div>
          <p className="eyebrow">Workbench</p>
          <h2>Position Editor</h2>
        </div>
        <p className="caption">
          Select a remaining piece, rotate or flip it, then click a square to place it. With no piece
          selected, clicking an occupied square erases one cell.
        </p>
      </div>
      <div className="paint-chip-row">
        <span className="paint-label">Editor</span>
        <span className={`paint-chip ${selectedPlacement?.color ?? "empty"}`}>
          {previewLabel}
        </span>
      </div>
      <div className="board-grid" role="grid" aria-label="Blokus board" onMouseLeave={onBoardLeave}>
        {board.map((row, rowIndex) =>
          row.map((cell, colIndex) => {
            const key = `${rowIndex}:${colIndex}`;
            const className = [
              "board-cell",
              cell ? `cell-${cell}` : "cell-empty",
              suggestionCells.has(key) ? "cell-highlighted" : "",
              previewCells.has(key) ? (previewInvalid ? "cell-preview-invalid" : "cell-preview") : ""
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <button
                key={key}
                type="button"
                role="gridcell"
                className={className}
                aria-label={`cell ${rowIndex + 1}-${colIndex + 1}`}
                onMouseEnter={() => onCellHover(rowIndex, colIndex)}
                onClick={() => onCellClick(rowIndex, colIndex)}
              />
            );
          })
        )}
      </div>
    </div>
  );
}
