import type { PlayerColor } from "../types/blokus";

interface BoardGridProps {
  board: (PlayerColor | null)[][];
  selectedPaintColor: PlayerColor | null;
  onCellClick: (row: number, col: number) => void;
  highlightedCells: Set<string>;
}

export function BoardGrid({
  board,
  selectedPaintColor,
  onCellClick,
  highlightedCells
}: BoardGridProps) {
  return (
    <div className="board-shell">
      <div className="board-header">
        <div>
          <p className="eyebrow">Workbench</p>
          <h2>Position Editor</h2>
        </div>
        <p className="caption">
          Paint arbitrary cells for reconstruction, or apply suggested legal moves from the side panel.
        </p>
      </div>
      <div className="paint-chip-row">
        <span className="paint-label">Brush</span>
        <span className={`paint-chip ${selectedPaintColor ?? "empty"}`}>
          {selectedPaintColor ?? "erase"}
        </span>
      </div>
      <div className="board-grid" role="grid" aria-label="Blokus board">
        {board.map((row, rowIndex) =>
          row.map((cell, colIndex) => {
            const key = `${rowIndex}:${colIndex}`;
            const className = [
              "board-cell",
              cell ? `cell-${cell}` : "cell-empty",
              highlightedCells.has(key) ? "cell-highlighted" : ""
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
                onClick={() => onCellClick(rowIndex, colIndex)}
              />
            );
          })
        )}
      </div>
    </div>
  );
}

