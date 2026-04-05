import type { PlayerColor } from "../types/blokus";

interface PieceMiniatureProps {
  cells: [number, number][];
  color?: PlayerColor;
  compact?: boolean;
}

export function PieceMiniature({
  cells,
  color = "blue",
  compact = false
}: PieceMiniatureProps) {
  if (!cells.length) {
    return null;
  }

  const maxRow = Math.max(...cells.map(([row]) => row));
  const maxCol = Math.max(...cells.map(([, col]) => col));

  return (
    <div
      className={compact ? "piece-miniature compact" : "piece-miniature"}
      style={{
        gridTemplateColumns: `repeat(${maxCol + 1}, 1fr)`,
        gridTemplateRows: `repeat(${maxRow + 1}, 1fr)`
      }}
      aria-hidden="true"
    >
      {Array.from({ length: (maxRow + 1) * (maxCol + 1) }, (_, index) => {
        const row = Math.floor(index / (maxCol + 1));
        const col = index % (maxCol + 1);
        const filled = cells.some(([cellRow, cellCol]) => cellRow === row && cellCol === col);
        return (
          <span
            key={`${row}-${col}`}
            className={
              filled
                ? `piece-mini-cell filled ${color}`
                : "piece-mini-cell empty"
            }
          />
        );
      })}
    </div>
  );
}
