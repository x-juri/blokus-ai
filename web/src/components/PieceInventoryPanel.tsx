import type { BoardState, PieceDescriptor, PlayerColor } from "../types/blokus";
import { countPlacedCells } from "../utils/state";

interface PieceInventoryPanelProps {
  boardState: BoardState;
  pieces: PieceDescriptor[];
  onTogglePiece: (color: PlayerColor, pieceId: string) => void;
}

export function PieceInventoryPanel({
  boardState,
  pieces,
  onTogglePiece
}: PieceInventoryPanelProps) {
  const placedCounts = countPlacedCells(boardState.board);

  return (
    <section className="panel inventory-panel">
      <div className="panel-header">
        <p className="eyebrow">Inventory</p>
        <h3>Remaining Pieces</h3>
      </div>
      <p className="caption">
        Toggle piece availability to match the real game state. Board paints do not infer piece usage.
      </p>
      <div className="inventory-columns">
        {(["blue", "yellow", "red", "green"] as PlayerColor[]).map((color) => (
          <div key={color} className={`inventory-column inventory-${color}`}>
            <div className="inventory-heading">
              <strong>{color}</strong>
              <span>{placedCounts[color]} cells on board</span>
            </div>
            <div className="piece-chip-grid">
              {pieces.map((piece) => {
                const enabled = boardState.remaining_pieces_by_color[color]?.includes(piece.piece_id);
                return (
                  <button
                    key={`${color}-${piece.piece_id}`}
                    type="button"
                    className={enabled ? "piece-chip enabled" : "piece-chip disabled"}
                    onClick={() => onTogglePiece(color, piece.piece_id)}
                  >
                    <span>{piece.piece_id}</span>
                    <small>{piece.size}</small>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

