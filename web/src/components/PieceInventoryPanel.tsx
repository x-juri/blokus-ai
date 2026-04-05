import { PieceMiniature } from "./PieceMiniature";
import type { BoardState, PieceDescriptor, PlayerColor } from "../types/blokus";
import { countPlacedCells } from "../utils/state";

interface PieceInventoryPanelProps {
  boardState: BoardState;
  pieces: PieceDescriptor[];
  selectedPlacement: { color: PlayerColor; pieceId: string } | null;
  onSelectPiece: (color: PlayerColor, pieceId: string) => void;
  onRestorePiece: (color: PlayerColor, pieceId: string) => void;
}

export function PieceInventoryPanel({
  boardState,
  pieces,
  selectedPlacement,
  onSelectPiece,
  onRestorePiece
}: PieceInventoryPanelProps) {
  const placedCounts = countPlacedCells(boardState.board);

  return (
    <section className="panel inventory-panel">
      <div className="panel-header">
        <p className="eyebrow">Inventory</p>
        <h3>Remaining Pieces</h3>
      </div>
      <p className="caption">
        Remaining pieces are selectable for manual placement. Placed pieces are shown separately per
        color and can be restored to inventory if you need to correct the state.
      </p>
      <div className="inventory-columns">
        {(["blue", "yellow", "red", "green"] as PlayerColor[]).map((color) => {
          const remainingIds = new Set(boardState.remaining_pieces_by_color[color] ?? []);
          const remainingPieces = pieces.filter((piece) => remainingIds.has(piece.piece_id));
          const placedPieces = pieces.filter((piece) => !remainingIds.has(piece.piece_id));

          return (
            <div key={color} className={`inventory-column inventory-${color}`}>
              <div className="inventory-heading">
                <strong>{color}</strong>
                <span>{placedCounts[color]} cells on board</span>
              </div>

              <div className="inventory-section">
                <div className="inventory-subheading">
                  <span>Remaining</span>
                  <small>{remainingPieces.length}</small>
                </div>
                <div className="piece-card-grid">
                  {remainingPieces.map((piece) => {
                    const selected =
                      selectedPlacement?.color === color && selectedPlacement.pieceId === piece.piece_id;
                    return (
                      <button
                        key={`${color}-remaining-${piece.piece_id}`}
                        type="button"
                        className={selected ? "piece-card selected" : "piece-card"}
                        onClick={() => onSelectPiece(color, piece.piece_id)}
                      >
                        <PieceMiniature cells={piece.cells} color={color} compact />
                        <div className="piece-card-meta">
                          <span>{piece.piece_id}</span>
                          <small>{piece.size} cells</small>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="inventory-section">
                <div className="inventory-subheading">
                  <span>Placed</span>
                  <small>{placedPieces.length}</small>
                </div>
                <div className="piece-card-grid placed">
                  {placedPieces.map((piece) => (
                    <button
                      key={`${color}-placed-${piece.piece_id}`}
                      type="button"
                      className="piece-card placed"
                      onClick={() => onRestorePiece(color, piece.piece_id)}
                    >
                      <PieceMiniature cells={piece.cells} color={color} compact />
                      <div className="piece-card-meta">
                        <span>{piece.piece_id}</span>
                        <small>restore</small>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
