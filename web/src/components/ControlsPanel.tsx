import type { BoardState, PlayerColor } from "../types/blokus";
import { playerColors } from "../constants/pieces";

interface ControlsPanelProps {
  boardState: BoardState;
  topK: number;
  simulations: number;
  candidateLimit: number;
  statusMessage: string;
  errorMessage: string;
  loading: boolean;
  selectedPaintColor: PlayerColor | null;
  onStateChange: (nextState: BoardState) => void;
  onTopKChange: (value: number) => void;
  onSimulationsChange: (value: number) => void;
  onCandidateLimitChange: (value: number) => void;
  onPaintColorChange: (color: PlayerColor | null) => void;
  onSuggest: () => void;
  onReset: () => void;
  onPass: () => void;
}

export function ControlsPanel({
  boardState,
  topK,
  simulations,
  candidateLimit,
  statusMessage,
  errorMessage,
  loading,
  selectedPaintColor,
  onStateChange,
  onTopKChange,
  onSimulationsChange,
  onCandidateLimitChange,
  onPaintColorChange,
  onSuggest,
  onReset,
  onPass
}: ControlsPanelProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <p className="eyebrow">Controls</p>
        <h3>Search Setup</h3>
      </div>

      <label className="field">
        <span>Variant</span>
        <select
          value={boardState.variant}
          onChange={(event) =>
            onStateChange({
              ...boardState,
              variant: event.target.value as BoardState["variant"]
            })
          }
        >
          <option value="standard-4">Standard 4-player</option>
          <option value="paired-2">2-player teams</option>
          <option value="shared-3">3-player shared color</option>
        </select>
      </label>

      <label className="field">
        <span>Active color</span>
        <select
          value={boardState.active_color}
          onChange={(event) =>
            onStateChange({
              ...boardState,
              active_color: event.target.value as PlayerColor
            })
          }
        >
          {playerColors.map((color) => (
            <option key={color} value={color}>
              {color}
            </option>
          ))}
        </select>
      </label>

      {boardState.variant === "shared-3" ? (
        <label className="field">
          <span>Shared color</span>
          <select
            value={boardState.shared_color ?? "green"}
            onChange={(event) =>
              onStateChange({
                ...boardState,
                shared_color: event.target.value as PlayerColor
              })
            }
          >
            {playerColors.map((color) => (
              <option key={color} value={color}>
                {color}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      <div className="field-grid">
        <label className="field">
          <span>Top suggestions</span>
          <input
            type="number"
            min={1}
            max={10}
            value={topK}
            onChange={(event) => onTopKChange(Number(event.target.value))}
          />
        </label>
        <label className="field">
          <span>MCTS simulations</span>
          <input
            type="number"
            min={8}
            max={512}
            step={8}
            value={simulations}
            onChange={(event) => onSimulationsChange(Number(event.target.value))}
          />
        </label>
      </div>

      <div className="field-grid">
        <label className="field">
          <span>Candidate limit</span>
          <input
            type="number"
            min={4}
            max={48}
            value={candidateLimit}
            onChange={(event) => onCandidateLimitChange(Number(event.target.value))}
          />
        </label>
        <label className="field">
          <span>Passes in row</span>
          <input
            type="number"
            min={0}
            max={4}
            value={boardState.passes_in_row}
            onChange={(event) =>
              onStateChange({
                ...boardState,
                passes_in_row: Number(event.target.value)
              })
            }
          />
        </label>
      </div>

      <div className="brush-row">
        <span className="brush-label">Board brush</span>
        <div className="brush-pills">
          <button
            type="button"
            className={selectedPaintColor === null ? "brush active" : "brush"}
            onClick={() => onPaintColorChange(null)}
          >
            erase
          </button>
          {playerColors.map((color) => (
            <button
              key={color}
              type="button"
              className={selectedPaintColor === color ? `brush active ${color}` : `brush ${color}`}
              onClick={() => onPaintColorChange(color)}
            >
              {color}
            </button>
          ))}
        </div>
      </div>

      <div className="actions">
        <button type="button" className="primary-button" onClick={onSuggest} disabled={loading}>
          {loading ? "Searching..." : "Suggest moves"}
        </button>
        <button type="button" className="secondary-button" onClick={onPass}>
          Pass turn
        </button>
        <button type="button" className="secondary-button" onClick={onReset}>
          Reset board
        </button>
      </div>

      <div className="message-stack">
        <p className="status-message">{statusMessage}</p>
        {errorMessage ? <p className="error-message">{errorMessage}</p> : null}
      </div>
    </section>
  );
}

