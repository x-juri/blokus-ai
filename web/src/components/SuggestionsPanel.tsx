import type { MoveSuggestion } from "../types/blokus";

interface SuggestionsPanelProps {
  suggestions: MoveSuggestion[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  onApply: (index: number) => void;
}

export function SuggestionsPanel({
  suggestions,
  selectedIndex,
  onSelect,
  onApply
}: SuggestionsPanelProps) {
  const selected = suggestions[selectedIndex];

  return (
    <section className="panel">
      <div className="panel-header">
        <p className="eyebrow">Suggestions</p>
        <h3>Next Move Ranking</h3>
      </div>
      {suggestions.length === 0 ? (
        <p className="caption">Run search to populate move suggestions for the active color.</p>
      ) : (
        <>
          <div className="suggestion-list">
            {suggestions.map((suggestion, index) => (
              <button
                key={`${suggestion.move.piece_id}-${index}`}
                type="button"
                className={index === selectedIndex ? "suggestion-card active" : "suggestion-card"}
                onClick={() => onSelect(index)}
              >
                <div>
                  <strong>{suggestion.move.is_pass ? "PASS" : suggestion.move.piece_id}</strong>
                  <span>
                    {suggestion.move.is_pass
                      ? "no legal placements"
                      : `(${suggestion.move.anchor_cell.row}, ${suggestion.move.anchor_cell.col})`}
                  </span>
                </div>
                <div className="suggestion-metrics">
                  <span>{suggestion.score.toFixed(2)}</span>
                  <small>{suggestion.visits} visits</small>
                </div>
              </button>
            ))}
          </div>
          {selected ? (
            <div className="suggestion-detail">
              {selected.move.is_pass ? null : (
                <>
                  <p>
                    <strong>Rotation:</strong> {selected.move.rotation} degrees
                  </p>
                  <p>
                    <strong>Reflection:</strong> {selected.move.reflection ? "yes" : "no"}
                  </p>
                </>
              )}
              <p>{selected.rationale}</p>
              <button type="button" className="primary-button" onClick={() => onApply(selectedIndex)}>
                Apply suggestion
              </button>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}
