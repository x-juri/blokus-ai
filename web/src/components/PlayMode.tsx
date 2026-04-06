import { useEffect, useState } from "react";

import { applyMove, fetchPieces, fetchReplayGame, newGame, runAiTurn } from "../api/client";
import type {
  AiPresetId,
  BoardState,
  Coordinate,
  Move,
  PieceDescriptor,
  PlacementSelection,
  ReplayGameResponse,
  TeamId
} from "../types/blokus";
import { aiPresetDescriptions, aiPresetLabels, buildLiveAgentConfig, buildReplayAgents } from "../utils/agents";
import { coordinatesToSet, isPlacementValid, pieceCellsForId, placePieceAtAnchor } from "../utils/pieces";
import { humanControlsColor, isTerminalState, summarizeGame, teamForColor } from "../utils/state";
import { BoardGrid } from "./BoardGrid";
import { PieceMiniature } from "./PieceMiniature";

const pairedConfig = { variant: "paired-2" } as const;

function formatMove(move: Move, index: number): string {
  if (move.is_pass) {
    return `${index + 1}. ${move.color} pass`;
  }
  return `${index + 1}. ${move.color} ${move.piece_id} @ ${move.anchor_cell.row},${move.anchor_cell.col}`;
}

function describeAiTurn(move: Move | null, forcedPassCount: number): string {
  if (!move) {
    return forcedPassCount
      ? `Forced passes advanced the game by ${forcedPassCount} turn${forcedPassCount === 1 ? "" : "s"}.`
      : "The turn advanced without an AI placement.";
  }
  const suffix = forcedPassCount
    ? ` ${forcedPassCount} forced pass${forcedPassCount === 1 ? "" : "es"} followed automatically.`
    : "";
  return move.is_pass
    ? `${move.color} had to pass.${suffix}`
    : `${move.color} played ${move.piece_id}.${suffix}`;
}

export function PlayMode() {
  const [pieces, setPieces] = useState<PieceDescriptor[]>([]);
  const [gameState, setGameState] = useState<BoardState | null>(null);
  const [humanSide, setHumanSide] = useState<TeamId>("player_a");
  const [selectedPlacement, setSelectedPlacement] = useState<PlacementSelection | null>(null);
  const [hoveredCell, setHoveredCell] = useState<Coordinate | null>(null);
  const [busy, setBusy] = useState(false);
  const [statusMessage, setStatusMessage] = useState(
    "Start a paired-2 game and choose which opposite-color team you want to control."
  );
  const [errorMessage, setErrorMessage] = useState("");
  const [replayData, setReplayData] = useState<ReplayGameResponse | null>(null);
  const [replayIndex, setReplayIndex] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replaySeed, setReplaySeed] = useState(7);
  const [replayBusy, setReplayBusy] = useState(false);
  const [livePreset, setLivePreset] = useState<AiPresetId>("fast");
  const [replayPreset, setReplayPreset] = useState<AiPresetId>("fast");

  useEffect(() => {
    void fetchPieces().then(setPieces);
  }, []);

  const isHumanTurn =
    gameState !== null && humanControlsColor(humanSide, gameState.active_color);
  const selectedPieceCells =
    selectedPlacement && gameState
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
    gameState !== null &&
    previewPlacement.length > 0 &&
    !isPlacementValid(gameState.board, previewPlacement);
  const liveSummary = gameState ? summarizeGame(gameState) : null;

  const replayState =
    replayData?.state_history[Math.min(replayIndex, Math.max(replayData.state_history.length - 1, 0))] ??
    null;

  useEffect(() => {
    if (!gameState || busy || isTerminalState(gameState) || isHumanTurn) {
      return;
    }

    const currentState = gameState;
    let cancelled = false;
    async function takeAiTurn() {
      setBusy(true);
      setErrorMessage("");
      try {
        const response = await runAiTurn(currentState, buildLiveAgentConfig(livePreset));
        if (cancelled) {
          return;
        }
        setGameState(response.state);
        setSelectedPlacement(null);
        setStatusMessage(
          describeAiTurn(response.move, response.forced_passes.length)
        );
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(error instanceof Error ? error.message : "AI turn failed.");
        }
      } finally {
        if (!cancelled) {
          setBusy(false);
        }
      }
    }

    void takeAiTurn();
    return () => {
      cancelled = true;
    };
  }, [busy, gameState, isHumanTurn, livePreset]);

  useEffect(() => {
    if (!replayPlaying || !replayData) {
      return;
    }
    const timer = window.setInterval(() => {
      setReplayIndex((currentIndex) => {
        const lastIndex = Math.max(replayData.state_history.length - 1, 0);
        if (currentIndex >= lastIndex) {
          setReplayPlaying(false);
          return currentIndex;
        }
        return currentIndex + 1;
      });
    }, 700);
    return () => window.clearInterval(timer);
  }, [replayData, replayPlaying]);

  useEffect(() => {
    if (gameState && selectedPlacement && selectedPlacement.color !== gameState.active_color) {
      setSelectedPlacement(null);
    }
  }, [gameState, selectedPlacement]);

  async function startGame(nextHumanSide: TeamId) {
    setBusy(true);
    setErrorMessage("");
    try {
      const state = await newGame(pairedConfig);
      setHumanSide(nextHumanSide);
      setGameState(state);
      setSelectedPlacement(null);
      setHoveredCell(null);
      setStatusMessage(
        `Started a paired-2 game. You control ${nextHumanSide === "player_a" ? "blue/red" : "yellow/green"}. `
          + `Live AI preset: ${aiPresetLabels[livePreset]}.`
      );
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not create a new game.");
    } finally {
      setBusy(false);
    }
  }

  function selectPiece(pieceId: string) {
    if (!gameState || !isHumanTurn) {
      return;
    }
    const shouldClear = selectedPlacement?.pieceId === pieceId;
    setSelectedPlacement(
      shouldClear
        ? null
        : {
            color: gameState.active_color,
            pieceId,
            rotation: 0,
            reflection: false
          }
    );
    setErrorMessage("");
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

  async function handleHumanPass() {
    if (!gameState) {
      return;
    }
    const move: Move = {
      color: gameState.active_color,
      piece_id: "PASS",
      anchor_cell: { row: 0, col: 0 },
      rotation: 0,
      reflection: false,
      is_pass: true
    };
    try {
      const nextState = await applyMove(gameState, move);
      setGameState(nextState);
      setSelectedPlacement(null);
      setStatusMessage(`${move.color} passed.`);
      setErrorMessage("");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Pass failed.");
    }
  }

  async function handleBoardClick(row: number, col: number) {
    if (!gameState) {
      return;
    }
    if (!isHumanTurn) {
      setStatusMessage("The AI is thinking for the active team.");
      return;
    }
    if (!selectedPlacement) {
      setStatusMessage("Select one of the active color's remaining pieces first.");
      return;
    }

    const move: Move = {
      color: gameState.active_color,
      piece_id: selectedPlacement.pieceId,
      anchor_cell: { row, col },
      rotation: selectedPlacement.rotation,
      reflection: selectedPlacement.reflection,
      is_pass: false
    };

    try {
      const nextState = await applyMove(gameState, move);
      setGameState(nextState);
      setSelectedPlacement(null);
      setErrorMessage("");
      setStatusMessage(`${move.color} placed ${move.piece_id}.`);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "That move is illegal.");
    }
  }

  async function generateReplay() {
    setReplayBusy(true);
    setErrorMessage("");
    try {
      const replay = await fetchReplayGame(replaySeed, buildReplayAgents(replayPreset), pairedConfig);
      setReplayData(replay);
      setReplayIndex(0);
      setReplayPlaying(false);
      setStatusMessage(
        `Generated AI replay with seed ${replay.seed} using the ${aiPresetLabels[replayPreset]} preset.`
      );
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Replay generation failed.");
    } finally {
      setReplayBusy(false);
    }
  }

  const activePieces =
    gameState && isHumanTurn
      ? pieces.filter((piece) =>
          gameState.remaining_pieces_by_color[gameState.active_color].includes(piece.piece_id)
        )
      : [];

  return (
    <div className="play-stack">
      <div className="layout-grid">
        <BoardGrid
          board={gameState?.board ?? Array.from({ length: 20 }, () => Array.from({ length: 20 }, () => null))}
          selectedPlacement={selectedPlacement}
          onCellClick={handleBoardClick}
          onCellHover={(row, col) => setHoveredCell({ row, col })}
          onBoardLeave={() => setHoveredCell(null)}
          suggestionCells={new Set<string>()}
          previewCells={previewCells}
          previewInvalid={Boolean(previewInvalid)}
          eyebrow="Play"
          title="Paired-2 Match"
          caption="You control one opposite-color team. Manual placement is only enabled for the active human color."
          chip={{
            label: "Turn",
            tone: gameState?.active_color ?? "empty",
            value: gameState ? `${gameState.active_color} (${teamForColor(gameState.active_color)})` : "start a game"
          }}
        />

        <div className="sidebar-stack">
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Game</p>
                <h3>Human Vs AI</h3>
              </div>
            </div>

            <div className="segmented-toggle">
              <button
                type="button"
                className={humanSide === "player_a" ? "toggle-button active" : "toggle-button"}
                onClick={() => void startGame("player_a")}
                disabled={busy}
              >
                Play blue/red
              </button>
              <button
                type="button"
                className={humanSide === "player_b" ? "toggle-button active" : "toggle-button"}
                onClick={() => void startGame("player_b")}
                disabled={busy}
              >
                Play yellow/green
              </button>
            </div>

            <label className="field">
              <span>AI speed</span>
              <select
                value={livePreset}
                onChange={(event) => setLivePreset(event.target.value as AiPresetId)}
              >
                {(["fast", "balanced", "strong"] as AiPresetId[]).map((preset) => (
                  <option key={preset} value={preset}>
                    {aiPresetLabels[preset]}
                  </option>
                ))}
              </select>
            </label>
            <p className="caption">{aiPresetDescriptions[livePreset]}</p>

            {gameState ? (
              <div className="game-meta-grid">
                <div className="meta-card">
                  <span>Active color</span>
                  <strong>{gameState.active_color}</strong>
                </div>
                <div className="meta-card">
                  <span>Active team</span>
                  <strong>{teamForColor(gameState.active_color)}</strong>
                </div>
                <div className="meta-card">
                  <span>Status</span>
                  <strong>{isTerminalState(gameState) ? "finished" : isHumanTurn ? "your turn" : "AI turn"}</strong>
                </div>
                <div className="meta-card">
                  <span>Passes in row</span>
                  <strong>{gameState.passes_in_row}</strong>
                </div>
              </div>
            ) : null}

            {liveSummary ? (
              <div className="score-summary">
                <div className="score-pill">
                  <span>player_a</span>
                  <strong>{liveSummary.group_scores.player_a ?? 0}</strong>
                </div>
                <div className="score-pill">
                  <span>player_b</span>
                  <strong>{liveSummary.group_scores.player_b ?? 0}</strong>
                </div>
              </div>
            ) : null}

            <div className="message-stack">
              <p className="status-message">{statusMessage}</p>
              {errorMessage ? <p className="error-message">{errorMessage}</p> : null}
            </div>

            <div className="actions">
              <button
                type="button"
                className="secondary-button"
                onClick={handleHumanPass}
                disabled={!gameState || !isHumanTurn}
              >
                Pass active color
              </button>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Placement</p>
                <h3>Active Pieces</h3>
              </div>
            </div>

            {!gameState ? (
              <p className="caption">Start a game to populate the active team's remaining pieces.</p>
            ) : !isHumanTurn ? (
              <p className="caption">Piece placement unlocks when the active color belongs to your team.</p>
            ) : (
              <>
                <div className="placement-summary">
                  <div>
                    <span className="brush-label">Selection</span>
                    <p className="placement-copy">
                      {selectedPlacement
                        ? `${selectedPlacement.color} ${selectedPlacement.pieceId}`
                        : `Choose a remaining ${gameState.active_color} piece.`}
                    </p>
                  </div>
                  {selectedPlacement ? (
                    <div className={`placement-chip ${selectedPlacement.color}`}>
                      {selectedPlacement.rotation}° {selectedPlacement.reflection ? "flip" : "normal"}
                    </div>
                  ) : null}
                </div>

                {selectedPlacement ? (
                  <div className="placement-preview">
                    <PieceMiniature cells={selectedPieceCells} color={selectedPlacement.color} />
                  </div>
                ) : null}

                <div className="brush-pills">
                  <button type="button" className="brush" onClick={rotateSelection} disabled={!selectedPlacement}>
                    rotate
                  </button>
                  <button type="button" className="brush" onClick={flipSelection} disabled={!selectedPlacement}>
                    flip
                  </button>
                  <button type="button" className="brush" onClick={clearSelection}>
                    clear
                  </button>
                </div>

                <div className="piece-card-grid">
                  {activePieces.map((piece) => {
                    const selected = selectedPlacement?.pieceId === piece.piece_id;
                    return (
                      <button
                        key={piece.piece_id}
                        type="button"
                        className={selected ? "piece-card selected" : "piece-card"}
                        onClick={() => selectPiece(piece.piece_id)}
                      >
                        <PieceMiniature cells={piece.cells} color={gameState.active_color} compact />
                        <div className="piece-card-meta">
                          <span>{piece.piece_id}</span>
                          <small>{piece.size} cells</small>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </section>

          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Moves</p>
                <h3>Live Log</h3>
              </div>
            </div>
            {gameState?.move_history.length ? (
              <div className="move-log">
                {gameState.move_history.slice(-10).map((move, index) => (
                  <div key={`${move.color}-${move.piece_id}-${index}`} className="move-log-row">
                    {formatMove(move, gameState.move_history.length - Math.min(10, gameState.move_history.length) + index)}
                  </div>
                ))}
              </div>
            ) : (
              <p className="caption">No turns recorded yet.</p>
            )}
          </section>
        </div>
      </div>

      <section className="panel replay-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Replay</p>
            <h3>AI Vs AI</h3>
          </div>
          <p className="caption">
            Generates a full paired-2 game and lets you play it back locally with the recorded state history.
          </p>
        </div>

        <div className="replay-toolbar">
          <label className="field">
            <span>Replay preset</span>
            <select
              value={replayPreset}
              onChange={(event) => setReplayPreset(event.target.value as AiPresetId)}
            >
              {(["fast", "balanced", "strong"] as AiPresetId[]).map((preset) => (
                <option key={preset} value={preset}>
                  {aiPresetLabels[preset]}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Seed</span>
            <input
              type="number"
              value={replaySeed}
              onChange={(event) => setReplaySeed(Number(event.target.value))}
            />
          </label>
          <button type="button" className="primary-button" onClick={() => void generateReplay()} disabled={replayBusy}>
            {replayBusy ? "Generating..." : "Generate replay"}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setReplayPlaying((current) => !current)}
            disabled={!replayData}
          >
            {replayPlaying ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setReplayIndex((current) => Math.max(current - 1, 0))}
            disabled={!replayData}
          >
            Step back
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() =>
              setReplayIndex((current) =>
                replayData ? Math.min(current + 1, replayData.state_history.length - 1) : current
              )
            }
            disabled={!replayData}
          >
            Step forward
          </button>
        </div>

        {replayState && replayData ? (
          <div className="replay-grid">
            <BoardGrid
              board={replayState.board}
              selectedPlacement={null}
              onCellClick={(_row, _col) => undefined}
              onCellHover={(_row, _col) => undefined}
              onBoardLeave={() => undefined}
              suggestionCells={new Set<string>()}
              previewCells={new Set<string>()}
              previewInvalid={false}
              eyebrow="Replay"
              title="Generated Match"
              caption="The replay view steps through the stored state history returned by the backend."
              chip={{
                label: "Ply",
                tone: replayState.active_color,
                value: `${replayIndex}/${Math.max(replayData.state_history.length - 1, 0)}`
              }}
            />

            <div className="replay-side">
              <div className="move-log replay-log">
                {replayData.moves.map((move, index) => (
                  <div
                    key={`${move.color}-${move.piece_id}-${index}`}
                    className={index + 1 === replayIndex ? "move-log-row active" : "move-log-row"}
                  >
                    {formatMove(move, index)}
                  </div>
                ))}
              </div>

              <div className="score-summary">
                <div className="score-pill">
                  <span>player_a</span>
                  <strong>{replayData.result.group_scores.player_a ?? 0}</strong>
                </div>
                <div className="score-pill">
                  <span>player_b</span>
                  <strong>{replayData.result.group_scores.player_b ?? 0}</strong>
                </div>
              </div>

              <p className="status-message">
                Winner: {replayData.result.winner_group ?? "draw"}.
              </p>
            </div>
          </div>
        ) : (
          <p className="caption">Generate a replay to inspect a full AI-vs-AI paired-2 game.</p>
        )}
      </section>
    </div>
  );
}
