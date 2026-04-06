import { useState } from "react";

import { PlayMode } from "./components/PlayMode";
import { WorkbenchMode } from "./components/WorkbenchMode";
import type { AppMode } from "./types/blokus";

function App() {
  const [mode, setMode] = useState<AppMode>("workbench");

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Blokus AI</p>
          <h1>{mode === "workbench" ? "Move Suggestion Workbench" : "Paired-2 Play Lab"}</h1>
        </div>
        <div className="hero-actions">
          <p className="hero-copy">
            Exact rules, search-first move ranking, paired-team gameplay, and the first reinforcement
            learning pipeline for policy-guided play.
          </p>
          <div className="segmented-toggle mode-toggle">
            <button
              type="button"
              className={mode === "workbench" ? "toggle-button active" : "toggle-button"}
              onClick={() => setMode("workbench")}
            >
              Workbench
            </button>
            <button
              type="button"
              className={mode === "play" ? "toggle-button active" : "toggle-button"}
              onClick={() => setMode("play")}
            >
              Play
            </button>
          </div>
        </div>
      </header>

      {mode === "workbench" ? <WorkbenchMode /> : <PlayMode />}
    </div>
  );
}

export default App;
