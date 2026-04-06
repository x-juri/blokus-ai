from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from blokus_ai.engine.models import Move, MoveSuggestion


@dataclass
class AgentDecision:
    chosen_move: Optional[Move]
    suggestions: list[MoveSuggestion]
    diagnostics: dict[str, Any] = field(default_factory=dict)
