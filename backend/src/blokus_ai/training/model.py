from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from blokus_ai.training.encoding import ACTION_SPACE_SIZE, NON_SPATIAL_FEATURES, SPATIAL_CHANNELS

if TYPE_CHECKING:
    import torch


CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "checkpoints"


@dataclass(frozen=True)
class LoadedCheckpoint:
    checkpoint_id: str
    path: Path
    metadata: dict[str, Any]
    model: Any


def build_policy_value_network(
    board_size: int = 20,
    spatial_channels: int = SPATIAL_CHANNELS,
    metadata_features: int = NON_SPATIAL_FEATURES,
    action_size: int = ACTION_SPACE_SIZE,
) -> Any:
    """Create the policy/value network lazily so importing the backend does not require torch."""

    import torch
    from torch import nn

    class PolicyValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((5, 5)),
            )
            self.spatial_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 5 * 5, 256),
                nn.ReLU(),
            )
            self.metadata_head = nn.Sequential(
                nn.Linear(metadata_features, 128),
                nn.ReLU(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(256 + 128, 512),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(512, action_size)
            self.value_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Tanh(),
            )

        def forward(
            self,
            spatial_inputs: "torch.Tensor",
            metadata_inputs: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            spatial_features = self.spatial_head(self.spatial_encoder(spatial_inputs))
            metadata_features = self.metadata_head(metadata_inputs)
            combined = self.trunk(torch.cat([spatial_features, metadata_features], dim=1))
            policy_logits = self.policy_head(combined)
            value = self.value_head(combined).squeeze(-1)
            return policy_logits, value

    return PolicyValueNet()


def checkpoint_directory() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR


def resolve_checkpoint_path(checkpoint_id: Optional[str]) -> Optional[Path]:
    checkpoint_root = checkpoint_directory()
    if checkpoint_id is None:
        checkpoints = sorted(checkpoint_root.glob("*.pt"))
        return checkpoints[-1] if checkpoints else None

    raw_path = Path(checkpoint_id)
    candidates = [
        raw_path,
        checkpoint_root / checkpoint_id,
        checkpoint_root / f"{checkpoint_id}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def save_policy_value_checkpoint(
    model: Any,
    checkpoint_id: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    import torch

    path = checkpoint_directory() / f"{checkpoint_id}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )
    _load_policy_value_checkpoint_cached.cache_clear()
    return path


@lru_cache(maxsize=8)
def _load_policy_value_checkpoint_cached(
    path_string: str,
    mtime_ns: int,
    size: int,
) -> LoadedCheckpoint:
    import torch

    path = Path(path_string)
    _ = (mtime_ns, size)
    payload = torch.load(path, map_location="cpu")
    model = build_policy_value_network()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LoadedCheckpoint(
        checkpoint_id=path.stem,
        path=path,
        metadata=payload.get("metadata", {}),
        model=model,
    )


def load_policy_value_checkpoint(checkpoint_id: Optional[str] = None) -> Optional[LoadedCheckpoint]:
    path = resolve_checkpoint_path(checkpoint_id)
    if path is None:
        return None

    stat = path.stat()
    return _load_policy_value_checkpoint_cached(
        str(path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
    )
