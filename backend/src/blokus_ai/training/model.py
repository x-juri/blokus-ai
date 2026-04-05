from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def build_policy_value_network(board_size: int = 20, channels: int = 7) -> Any:
    """Create a lightweight CNN lazily so importing the backend does not require torch."""

    import torch
    from torch import nn

    class PolicyValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            flattened = 64 * board_size * board_size
            self.policy_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
            self.value_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Tanh(),
            )

        def forward(self, inputs: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            encoded = self.encoder(inputs)
            return self.policy_head(encoded), self.value_head(encoded)

    return PolicyValueNet()

