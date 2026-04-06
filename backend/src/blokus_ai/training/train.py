from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from blokus_ai.engine.models import BoardState
from blokus_ai.training.encoding import ACTION_SPACE_SIZE, encode_state
from blokus_ai.training.model import save_policy_value_checkpoint


def load_self_play_records(records_path: str | Path) -> list[dict[str, Any]]:
    path = Path(records_path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _dense_policy_target(record: dict[str, Any]) -> list[float]:
    target = [0.0] * ACTION_SPACE_SIZE
    visit_counts = {
        int(action_index): int(visits)
        for action_index, visits in record["visit_counts_by_action"].items()
    }
    total_visits = sum(visit_counts.values()) or 1
    for action_index, visits in visit_counts.items():
        target[action_index] = visits / total_visits
    return target


def train_policy_value_network(
    records_path: str | Path,
    checkpoint_id: str,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    report_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from blokus_ai.training.model import build_policy_value_network

    records = load_self_play_records(records_path)
    model = build_policy_value_network()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    spatial_tensors = []
    metadata_tensors = []
    policy_targets = []
    value_targets = []
    for record in records:
        state = BoardState.model_validate(record["state"])
        encoded = encode_state(state)
        spatial_tensors.append(torch.tensor(encoded.spatial, dtype=torch.float32))
        metadata_tensors.append(torch.tensor(encoded.metadata, dtype=torch.float32))
        policy_targets.append(torch.tensor(_dense_policy_target(record), dtype=torch.float32))
        value_targets.append(torch.tensor(float(record["final_value_target"]), dtype=torch.float32))

    if not spatial_tensors:
        raise ValueError("No self-play records were found.")

    spatial_dataset = torch.stack(spatial_tensors)
    metadata_dataset = torch.stack(metadata_tensors)
    policy_dataset = torch.stack(policy_targets)
    value_dataset = torch.stack(value_targets)

    losses: list[float] = []
    for _ in range(epochs):
        for start in range(0, len(records), batch_size):
            end = start + batch_size
            batch_spatial = spatial_dataset[start:end]
            batch_metadata = metadata_dataset[start:end]
            batch_policy = policy_dataset[start:end]
            batch_value = value_dataset[start:end]

            optimizer.zero_grad()
            policy_logits, values = model(batch_spatial, batch_metadata)
            policy_loss = -(batch_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(values, batch_value)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

    checkpoint_path = save_policy_value_checkpoint(
        model,
        checkpoint_id=checkpoint_id,
        metadata={
            "epochs": epochs,
            "batch_size": batch_size,
            "records_path": str(records_path),
            "mean_loss": sum(losses) / len(losses),
        },
    )

    report = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "records": len(records),
        "epochs": epochs,
        "batch_size": batch_size,
        "mean_loss": sum(losses) / len(losses),
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
