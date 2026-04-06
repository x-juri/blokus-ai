from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

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
            "learning_rate": learning_rate,
        },
    )

    report = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "records": len(records),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mean_loss": sum(losses) / len(losses),
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Blokus policy/value checkpoint from JSONL traces.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser


def main() -> None:
    parser = build_train_arg_parser()
    args = parser.parse_args()
    report = train_policy_value_network(
        records_path=args.records,
        checkpoint_id=args.checkpoint_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
