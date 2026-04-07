from __future__ import annotations

import argparse
import json
import random
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


def _sparse_policy_target(record: dict[str, Any]) -> dict[int, float]:
    visit_counts = {
        int(action_index): int(visits)
        for action_index, visits in record["visit_counts_by_action"].items()
    }
    total_visits = sum(visit_counts.values()) or 1
    return {
        action_index: visits / total_visits
        for action_index, visits in visit_counts.items()
    }


def _split_record_indices(
    total_records: int,
    validation_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices = list(range(total_records))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if total_records < 2 or validation_split <= 0.0:
        return indices, []

    validation_size = max(1, int(total_records * validation_split))
    validation_size = min(validation_size, total_records - 1)
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return train_indices, validation_indices


def _policy_loss_from_sparse_targets(
    log_policy: Any,
    sparse_targets: list[dict[int, float]],
) -> Any:
    import torch

    per_row_losses = []
    for row_index, target in enumerate(sparse_targets):
        action_indices = torch.tensor(list(target.keys()), dtype=torch.long, device=log_policy.device)
        action_weights = torch.tensor(list(target.values()), dtype=torch.float32, device=log_policy.device)
        per_row_losses.append(-(log_policy[row_index, action_indices] * action_weights).sum())
    return torch.stack(per_row_losses).mean()


def _run_batches(
    model: Any,
    spatial_dataset: Any,
    metadata_dataset: Any,
    sparse_policy_targets: list[dict[int, float]],
    value_dataset: Any,
    indices: list[int],
    batch_size: int,
    optimizer: Optional[Any] = None,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_examples = 0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        index_tensor = torch.tensor(batch_indices, dtype=torch.long)
        batch_spatial = spatial_dataset.index_select(0, index_tensor)
        batch_metadata = metadata_dataset.index_select(0, index_tensor)
        batch_values = value_dataset.index_select(0, index_tensor)
        batch_sparse_targets = [sparse_policy_targets[index] for index in batch_indices]

        if optimizer is not None:
            optimizer.zero_grad()

        context = torch.enable_grad() if optimizer is not None else torch.no_grad()
        with context:
            policy_logits, values = model(batch_spatial, batch_metadata)
            log_policy = F.log_softmax(policy_logits, dim=1)
            policy_loss = _policy_loss_from_sparse_targets(log_policy, batch_sparse_targets)
            value_loss = F.mse_loss(values, batch_values)
            loss = policy_loss + value_loss

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_examples = len(batch_indices)
        total_examples += batch_examples
        total_policy_loss += float(policy_loss.item()) * batch_examples
        total_value_loss += float(value_loss.item()) * batch_examples
        total_loss += float(loss.item()) * batch_examples

    return {
        "policy_loss": total_policy_loss / max(total_examples, 1),
        "value_loss": total_value_loss / max(total_examples, 1),
        "total_loss": total_loss / max(total_examples, 1),
    }


def train_policy_value_network(
    records_path: str | Path,
    checkpoint_id: str,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    validation_split: float = 0.1,
    seed: int = 7,
    report_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    import torch

    from blokus_ai.training.model import build_policy_value_network

    records = load_self_play_records(records_path)
    model = build_policy_value_network()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    spatial_tensors = []
    metadata_tensors = []
    sparse_policy_targets = []
    value_targets = []
    for record in records:
        state = BoardState.model_validate(record["state"])
        encoded = encode_state(state)
        spatial_tensors.append(torch.tensor(encoded.spatial, dtype=torch.float32))
        metadata_tensors.append(torch.tensor(encoded.metadata, dtype=torch.float32))
        sparse_policy_targets.append(_sparse_policy_target(record))
        value_targets.append(torch.tensor(float(record["final_value_target"]), dtype=torch.float32))

    if not spatial_tensors:
        raise ValueError("No self-play records were found.")

    spatial_dataset = torch.stack(spatial_tensors)
    metadata_dataset = torch.stack(metadata_tensors)
    value_dataset = torch.stack(value_targets)
    train_indices, validation_indices = _split_record_indices(
        total_records=len(records),
        validation_split=validation_split,
        seed=seed,
    )

    epoch_metrics: list[dict[str, Any]] = []
    mean_losses: list[float] = []
    shuffle_rng = random.Random(seed)
    for epoch_index in range(epochs):
        shuffled_train_indices = train_indices[:]
        shuffle_rng.shuffle(shuffled_train_indices)
        train_metrics = _run_batches(
            model=model,
            spatial_dataset=spatial_dataset,
            metadata_dataset=metadata_dataset,
            sparse_policy_targets=sparse_policy_targets,
            value_dataset=value_dataset,
            indices=shuffled_train_indices,
            batch_size=batch_size,
            optimizer=optimizer,
        )
        validation_metrics = None
        if validation_indices:
            validation_metrics = _run_batches(
                model=model,
                spatial_dataset=spatial_dataset,
                metadata_dataset=metadata_dataset,
                sparse_policy_targets=sparse_policy_targets,
                value_dataset=value_dataset,
                indices=validation_indices,
                batch_size=batch_size,
                optimizer=None,
            )
        mean_losses.append(train_metrics["total_loss"])
        epoch_metrics.append(
            {
                "epoch": epoch_index + 1,
                "train_policy_loss": train_metrics["policy_loss"],
                "train_value_loss": train_metrics["value_loss"],
                "train_total_loss": train_metrics["total_loss"],
                "validation_policy_loss": None if validation_metrics is None else validation_metrics["policy_loss"],
                "validation_value_loss": None if validation_metrics is None else validation_metrics["value_loss"],
                "validation_total_loss": None if validation_metrics is None else validation_metrics["total_loss"],
            }
        )

    final_epoch = epoch_metrics[-1]
    best_epoch = min(
        epoch_metrics,
        key=lambda metrics: (
            float("inf")
            if metrics["validation_total_loss"] is None
            else metrics["validation_total_loss"],
            metrics["train_total_loss"],
        ),
    )

    checkpoint_metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "records_path": str(records_path),
        "mean_loss": sum(mean_losses) / len(mean_losses),
        "learning_rate": learning_rate,
        "validation_split": validation_split,
        "seed": seed,
        "train_records": len(train_indices),
        "validation_records": len(validation_indices),
        "train_policy_loss": final_epoch["train_policy_loss"],
        "train_value_loss": final_epoch["train_value_loss"],
        "train_total_loss": final_epoch["train_total_loss"],
        "validation_policy_loss": final_epoch["validation_policy_loss"],
        "validation_value_loss": final_epoch["validation_value_loss"],
        "validation_total_loss": final_epoch["validation_total_loss"],
        "best_epoch": best_epoch["epoch"],
        "best_validation_total_loss": best_epoch["validation_total_loss"],
        "epoch_metrics": epoch_metrics,
    }
    checkpoint_path = save_policy_value_checkpoint(
        model,
        checkpoint_id=checkpoint_id,
        metadata=checkpoint_metadata,
    )

    report = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "records": len(records),
        "train_records": len(train_indices),
        "validation_records": len(validation_indices),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_split": validation_split,
        "seed": seed,
        "mean_loss": sum(mean_losses) / len(mean_losses),
        "train_policy_loss": final_epoch["train_policy_loss"],
        "train_value_loss": final_epoch["train_value_loss"],
        "train_total_loss": final_epoch["train_total_loss"],
        "validation_policy_loss": final_epoch["validation_policy_loss"],
        "validation_value_loss": final_epoch["validation_value_loss"],
        "validation_total_loss": final_epoch["validation_total_loss"],
        "best_epoch": best_epoch["epoch"],
        "best_validation_total_loss": best_epoch["validation_total_loss"],
        "epoch_metrics": epoch_metrics,
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
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
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
        validation_split=args.validation_split,
        seed=args.seed,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
