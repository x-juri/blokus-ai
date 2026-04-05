from __future__ import annotations

import json
from pathlib import Path
from typing import Union


def load_jsonl_traces(path: Union[str, Path]) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
