"""Dataset loaders (Hugging Face, JSONL, CSV) with a field mapping onto `Sample`."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from owl.types import Sample


class FieldMap(BaseModel):
    """Which columns of a source row become which parts of a `Sample`.

    `id` is optional; when absent, row indices are used. Any column named in
    `metadata` is copied into `Sample.metadata`.
    """

    input: str = "input"
    target: str = "target"
    id: str | None = None
    metadata: list[str] = []


class Dataset(BaseModel):
    name: str
    samples: list[Sample]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def content_hash(self) -> str:
        return hash_samples(self.samples)


def hash_samples(samples: Iterable[Sample]) -> str:
    """Stable SHA-256 over the ordered sample contents."""
    h = hashlib.sha256()
    for s in samples:
        record = [s.id, s.input, s.target, s.metadata]
        h.update(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def rows_to_samples(rows: Iterable[Mapping[str, Any]], fields: FieldMap) -> list[Sample]:
    samples: list[Sample] = []
    for i, row in enumerate(rows):
        sample_id = str(row[fields.id]) if fields.id else str(i)
        samples.append(
            Sample(
                id=sample_id,
                input=str(row[fields.input]),
                target=str(row[fields.target]),
                metadata={k: row[k] for k in fields.metadata},
            )
        )
    return samples


def jsonl_dataset(path: str | Path, fields: FieldMap | None = None) -> Dataset:
    fields = fields or FieldMap()
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return Dataset(name=path.name, samples=rows_to_samples(rows, fields))


def csv_dataset(path: str | Path, fields: FieldMap | None = None) -> Dataset:
    fields = fields or FieldMap()
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return Dataset(name=path.name, samples=rows_to_samples(rows, fields))


def hf_dataset(
    path: str,
    split: str,
    fields: FieldMap | None = None,
    name: str | None = None,
) -> Dataset:
    """Load a Hugging Face dataset split. Requires the `hf` extra (`datasets`)."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on installed extras
        raise RuntimeError("Hugging Face datasets need the 'hf' extra: pip install 'owl-harness[hf]'") from e

    fields = fields or FieldMap()
    ds = load_dataset(path, name, split=split)
    return Dataset(name=f"{path}:{split}", samples=rows_to_samples(ds, fields))
