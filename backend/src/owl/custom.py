"""User-supplied datasets and the custom tests built from them.

A dataset is a CSV, TSV, JSONL or JSON file uploaded through the dashboard. It is stored
as-is under the data directory and parsed on demand; nothing in it is ever executed.
A `CustomTaskSpec` says which columns to use, how to score, and what to ask the model.
"""

from __future__ import annotations

import csv
import io
import json
import re
import secrets
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from owl.datasets import Dataset
from owl.scorers import binary_label, exact_match
from owl.solvers import chain, generate, system_message
from owl.tasks import BINARY_ANSWER_INSTRUCTION, Task, list_tasks
from owl.types import GenerateConfig, Sample

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
_ID = re.compile(r"^[0-9a-f]{12}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

Format = Literal["csv", "tsv", "jsonl", "json"]


def _clean_name(value: str, limit: int) -> str:
    return _CONTROL.sub(" ", value).strip()[:limit]


# --- parsing --------------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_table(filename: str, data: bytes) -> tuple[Format, list[str], list[dict[str, str]]]:
    """Parse an uploaded file into (format, columns, rows). Raises ValueError with a clear message."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("file must be UTF-8 text") from None

    ext = Path(filename).suffix.lower()
    columns: list[str] = []
    rows: list[dict[str, str]] = []
    if ext in (".csv", ".tsv"):
        fmt: Format = "csv" if ext == ".csv" else "tsv"
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="," if fmt == "csv" else "\t")
        columns = [c for c in (reader.fieldnames or [])]
        if not columns or any(not c.strip() for c in columns):
            raise ValueError("the first row must be a header with a name for every column")
        if len(set(columns)) != len(columns):
            raise ValueError("column names must be unique")
        for r in reader:
            # A row with more or fewer fields than the header almost always means an unquoted comma.
            if None in r or any(v is None for v in r.values()):
                raise ValueError(
                    f"line {reader.line_num} has a different number of columns than the header. "
                    "Put text that contains commas in double quotes."
                )
            rows.append({c: _text(r[c]) for c in columns})
            if len(rows) > MAX_ROWS:
                raise ValueError(f"more than {MAX_ROWS:,} rows")
    elif ext in (".jsonl", ".ndjson", ".json"):
        fmt = "json" if ext == ".json" else "jsonl"
        try:
            records = json.loads(text) if fmt == "json" else [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid {fmt.upper()}: {e.msg} (line {e.lineno})") from None
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValueError("expected a list of objects, one per sample")
        if len(records) > MAX_ROWS:
            raise ValueError(f"more than {MAX_ROWS:,} rows")
        for r in records:
            for k in r:
                if k not in columns:
                    columns.append(k)
        rows = [{c: _text(r.get(c)) for c in columns} for r in records]
    else:
        raise ValueError("unsupported file type; use .csv, .tsv, .jsonl or .json")

    if not rows:
        raise ValueError("the file has no rows")
    return fmt, columns, rows


# --- library --------------------------------------------------------------------


class DatasetMeta(BaseModel):
    id: str
    name: str
    filename: str
    format: Format
    rows: int
    columns: list[str]
    size_bytes: int
    created_at: str


class DatasetLibrary:
    """Uploaded datasets on disk: `<root>/<id>/data` plus `meta.json`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, dataset_id: str) -> Path:
        if not _ID.match(dataset_id):
            raise KeyError(f"no dataset '{dataset_id}'")
        return self.root / dataset_id

    def add(self, filename: str, data: bytes, name: str | None = None) -> DatasetMeta:
        filename = _clean_name(Path(filename.replace("\\", "/")).name, 100) or "dataset"
        fmt, columns, rows = parse_table(filename, data)
        dataset_id = secrets.token_hex(6)
        meta = DatasetMeta(
            id=dataset_id,
            name=_clean_name(name or Path(filename).stem, 60) or "dataset",
            filename=filename,
            format=fmt,
            rows=len(rows),
            columns=columns,
            size_bytes=len(data),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        d = self._dir(dataset_id)
        d.mkdir(parents=True, exist_ok=False)
        (d / f"data.{Path(filename).suffix.lower().lstrip('.')}").write_bytes(data)
        (d / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        return meta

    def get(self, dataset_id: str) -> DatasetMeta:
        path = self._dir(dataset_id) / "meta.json"
        if not path.is_file():
            raise KeyError(f"no dataset '{dataset_id}'")
        return DatasetMeta.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[DatasetMeta]:
        if not self.root.is_dir():
            return []
        out = []
        for d in self.root.iterdir():
            if _ID.match(d.name) and (d / "meta.json").is_file():
                out.append(self.get(d.name))
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    def delete(self, dataset_id: str) -> None:
        d = self._dir(dataset_id)
        if not (d / "meta.json").is_file():
            raise KeyError(f"no dataset '{dataset_id}'")
        shutil.rmtree(d)

    def rows(self, dataset_id: str) -> list[dict[str, str]]:
        meta = self.get(dataset_id)
        path = next(self._dir(dataset_id).glob("data.*"))
        return parse_table(meta.filename, path.read_bytes())[2]

    def preview(self, dataset_id: str, n: int = 5) -> list[dict[str, str]]:
        return self.rows(dataset_id)[:n]

    def column_values(self, dataset_id: str, column: str, limit: int = 50) -> dict[str, Any]:
        meta = self.get(dataset_id)
        if column not in meta.columns:
            raise KeyError(f"dataset has no column '{column}'")
        counts = Counter(r[column].strip() for r in self.rows(dataset_id))
        top = counts.most_common(limit)
        return {
            "distinct": len(counts),
            "values": [{"value": v, "count": c} for v, c in top],
        }


# --- custom tasks ---------------------------------------------------------------


class CustomTaskSpec(BaseModel):
    """A test defined in the dashboard: a dataset, two columns, a scoring mode and a prompt."""

    name: str = Field(min_length=1, max_length=60)
    dataset_id: str
    input_column: str
    target_column: str
    kind: Literal["binary", "exact_match"]
    # binary: the target value that means "yes", and the yes/no question asked of the model
    positive_label: str | None = None
    question: str | None = Field(default=None, max_length=4000)
    # exact_match: optional instructions; binary: optional override of the whole system prompt
    system_prompt: str | None = Field(default=None, max_length=8000)
    case_sensitive: bool = False
    temperature: float | None = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = _clean_name(v, 60)
        if not v:
            raise ValueError("name cannot be empty")
        return v

    @model_validator(mode="after")
    def _binary_fields(self) -> "CustomTaskSpec":
        if self.kind == "binary":
            if not (self.question or "").strip():
                raise ValueError("a yes/no test needs a question")
            if self.positive_label is None or not self.positive_label.strip():
                raise ValueError("a yes/no test needs the target value that means yes")
        return self


def default_max_tokens(kind: str) -> int:
    return 16 if kind == "binary" else 256


def build_task(spec: CustomTaskSpec, library: DatasetLibrary) -> Task:
    """Turn a spec into a runnable `Task`. Raises ValueError/KeyError for anything wrong."""
    if spec.name in list_tasks():
        raise ValueError(f"'{spec.name}' is the name of a built-in test; choose another name")
    meta = library.get(spec.dataset_id)
    for col in (spec.input_column, spec.target_column):
        if col not in meta.columns:
            raise ValueError(f"dataset has no column '{col}'")
    if spec.input_column == spec.target_column:
        raise ValueError("the input and target columns must be different")

    rows = library.rows(spec.dataset_id)
    samples: list[Sample] = []
    positives = 0
    for i, row in enumerate(rows):
        raw = row[spec.target_column].strip()
        if spec.kind == "binary":
            is_pos = raw == (spec.positive_label or "").strip()
            positives += is_pos
            target = "1" if is_pos else "0"
        else:
            target = raw
        samples.append(Sample(id=str(i), input=row[spec.input_column], target=target, metadata={"label": raw}))
    if spec.kind == "binary" and positives == 0:
        raise ValueError(f"no row has the value '{spec.positive_label}' in column '{spec.target_column}'")

    config = GenerateConfig(temperature=spec.temperature, max_tokens=spec.max_tokens or default_max_tokens(spec.kind))
    dataset = Dataset(name=meta.name, samples=samples)
    description = f"Custom test on {meta.name} ({len(samples)} samples)"

    if spec.kind == "binary":
        question = (spec.question or "").strip()
        prompt = (spec.system_prompt or "").strip() or f"{question}\n{BINARY_ANSWER_INSTRUCTION}"
        return Task(
            name=spec.name,
            load_dataset=lambda: dataset,
            solver=chain(system_message(prompt), generate()),
            scorer=binary_label(positive="1"),
            generate=config,
            positive_class="1",
            # Laya ignores the system message; it gets the question as its yes/no instruction.
            provider_args={"laya": {"instruction": question}},
            system_prompt=prompt,
            description=description,
            spec=spec.model_dump(),
        )

    prompt = (spec.system_prompt or "").strip()
    return Task(
        name=spec.name,
        load_dataset=lambda: dataset,
        solver=chain(*([system_message(prompt)] if prompt else []), generate()),
        scorer=exact_match(case_sensitive=spec.case_sensitive),
        generate=config,
        system_prompt=prompt or None,
        description=description,
        spec=spec.model_dump(),
    )
