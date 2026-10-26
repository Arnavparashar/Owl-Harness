"""User-supplied datasets and the tests built from them.

A dataset is a file uploaded through the dashboard (CSV, TSV, JSONL, JSON or Parquet) or a
split imported from the Hugging Face Hub. Either way it is stored once under the data
directory as a fixed snapshot and parsed on demand; nothing in it is ever executed.

A `CustomTaskSpec` turns a dataset into a test: which columns to use, how to score
(yes/no, multi-class, exact match or judge-graded), and the prompt templates.
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
from owl.judge import JudgeScorer, default_rubric
from owl.scorers import binary_label, exact_match, label_match
from owl.solvers import chain, generate, templated_prompt
from owl.tasks import BINARY_ANSWER_INSTRUCTION, Task, list_tasks
from owl.types import GenerateConfig, Sample

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMPORT_BYTES = 200 * 1024 * 1024
MAX_ROWS = 100_000
MAX_LABELS = 100
_ID = re.compile(r"^[0-9a-f]{12}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

Format = Literal["csv", "tsv", "jsonl", "json", "parquet"]
FORMATS_HELP = ".csv, .tsv, .jsonl, .json or .parquet"


def _clean_name(value: str, limit: int) -> str:
    return _CONTROL.sub(" ", value).strip()[:limit]


# --- parsing --------------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _records_to_rows(records: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, str]]]:
    columns: list[str] = []
    for r in records:
        for k in r:
            if k not in columns:
                columns.append(str(k))
    return columns, [{c: _text(r.get(c)) for c in columns} for r in records]


def parse_table(filename: str, data: bytes, max_bytes: int | None = None) -> tuple[Format, list[str], list[dict[str, str]]]:
    """Parse a dataset file into (format, columns, rows). Raises ValueError with a clear message."""
    limit = MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    if len(data) > limit:
        raise ValueError(f"file is larger than {limit // (1024 * 1024)} MB")
    ext = Path(filename).suffix.lower()

    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ValueError("Parquet files need the 'parquet' extra: pip install 'owl-harness[parquet]'") from None
        try:
            table = pq.read_table(io.BytesIO(data))
        except Exception as e:  # pyarrow raises several types for a bad file
            raise ValueError(f"not a valid Parquet file: {e}") from None
        if table.num_rows > MAX_ROWS:
            raise ValueError(f"more than {MAX_ROWS:,} rows")
        columns, rows = _records_to_rows(table.to_pylist())
        columns = [str(c) for c in table.column_names]
        if not rows:
            raise ValueError("the file has no rows")
        return "parquet", columns, rows

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("file must be UTF-8 text") from None

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
        columns, rows = _records_to_rows(records)
    else:
        raise ValueError(f"unsupported file type; use {FORMATS_HELP}")

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
    # Where an imported dataset came from: {"hub": id, "config": ..., "split": ..., "revision": ...}
    source: dict[str, Any] | None = None


class DatasetLibrary:
    """Datasets on disk: `<root>/<id>/data.<ext>` plus `meta.json`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, dataset_id: str) -> Path:
        if not _ID.match(dataset_id):
            raise KeyError(f"no dataset '{dataset_id}'")
        return self.root / dataset_id

    def add(
        self,
        filename: str,
        data: bytes,
        name: str | None = None,
        source: dict[str, Any] | None = None,
        max_bytes: int | None = None,
    ) -> DatasetMeta:
        filename = _clean_name(Path(filename.replace("\\", "/")).name, 100) or "dataset"
        fmt, columns, rows = parse_table(filename, data, max_bytes)
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
            source=source,
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
        return parse_table(meta.filename, path.read_bytes(), max_bytes=MAX_IMPORT_BYTES)[2]

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


# --- Hugging Face import ------------------------------------------------------------


def _load_hf_rows(dataset: str, config: str | None, split: str, max_rows: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Download a split and return (rows, source facts). Class-label columns become label names."""
    try:
        from datasets import ClassLabel, load_dataset
    except ImportError:
        raise ValueError("importing from the Hub needs the 'hf' extra: pip install 'owl-harness[hf]'") from None
    try:
        ds = load_dataset(dataset, config, split=split)
    except Exception as e:  # unknown id, missing config, network: say what went wrong
        raise ValueError(f"could not load {dataset} ({config or 'default'}, {split}): {type(e).__name__}: {e}"[:400]) from None
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    if len(ds) > MAX_ROWS:
        raise ValueError(f"the split has {len(ds):,} rows; import at most {MAX_ROWS:,} by setting a row limit")
    label_columns = {name: feat for name, feat in ds.features.items() if isinstance(feat, ClassLabel)}
    rows = []
    for r in ds:
        r = dict(r)
        for name, feat in label_columns.items():
            if isinstance(r.get(name), int) and r[name] >= 0:
                r[name] = feat.int2str(r[name])
        rows.append(r)
    revision = None
    try:
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(dataset).sha
    except Exception:
        revision = None
    source = {"hub": dataset, "config": config, "split": split, "revision": revision}
    if max_rows is not None:
        source["max_rows"] = max_rows
    return rows, source


def import_hf_dataset(
    library: DatasetLibrary,
    dataset: str,
    split: str,
    config: str | None = None,
    max_rows: int | None = None,
    name: str | None = None,
) -> DatasetMeta:
    """Snapshot a Hub split into the library as JSONL, so runs on it are reproducible."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?", dataset):
        raise ValueError("dataset must be a Hub id such as 'fancyzhx/ag_news'")
    rows, source = _load_hf_rows(dataset, config or None, split, max_rows)
    if not rows:
        raise ValueError("the split has no rows")
    body = "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows).encode("utf-8")
    label = name or f"{dataset.split('/')[-1]}-{split}"
    return library.add(f"{_clean_name(label, 60)}.jsonl", body, name=label, source=source, max_bytes=MAX_IMPORT_BYTES)


# --- custom tests ---------------------------------------------------------------


class Normalization(BaseModel):
    """What exact match ignores. Stored on the run."""

    case: bool = True
    whitespace: bool = True
    punctuation: bool = False


class JudgeSpec(BaseModel):
    """The judge for a judge-graded test: a saved model (`model_id`) or an ad hoc one."""

    model_id: str | None = None
    provider: str | None = None
    model: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    rubric: Literal["passfail", "scale"] = "passfail"
    template: str | None = Field(default=None, max_length=8000)

    @model_validator(mode="after")
    def _model(self) -> "JudgeSpec":
        if not self.model_id and not (self.provider and self.model):
            raise ValueError("choose a judge model")
        return self


class CustomTaskSpec(BaseModel):
    """A test defined in the dashboard."""

    name: str = Field(min_length=1, max_length=60)
    dataset_id: str
    input_column: str
    # Optional only for judge-graded tests, where a reference answer is optional.
    target_column: str | None = None
    kind: Literal["binary", "multiclass", "exact_match", "judge"]
    # binary: the target value that means "yes", and the yes/no question
    positive_label: str | None = None
    question: str | None = Field(default=None, max_length=4000)
    # multiclass: the label set (defaults to the target column's distinct values)
    labels: list[str] | None = None
    # Templates. `{column}` fills a dataset column, `{input}` the input column, `{labels}` the label list.
    # Binary: system_prompt overrides the default prompt built from the question.
    system_prompt: str | None = Field(default=None, max_length=8000)
    user_template: str | None = Field(default=None, max_length=8000)
    # exact match
    case_sensitive: bool = False  # kept for tests saved before `normalization` existed
    normalization: Normalization | None = None
    # judge
    judge: JudgeSpec | None = None
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
    def _fields_for_kind(self) -> "CustomTaskSpec":
        if self.kind != "judge" and not self.target_column:
            raise ValueError("choose the target column (the correct answer)")
        if self.kind == "binary":
            if not (self.question or "").strip():
                raise ValueError("a yes/no test needs a question")
            if self.positive_label is None or not self.positive_label.strip():
                raise ValueError("a yes/no test needs the target value that means yes")
        if self.kind == "judge" and self.judge is None:
            raise ValueError("a judge-graded test needs a judge model")
        if self.labels is not None:
            cleaned = [l.strip() for l in self.labels if l.strip()]
            if len(cleaned) < 2:
                raise ValueError("a multi-class test needs at least two labels")
            if len({l.casefold() for l in cleaned}) != len(cleaned):
                raise ValueError("labels must be unique (ignoring case)")
            if len(cleaned) > MAX_LABELS:
                raise ValueError(f"at most {MAX_LABELS} labels")
            self.labels = cleaned
        return self


DEFAULT_USER_TEMPLATE = "{input}"
MULTICLASS_SYSTEM_PROMPT = "Classify the text into exactly one of these labels: {labels}.\nAnswer with only the label, nothing else."


def default_max_tokens(kind: str) -> int:
    return {"binary": 16, "multiclass": 32, "exact_match": 256, "judge": 512}.get(kind, 256)


def resolved_judge_template(spec: CustomTaskSpec) -> str:
    assert spec.judge is not None
    return spec.judge.template or default_rubric(spec.judge.rubric, has_reference=bool(spec.target_column))


def task_from_rows(
    spec: CustomTaskSpec,
    rows: list[dict[str, str]],
    columns: list[str],
    dataset_name: str,
    timeout_s: float = 120.0,
    example: bool = False,
) -> Task:
    """Turn a spec and the dataset's rows into a runnable `Task`."""
    for col in (spec.input_column, spec.target_column):
        if col is not None and col not in columns:
            raise ValueError(f"dataset has no column '{col}'")
    if spec.target_column is not None and spec.input_column == spec.target_column:
        raise ValueError("the input and target columns must be different")

    labels: list[str] | None = None
    if spec.kind == "multiclass":
        assert spec.target_column is not None
        counts = Counter(r[spec.target_column].strip() for r in rows)
        labels = spec.labels or sorted(v for v in counts if v)
        if len(labels) < 2:
            raise ValueError(f"column '{spec.target_column}' has fewer than two distinct values")
        if len(labels) > MAX_LABELS:
            raise ValueError(f"column '{spec.target_column}' has {len(labels)} distinct values; at most {MAX_LABELS} labels are supported")

    samples: list[Sample] = []
    positives = 0
    for i, row in enumerate(rows):
        raw = row[spec.target_column].strip() if spec.target_column else ""
        if spec.kind == "binary":
            is_pos = raw == (spec.positive_label or "").strip()
            positives += is_pos
            target = "1" if is_pos else "0"
        else:
            target = raw
        samples.append(Sample(id=str(i), input=row[spec.input_column], target=target, metadata={"label": raw, "row": row}))
    if spec.kind == "binary" and positives == 0:
        raise ValueError(f"no row has the value '{spec.positive_label}' in column '{spec.target_column}'")

    config = GenerateConfig(temperature=spec.temperature, max_tokens=spec.max_tokens or default_max_tokens(spec.kind))
    dataset = Dataset(name=dataset_name, samples=samples)
    user = (spec.user_template or "").strip() or DEFAULT_USER_TEMPLATE
    common: dict[str, Any] = {
        "name": spec.name,
        "load_dataset": lambda: dataset,
        "generate": config,
        "description": f"Custom test on {dataset_name} ({len(samples)} samples)",
        "spec": spec.model_dump(),
        "example": example,
    }

    if spec.kind == "binary":
        question = (spec.question or "").strip()
        system = (spec.system_prompt or "").strip() or f"{question}\n{BINARY_ANSWER_INSTRUCTION}"
        return Task(
            solver=chain(templated_prompt(system, user), generate()),
            scorer=binary_label(positive="1"),
            positive_class="1",
            scoring_mode="binary",
            question=question,
            system_prompt=system,
            prompt={"system": system, "user": user},
            scorer_config={"positive_label": spec.positive_label, "answer_format": "1 = yes, 0 = no"},
            **common,
        )

    if spec.kind == "multiclass":
        assert labels is not None
        system = (spec.system_prompt or "").strip() or MULTICLASS_SYSTEM_PROMPT
        extra = {"labels": ", ".join(labels)}
        return Task(
            solver=chain(templated_prompt(system, user, extra), generate()),
            scorer=label_match(labels),
            scoring_mode="multiclass",
            labels=labels,
            system_prompt=system,
            prompt={"system": system, "user": user, "variables": extra},
            scorer_config={"labels": labels, "matching": "case-insensitive; exactly one label must be named"},
            **common,
        )

    if spec.kind == "exact_match":
        norm = spec.normalization or Normalization(case=not spec.case_sensitive)
        system = (spec.system_prompt or "").strip()
        return Task(
            solver=chain(templated_prompt(system or None, user), generate()),
            scorer=exact_match(ignore_case=norm.case, ignore_whitespace=norm.whitespace, ignore_punctuation=norm.punctuation),
            scoring_mode="exact_match",
            system_prompt=system or None,
            prompt={"system": system or None, "user": user},
            scorer_config={"normalization": norm.model_dump()},
            **common,
        )

    # judge
    assert spec.judge is not None and spec.judge.provider and spec.judge.model, "resolve the judge before building"
    judge = JudgeScorer(
        spec.judge.provider,
        spec.judge.model,
        spec.judge.args,
        spec.judge.rubric,
        resolved_judge_template(spec),
        timeout_s=timeout_s,
    )
    system = (spec.system_prompt or "").strip()
    return Task(
        solver=chain(templated_prompt(system or None, user), generate()),
        scorer=judge,
        scoring_mode="judge_passfail" if spec.judge.rubric == "passfail" else "judge_scale",
        system_prompt=system or None,
        prompt={"system": system or None, "user": user},
        scorer_config=judge.describe(),
        resources=[judge],
        **common,
    )


def build_task(spec: CustomTaskSpec, library: DatasetLibrary, timeout_s: float = 120.0) -> Task:
    """A runnable `Task` for a test defined on a library dataset."""
    if spec.name in list_tasks():
        raise ValueError(f"'{spec.name}' is the name of a built-in test; choose another name")
    meta = library.get(spec.dataset_id)
    return task_from_rows(spec, library.rows(spec.dataset_id), meta.columns, meta.name, timeout_s=timeout_s)
