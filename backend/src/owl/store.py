"""SQLite persistence for runs, per-sample results and metrics."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from owl import __version__
from owl.metrics import Row
from owl.types import RunConfig

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    config_json TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    owl_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    sample_id TEXT NOT NULL,
    input TEXT NOT NULL,
    target TEXT NOT NULL,
    output TEXT,
    pred TEXT,
    correct INTEGER NOT NULL DEFAULT 0,
    probability REAL,
    latency_ms REAL,
    stop_reason TEXT,
    error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, sample_id)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);
"""

SAMPLE_FILTERS = ("all", "correct", "incorrect", "false_positive", "false_negative", "unparseable")


def default_db_path() -> Path:
    """`OWL_DB` if set, else backend/data/owl.db next to the source tree."""
    env = os.environ.get("OWL_DB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "owl.db"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


class SampleRecord(BaseModel):
    """One finished sample, as written to the store."""

    sample_id: str
    input: str
    target: str
    output: str | None = None
    pred: str | None = None
    correct: bool = False
    probability: float | None = None
    latency_ms: float | None = None
    stop_reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = {}


class Store:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _all(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # --- runs ---------------------------------------------------------------

    def create_run(self, config: RunConfig, provider: str, dataset_hash: str, run_id: str | None = None) -> str:
        run_id = run_id or new_run_id()
        with self._tx() as c:
            c.execute(
                "INSERT INTO runs (id, task, model, provider, config_json, dataset_hash, status,"
                " started_at, finished_at, owl_version) VALUES (?,?,?,?,?,?,?,?,NULL,?)",
                (
                    run_id,
                    config.task,
                    config.model,
                    provider,
                    config.model_dump_json(),
                    dataset_hash,
                    "running",
                    utcnow(),
                    __version__,
                ),
            )
        return run_id

    def set_status(self, run_id: str, status: str, finished: bool = False) -> None:
        with self._tx() as c:
            if finished:
                c.execute("UPDATE runs SET status=?, finished_at=? WHERE id=?", (status, utcnow(), run_id))
            else:
                c.execute("UPDATE runs SET status=?, finished_at=NULL WHERE id=?", (status, run_id))

    @staticmethod
    def _run_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        return d

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM runs WHERE id=?", (run_id,))
        return self._run_dict(rows[0]) if rows else None

    def delete_run(self, run_id: str) -> bool:
        """Remove a run with its samples and metrics. False if there was no such run."""
        with self._tx() as c:
            return c.execute("DELETE FROM runs WHERE id=?", (run_id,)).rowcount > 0

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM runs ORDER BY started_at DESC, id DESC")
        return [self._run_dict(r) for r in rows]

    def delete_run(self, run_id: str) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM runs WHERE id=?", (run_id,))

    # --- samples ------------------------------------------------------------

    def add_sample(self, run_id: str, rec: SampleRecord) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO samples (run_id, sample_id, input, target, output, pred, correct,"
                " probability, latency_ms, stop_reason, error, metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    rec.sample_id,
                    rec.input,
                    rec.target,
                    rec.output,
                    rec.pred,
                    int(rec.correct),
                    rec.probability,
                    rec.latency_ms,
                    rec.stop_reason,
                    rec.error,
                    json.dumps(rec.metadata, ensure_ascii=False, default=str),
                ),
            )

    def completed_sample_ids(self, run_id: str) -> set[str]:
        """Samples with a result (errored samples are retried on resume)."""
        rows = self._all("SELECT sample_id FROM samples WHERE run_id=? AND error IS NULL", (run_id,))
        return {r["sample_id"] for r in rows}

    def count_samples(self, run_id: str) -> dict[str, int]:
        rows = self._all(
            "SELECT COUNT(*) AS total, SUM(error IS NOT NULL) AS errors FROM samples WHERE run_id=?", (run_id,)
        )
        return {"total": rows[0]["total"] or 0, "errors": rows[0]["errors"] or 0}

    def rows_for_metrics(self, run_id: str) -> list[Row]:
        """Scored (non-errored) samples in sample order."""
        rows = self._all(
            "SELECT target, pred, correct, probability, latency_ms FROM samples"
            f" WHERE run_id=? AND error IS NULL ORDER BY {_ORDER}",
            (run_id,),
        )
        return [
            Row(
                target=r["target"],
                pred=r["pred"],
                correct=bool(r["correct"]),
                probability=r["probability"],
                latency_ms=r["latency_ms"],
            )
            for r in rows
        ]

    def query_samples(
        self,
        run_id: str,
        positive_class: str | None,
        filter: str = "all",
        q: str | None = None,
        page: int = 1,
        page_size: int = 25,
    ) -> tuple[int, list[dict[str, Any]]]:
        if filter not in SAMPLE_FILTERS:
            raise ValueError(f"filter must be one of {SAMPLE_FILTERS}")
        where = ["run_id = ?"]
        params: list[Any] = [run_id]
        if filter == "correct":
            where.append("error IS NULL AND correct = 1")
        elif filter == "incorrect":
            where.append("error IS NULL AND correct = 0")
        elif filter == "unparseable":
            where.append("error IS NULL AND pred IS NULL")
        elif filter in ("false_positive", "false_negative"):
            if positive_class is None:
                where.append("0")
            elif filter == "false_positive":
                where.append("error IS NULL AND pred = ? AND target != ?")
                params += [positive_class, positive_class]
            else:
                where.append("error IS NULL AND pred IS NOT NULL AND pred != ? AND target = ?")
                params += [positive_class, positive_class]
        if q:
            where.append("(instr(lower(input), lower(?)) > 0 OR instr(lower(coalesce(output, '')), lower(?)) > 0)")
            params += [q, q]
        clause = " AND ".join(where)
        total = self._all(f"SELECT COUNT(*) AS c FROM samples WHERE {clause}", params)[0]["c"]
        rows = self._all(
            f"SELECT * FROM samples WHERE {clause} ORDER BY {_ORDER} LIMIT ? OFFSET ?",
            [*params, page_size, (page - 1) * page_size],
        )
        return total, [self._sample_dict(r) for r in rows]

    def all_samples(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._all(f"SELECT * FROM samples WHERE run_id=? ORDER BY {_ORDER}", (run_id,))
        return [self._sample_dict(r) for r in rows]

    @staticmethod
    def _sample_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["correct"] = bool(d["correct"])
        d["metadata"] = json.loads(d.pop("metadata_json"))
        return d

    # --- metrics ------------------------------------------------------------

    def set_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
            c.executemany(
                "INSERT INTO metrics (run_id, name, value) VALUES (?,?,?)",
                [(run_id, k, float(v)) for k, v in metrics.items()],
            )

    def get_metrics(self, run_id: str) -> dict[str, float]:
        return {r["name"]: r["value"] for r in self._all("SELECT name, value FROM metrics WHERE run_id=?", (run_id,))}


# Numeric sample ids sort numerically; others fall back to text order.
_ORDER = "CASE WHEN sample_id GLOB '[0-9]*' THEN CAST(sample_id AS INTEGER) END, sample_id"
