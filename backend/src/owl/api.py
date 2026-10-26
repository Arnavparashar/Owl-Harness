"""FastAPI application: JSON API under /api plus the built dashboard at /."""

from __future__ import annotations

import asyncio
import csv
import io
import os
import statistics
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from owl import __version__
from owl.custom import (
    FORMATS_HELP,
    MAX_UPLOAD_BYTES,
    MULTICLASS_SYSTEM_PROMPT,
    DatasetLibrary,
    default_max_tokens,
    import_hf_dataset,
)
from owl.jobs import JobManager, RunRequest
from owl.judge import default_rubric
from owl.metrics import (
    accuracy_by_length,
    calibration,
    latency_histogram,
    mcnemar_exact,
    multiclass_report,
    pr_curve,
    roc_curve,
    score_distribution,
    threshold_curve,
)
from owl.providers.base import (
    ArgSpec,
    available_providers,
    describe_provider,
    get_provider_class,
    validate_model_args,
)
from owl.reporting import fill_secondary_interval, headline, labels_of, scoring_mode
from owl.runner import DEFAULT_SEED, DEFAULT_TEMPERATURE
from owl.store import SAMPLE_FILTERS, Store, default_db_path
from owl.tasks import BINARY_ANSWER_INSTRUCTION, describe_tasks, list_tasks
from owl.types import GenerateConfig, Message

# Inputs used when a provider that needs task values (a yes/no decision model, say) is
# tested on its own from the Models page.
PROBE_PROMPT = "Reply with the single word: OK"
PROBE_TASK_INPUTS = {"question": "Is this text a greeting?", "positive_class": "1", "labels": ["greeting", "other"]}


def default_frontend_dist() -> Path:
    env = os.environ.get("OWL_FRONTEND_DIST")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_text(value: object) -> object:
    """Neutralise spreadsheet formulas: dataset text is untrusted and may start with `=` etc."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


class SavedModelIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=400)
    settings: dict[str, Any] = Field(default_factory=dict)


class HFImportIn(BaseModel):
    dataset: str = Field(min_length=1, max_length=200)
    split: str = Field(min_length=1, max_length=60)
    config: str | None = Field(default=None, max_length=120)
    max_rows: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, max_length=60)


async def probe_model(provider: str, model: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Load a model and send it one short prompt. Reports success, timings and the raw output."""
    instance = None
    try:
        cls = get_provider_class(provider)
        args = validate_model_args(cls, settings)
        for name in cls.task_inputs:
            args.setdefault(name, PROBE_TASK_INPUTS.get(name))
        instance = cls(model, **args)
        t0 = time.perf_counter()
        await asyncio.wait_for(instance.setup(), timeout=600)
        setup_ms = (time.perf_counter() - t0) * 1000
        out = await asyncio.wait_for(
            instance.generate([Message(role="user", content=PROBE_PROMPT)], GenerateConfig(temperature=0.0, max_tokens=16, seed=DEFAULT_SEED)),
            timeout=120,
        )
        return {
            "ok": True,
            "latency_ms": out.latency_ms,
            "setup_ms": setup_ms,
            "output": out.text,
            "probability": out.probability,
            "model_info": instance.model_info(),
            "error": None,
        }
    except Exception as e:
        message = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        return {"ok": False, "latency_ms": None, "setup_ms": None, "output": None, "probability": None, "model_info": {}, "error": message[:500]}
    finally:
        if instance is not None:
            try:
                await instance.close()
            except Exception:
                pass


def create_app(
    store: Store | None = None,
    frontend_dist: Path | None = None,
    library: DatasetLibrary | None = None,
    jobs: JobManager | None = None,
    allowed_hosts: set[str] | None = None,
) -> FastAPI:
    """Build the app.

    `allowed_hosts`, when given, is the set of Host header names the server answers to. The
    CLI sets it for a loopback bind so a web page on another site cannot reach the API through
    DNS rebinding. State-changing requests that carry a different Origin are always refused.
    """
    store = store or Store()
    library = library or DatasetLibrary(default_db_path().parent / "datasets")
    jobs = jobs or JobManager(store, library)
    dist = frontend_dist if frontend_dist is not None else default_frontend_dist()
    app = FastAPI(title="Owl Harness", version=__version__)

    @app.middleware("http")
    async def local_only(request: Request, call_next):  # type: ignore[no-untyped-def]
        host = request.headers.get("host", "")
        if allowed_hosts is not None and (urlsplit(f"//{host}").hostname or "").lower() not in allowed_hosts:
            return JSONResponse({"detail": "unexpected Host header"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin is not None and urlsplit(origin).netloc.lower() != host.lower():
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    def example_tasks() -> set[str]:
        return {t["name"] for t in describe_tasks() if t["example"]}

    def summary(run: dict[str, Any], examples: set[str] | None = None) -> dict[str, Any]:
        m = store.get_metrics(run["id"])
        head = headline(run, m)
        main = head["main"]
        ci = [main["ci_low"], main["ci_high"]] if main and main["key"] != "mean_score" else None
        return {
            "id": run["id"],
            "task": run["task"],
            "model": run["model"],
            "provider": run["provider"],
            "status": run["status"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "dataset_hash": run["dataset_hash"],
            "owl_version": run["owl_version"],
            "positive_class": run["config"].get("positive_class"),
            "scoring_mode": head["mode"],
            "example": run["task"] in (examples if examples is not None else example_tasks()),
            "metrics": m,
            "accuracy_ci": ci,  # kept for clients written against v0.1
            "headline": head,
        }

    def get_run_or_404(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"run '{run_id}' not found")
        return run

    # --- app: what can be run -------------------------------------------------

    @app.get("/api/config")
    def app_config() -> dict[str, Any]:
        return {
            "version": __version__,
            "binary_answer_instruction": BINARY_ANSWER_INSTRUCTION,
            "multiclass_system_prompt": MULTICLASS_SYSTEM_PROMPT,
            "rubrics": {
                "passfail": {"with_reference": default_rubric("passfail", True), "without_reference": default_rubric("passfail", False)},
                "scale": {"with_reference": default_rubric("scale", True), "without_reference": default_rubric("scale", False)},
            },
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "formats": FORMATS_HELP,
            "default_temperature": DEFAULT_TEMPERATURE,
            "default_seed": DEFAULT_SEED,
            "default_max_tokens": {k: default_max_tokens(k) for k in ("binary", "multiclass", "exact_match", "judge")},
        }

    @app.get("/api/providers")
    def providers() -> list[dict[str, Any]]:
        out = []
        for name in sorted(available_providers()):
            try:
                out.append(describe_provider(name, get_provider_class(name)))
            except Exception as e:  # a broken plugin should not hide the others
                out.append(
                    {"name": name, "display_name": name, "error": f"{type(e).__name__}: {e}", "args": [], "description": "", "binary_only": False, "max_concurrency": 1}
                )
        return out

    @app.get("/api/providers/{name}/models")
    async def provider_models(name: str, request: Request) -> dict[str, Any]:
        try:
            cls = get_provider_class(name)
        except KeyError as e:
            raise HTTPException(404, e.args[0]) from None
        # only settings that change which models exist (such as the server URL) apply here
        relevant: dict[str, ArgSpec] = {a.name: a for a in cls.arg_specs if a.affects_models}
        try:
            args = validate_model_args(cls, {k: v for k, v in request.query_params.items() if k in relevant})
            models = await asyncio.wait_for(cls.list_models(**args), timeout=10)
        except Exception as e:  # server down, wrong URL, bad key: report it, the UI lets you type a name
            return {"models": [], "error": f"{type(e).__name__}: {e}"[:300]}
        return {"models": models, "error": None}

    @app.get("/api/tasks")
    def tasks() -> list[dict[str, Any]]:
        return describe_tasks()

    # --- app: model registry ----------------------------------------------------

    def validated_model(body: SavedModelIn) -> tuple[str, str, str, dict[str, Any]]:
        try:
            cls = get_provider_class(body.provider)
            settings = validate_model_args(cls, body.settings)
        except KeyError as e:
            raise HTTPException(422, e.args[0]) from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return body.name.strip(), body.provider, body.model.strip(), settings

    @app.get("/api/models")
    def list_models() -> list[dict[str, Any]]:
        return store.list_models()

    @app.post("/api/models", status_code=201)
    def create_model(body: SavedModelIn) -> dict[str, Any]:
        name, provider, model, settings = validated_model(body)
        return store.save_model(name, provider, model, settings)

    @app.put("/api/models/{model_id}")
    def update_model(model_id: str, body: SavedModelIn) -> dict[str, Any]:
        if store.get_model(model_id) is None:
            raise HTTPException(404, f"no saved model '{model_id}'")
        name, provider, model, settings = validated_model(body)
        return store.save_model(name, provider, model, settings, model_id=model_id)

    @app.delete("/api/models/{model_id}", status_code=204)
    def delete_model(model_id: str) -> Response:
        if not store.delete_model(model_id):
            raise HTTPException(404, f"no saved model '{model_id}'")
        return Response(status_code=204)

    @app.post("/api/models/{model_id}/test")
    async def test_saved_model(model_id: str) -> dict[str, Any]:
        saved = store.get_model(model_id)
        if saved is None:
            raise HTTPException(404, f"no saved model '{model_id}'")
        return await probe_model(saved["provider"], saved["model"], saved["settings"])

    @app.post("/api/models/test")
    async def test_unsaved_model(body: SavedModelIn) -> dict[str, Any]:
        return await probe_model(body.provider, body.model, body.settings)

    @app.get("/api/models/suggestions")
    async def model_suggestions() -> dict[str, Any]:
        """Models installed in the local Ollama that are not saved yet: one click to add."""
        try:
            installed = await asyncio.wait_for(get_provider_class("ollama").list_models(), timeout=10)
        except Exception as e:
            return {"items": [], "error": f"{type(e).__name__}: {e}"[:300]}
        saved = {(m["provider"], m["model"]) for m in store.list_models()}
        items = [{"name": m, "provider": "ollama", "model": m, "settings": {}} for m in installed if ("ollama", m) not in saved]
        return {"items": items, "error": None}

    # --- app: datasets ----------------------------------------------------------

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return [m.model_dump() for m in library.list()]

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(
        request: Request,
        filename: str = Query(..., min_length=1, max_length=200),
        name: str | None = Query(None, max_length=60),
    ) -> dict[str, Any]:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            chunks.append(chunk)
        try:
            meta = library.add(filename, b"".join(chunks), name)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {**meta.model_dump(), "preview": library.preview(meta.id)}

    @app.post("/api/datasets/import-hf", status_code=201)
    def import_from_hub(body: HFImportIn) -> dict[str, Any]:
        try:
            meta = import_hf_dataset(library, body.dataset, body.split, body.config, body.max_rows, body.name)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {**meta.model_dump(), "preview": library.preview(meta.id)}

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, rows: int = Query(5, ge=1, le=50)) -> dict[str, Any]:
        try:
            meta = library.get(dataset_id)
        except KeyError as e:
            raise HTTPException(404, e.args[0]) from None
        return {**meta.model_dump(), "preview": library.preview(dataset_id, rows)}

    @app.get("/api/datasets/{dataset_id}/values")
    def dataset_values(dataset_id: str, column: str) -> dict[str, Any]:
        try:
            return library.column_values(dataset_id, column)
        except KeyError as e:
            raise HTTPException(404, e.args[0]) from None

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    def delete_dataset(dataset_id: str) -> Response:
        try:
            library.delete(dataset_id)
        except KeyError as e:
            raise HTTPException(404, e.args[0]) from None
        return Response(status_code=204)

    # --- app: jobs --------------------------------------------------------------

    @app.post("/api/jobs", status_code=201)
    def create_job(req: RunRequest) -> dict[str, Any]:
        try:
            return jobs.submit(req).to_dict(store)
        except (ValueError, KeyError) as e:
            raise HTTPException(422, e.args[0] if isinstance(e, KeyError) else str(e)) from None

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [j.to_dict(store) for j in jobs.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id).to_dict(store)
        except KeyError as e:
            raise HTTPException(404, e.args[0]) from None

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id).to_dict(store)
        except KeyError as e:
            raise HTTPException(404, e.args[0]) from None

    # --- runs ----------------------------------------------------------------

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        examples = example_tasks()
        return [summary(r, examples) for r in store.list_runs()]

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str) -> Response:
        get_run_or_404(run_id)
        if run_id in jobs.active_run_ids():
            raise HTTPException(409, "this run is still in progress; cancel it first")
        store.delete_run(run_id)
        return Response(status_code=204)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        run = get_run_or_404(run_id)
        rows = store.rows_for_metrics(run_id)
        latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
        out = {
            **summary(run),
            "config": run["config"],
            "labels": labels_of(run),
            "sample_counts": store.count_samples(run_id),
            "has_probabilities": any(r.probability is not None for r in rows),
            "latency_histogram": latency_histogram(latencies),
        }
        fill_secondary_interval(run, out["headline"], rows)
        return out

    @app.get("/api/runs/{run_id}/samples")
    def run_samples(
        run_id: str,
        filter: str = "all",
        q: str | None = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=200),
    ) -> dict[str, Any]:
        run = get_run_or_404(run_id)
        if filter not in SAMPLE_FILTERS:
            raise HTTPException(422, f"filter must be one of {', '.join(SAMPLE_FILTERS)}")
        total, items = store.query_samples(
            run_id, run["config"].get("positive_class"), filter=filter, q=q, page=page, page_size=page_size
        )
        return {"total": total, "page": page, "page_size": page_size, "items": items}

    @app.get("/api/runs/{run_id}/threshold-curve")
    def run_threshold_curve(run_id: str) -> dict[str, Any]:
        run = get_run_or_404(run_id)
        positive = run["config"].get("positive_class")
        rows = [r for r in store.rows_for_metrics(run_id) if r.probability is not None]
        if positive is None or not rows:
            raise HTTPException(404, "this run has no sample probabilities")
        return {
            "positive_class": positive,
            "n": len(rows),
            "run_threshold": run["config"].get("model_args", {}).get("threshold", 0.5),
            "points": threshold_curve(rows, positive),
        }

    @app.get("/api/runs/{run_id}/analysis")
    def run_analysis(run_id: str) -> dict[str, Any]:
        """Everything the dashboard's analysis charts need, computed from stored samples."""
        run = get_run_or_404(run_id)
        mode = scoring_mode(run)
        positive = run["config"].get("positive_class")
        rows = store.rows_for_metrics(run_id)
        scored = [s for s in store.all_samples(run_id) if s["error"] is None]
        has_probs = positive is not None and any(r.probability is not None for r in rows)

        def latency_stats(subset: list[float]) -> dict[str, float] | None:
            return {"n": len(subset), "median_ms": statistics.median(subset)} if subset else None

        correct_lat = [r.latency_ms for r in rows if r.correct and r.latency_ms is not None]
        wrong_lat = [r.latency_ms for r in rows if not r.correct and r.latency_ms is not None]

        multiclass = None
        labels = labels_of(run)
        if mode == "multiclass" and labels:
            multiclass = multiclass_report(rows, labels)

        judge = None
        if mode in ("judge_passfail", "judge_scale"):
            counts = Counter(r.pred for r in rows)
            values = ["1", "0"] if mode == "judge_passfail" else ["5", "4", "3", "2", "1"]
            names = {"1": "pass", "0": "fail"} if mode == "judge_passfail" else {v: v for v in values}
            judge = {
                "kind": "passfail" if mode == "judge_passfail" else "scale",
                "distribution": [{"score": names[v], "count": counts.get(v, 0)} for v in values]
                + [{"score": "unparseable", "count": counts.get(None, 0)}],
            }

        return {
            "roc": roc_curve(rows, positive) if has_probs else None,
            "pr": pr_curve(rows, positive) if has_probs else None,
            "score_distribution": score_distribution(rows, positive) if has_probs else None,
            "calibration": calibration(rows, positive) if has_probs else None,
            "multiclass": multiclass,
            "judge": judge,
            "length_slices": accuracy_by_length([(len(s["input"]), s["correct"]) for s in scored]),
            "latency_points": [
                {"index": i, "latency_ms": r.latency_ms, "correct": r.correct}
                for i, r in enumerate(rows)
                if r.latency_ms is not None
            ],
            "latency_by_outcome": {"correct": latency_stats(correct_lat), "incorrect": latency_stats(wrong_lat)},
        }

    @app.get("/api/runs/{run_id}/export.csv")
    def export_csv(run_id: str) -> Response:
        get_run_or_404(run_id)
        buf = io.StringIO()
        out = csv.writer(buf)
        cols = ["sample_id", "input", "target", "output", "pred", "correct", "probability", "latency_ms", "stop_reason", "error"]
        out.writerow(cols)
        for s in store.all_samples(run_id):
            out.writerow([_csv_text(s[c]) if c in ("input", "output", "error") else s[c] for c in cols])
        return Response(
            buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="owl-{run_id}.csv"'},
        )

    @app.get("/api/compare")
    def compare(run_ids: str, limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
        ids = [i for i in run_ids.split(",") if i]
        if len(ids) < 1:
            raise HTTPException(422, "run_ids must list at least one run")
        runs = [get_run_or_404(i) for i in ids]
        by_run = {i: {s["sample_id"]: s for s in store.all_samples(i)} for i in ids}

        shared = set.intersection(*(set(d) for d in by_run.values()))
        order = sorted(shared, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))
        disagreements: list[dict[str, Any]] = []
        for sid in order:
            preds = {i: by_run[i][sid]["pred"] for i in ids}
            if len(set(preds.values())) > 1:
                first = by_run[ids[0]][sid]
                disagreements.append(
                    {
                        "sample_id": sid,
                        "input": first["input"],
                        "target": first["target"],
                        "preds": preds,
                        "correct": {i: by_run[i][sid]["correct"] for i in ids},
                    }
                )
        outcomes = {i: [by_run[i][sid]["correct"] for sid in order] for i in ids}
        n_shared = len(order)
        n_right = [sum(outcomes[i][k] for i in ids) for k in range(n_shared)]
        pairwise = []
        for a, b in combinations(ids, 2):
            a_only = sum(1 for k in range(n_shared) if outcomes[a][k] and not outcomes[b][k])
            b_only = sum(1 for k in range(n_shared) if outcomes[b][k] and not outcomes[a][k])
            same = sum(1 for sid in order if by_run[a][sid]["pred"] == by_run[b][sid]["pred"])
            pairwise.append(
                {
                    "a": a,
                    "b": b,
                    "agreement": same / n_shared if n_shared else 0.0,
                    "a_only_correct": a_only,
                    "b_only_correct": b_only,
                    "mcnemar_p": mcnemar_exact(a_only, b_only),
                }
            )
        rocs = {}
        for r in runs:
            pos = r["config"].get("positive_class")
            if pos is not None:
                curve = roc_curve(store.rows_for_metrics(r["id"]), pos)
                if curve:
                    rocs[r["id"]] = curve
        examples = example_tasks()
        return {
            "runs": [summary(r, examples) for r in runs],
            "sample_ids": order,
            "outcomes": outcomes,
            "coverage": {
                "all_correct": sum(1 for c in n_right if c == len(ids)),
                "all_wrong": sum(1 for c in n_right if c == 0),
                "any_correct": sum(1 for c in n_right if c > 0),
            },
            "pairwise": pairwise,
            "roc": rocs,
            "shared_samples": len(shared),
            "total_disagreements": len(disagreements),
            "disagreements": disagreements[:limit],
        }

    # --- built dashboard ----------------------------------------------------
    if dist.is_dir() and (dist / "index.html").is_file():
        index = dist / "index.html"

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404, "not found")
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)

    return app


__all__ = ["create_app", "list_tasks"]
