"""`owl` command line interface."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import typer

from owl.runner import run_task
from owl.store import SampleRecord, Store, default_db_path
from owl.tasks import list_tasks

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Owl Harness: run and inspect model evaluations.")


def parse_model_args(items: list[str]) -> dict[str, Any]:
    """Turn ['key=value', ...] into a dict; values are parsed as JSON when possible."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"model arg must look like key=value, got '{item}'")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _fmt(metrics: dict[str, float], key: str, spec: str = ".3f") -> str:
    return format(metrics[key], spec) if key in metrics else "-"


@app.command()
def run(
    task: str = typer.Argument(..., help=f"Task name. Built in: {', '.join(list_tasks())}"),
    model: list[str] = typer.Option(..., "--model", "-m", help="provider/model. Repeat to run several, one at a time."),
    model_arg: list[str] = typer.Option([], "-M", help="Model arg key=value (applies to every --model)."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Run only the first N samples."),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume an interrupted run by id (one --model)."),
    max_concurrency: Optional[int] = typer.Option(None, "--max-concurrency", help="Override the provider default."),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-sample timeout in seconds."),
    retries: int = typer.Option(2, "--retries", help="Retries per sample after a failure."),
) -> None:
    """Run TASK against one or more models, sequentially."""
    if resume and len(model) != 1:
        raise typer.BadParameter("--resume works with exactly one --model")
    args = parse_model_args(model_arg)
    store = Store()

    def progress(done: int, total: int, rec: SampleRecord) -> None:
        status = "ERR" if rec.error else ("ok " if rec.correct else "bad")
        typer.echo(f"\r  {done}/{total} {status}", nl=False)

    for m in model:
        typer.echo(f"running {task} on {m}")
        try:
            run_id = asyncio.run(
                run_task(
                    task,
                    m,
                    store=store,
                    model_args=args,
                    limit=limit,
                    max_concurrency=max_concurrency,
                    timeout_s=timeout,
                    retries=retries,
                    resume_run_id=resume,
                    on_progress=progress,
                )
            )
        except Exception as e:
            typer.echo("")
            typer.secho(f"error: {type(e).__name__}: {e}", fg="red", err=True)
            raise typer.Exit(1)
        typer.echo("")
        r = store.get_run(run_id)
        met = store.get_metrics(run_id)
        typer.echo(
            f"  run {run_id} [{r['status'] if r else '?'}]  n={_fmt(met, 'n', '.0f')} "
            f"acc={_fmt(met, 'accuracy')} (se {_fmt(met, 'stderr')}) p={_fmt(met, 'precision')} "
            f"r={_fmt(met, 'recall')} f1={_fmt(met, 'f1')} unparseable={_fmt(met, 'unparseable', '.0f')} "
            f"median={_fmt(met, 'latency_median_ms', '.1f')}ms wall={_fmt(met, 'wall_time_s', '.1f')}s"
        )


@app.command()
def ls() -> None:
    """List recorded runs."""
    store = Store()
    runs = store.list_runs()
    if not runs:
        typer.echo(f"no runs in {default_db_path()}")
        return
    typer.echo(f"{'id':<22}{'task':<18}{'model':<24}{'status':<10}{'n':>5}{'acc':>8}{'f1':>8}{'med ms':>9}")
    for r in runs:
        m = store.get_metrics(r["id"])
        typer.echo(
            f"{r['id']:<22}{r['task']:<18}{r['model']:<24}{r['status']:<10}"
            f"{_fmt(m, 'n', '.0f'):>5}{_fmt(m, 'accuracy'):>8}{_fmt(m, 'f1'):>8}{_fmt(m, 'latency_median_ms', '.1f'):>9}"
        )


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind."),
) -> None:
    """Serve the API and the dashboard."""
    import uvicorn

    from owl.api import create_app

    loopback = {"127.0.0.1", "localhost", "::1"}
    if host in loopback:
        allowed: set[str] | None = loopback
    else:
        allowed = None
        typer.secho(
            f"warning: serving on {host} with no login. Anyone who can reach this address can run models "
            "and upload files. Use the default 127.0.0.1 unless the network is trusted.",
            fg="yellow",
            err=True,
        )
    uvicorn.run(create_app(Store(), allowed_hosts=allowed), host=host, port=port)


if __name__ == "__main__":
    app()
