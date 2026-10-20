# owl-laya

A provider plugin that lets Owl Harness evaluate [Laya](https://pypi.org/project/laya/), a small encoder that answers yes/no questions with a probability.

It is a separate package on purpose: it shows how any model can be added to Owl without touching Owl's code, and it keeps Laya's dependencies (PyTorch) out of the base install.

## Install

Into the same environment as Owl:

```powershell
cd examples\plugins\owl-laya
..\..\..\backend\.venv\Scripts\python -m pip install -e .
```

This installs `laya` and CPU PyTorch from PyPI. Owl discovers the provider through the `owl.providers` entry point declared in `pyproject.toml`; restart `owl serve` afterwards and "Laya (plugin)" appears in the model picker.

## Use

```powershell
owl run prompt_injection --model laya/english
owl run prompt_injection --model laya/english -M threshold=0.4
```

- Models: `english`, `multilingual`, `typed-decisions` (the names Laya ships). Weights for a model you have not used before are downloaded from Hugging Face on first run.
- Yes/no tests only. Owl passes the test's question to Laya (the provider declares `task_inputs = ("question",)`); the system prompt is not used, because Laya is not a prompted model.
- `threshold` (default 0.5): answer 1 when P(yes) is at least this value. The raw P(yes) is stored on every sample, so the dashboard's threshold analysis and ROC work without re-running.
- Runs on CPU, one forward pass at a time, with one warm-up call before timing starts.

## Without the plugin

Runs recorded with Laya stay in the database and open normally in the dashboard when the plugin is not installed; only starting new Laya runs needs it.

Laya can also be benchmarked with no plugin at all if it is served behind an HTTP endpoint: use Owl's `http` provider with a request template and an answer path.
