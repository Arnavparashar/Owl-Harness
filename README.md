# Owl Harness

Owl Harness benchmarks language models and classifiers on your own data, the same way every time.

- **Any model.** Local models through Ollama or Hugging Face `transformers`, any OpenAI-compatible API, or any model behind a REST endpoint described by a JSON request template. More providers install as plugins.
- **Any labelled dataset.** Upload CSV, TSV, JSONL, JSON or Parquet, or import a split from the Hugging Face Hub. Map the columns, pick a scoring mode, and write the prompt with any column as a variable.
- **One evaluation standard.** Every run, for every provider and scoring mode, is held to the same rules (below), so numbers from different models are comparable.
- **A dashboard.** Start runs, watch progress, read results with confidence intervals, and compare models with significance tests. Everything runs and stays on your machine.

![Overview](docs/screenshots/overview.jpg)

## The evaluation standard

1. **Fixed data.** Each run stores the content hash of the exact samples it used. Hub imports are saved as local snapshots with their revision.
2. **Deterministic by default.** Temperature 0 and seed 42, passed to every provider that supports a seed (Ollama, OpenAI-compatible, Hugging Face, HTTP).
3. **Everything recorded.** The resolved prompt templates, scorer configuration (labels, normalisation, rubric, judge), provider settings (never secrets), the Owl version, and the model version or digest the provider reports (Ollama digest, Hub revision, served model name and fingerprint).
4. **Uncertainty with every headline number.** Accuracy and pass rate with standard error and a 95% Wilson interval; F1 and macro-F1 with a seeded bootstrap interval; a 1–5 mean score with a normal interval. A "small sample" badge appears when an interval is 0.10 wide or more.
5. **Nothing dropped.** An answer that cannot be parsed is counted, shown, and scored as wrong.
6. **One model at a time.** Models in a job run one after another, so they do not compete for the GPU or CPU.
7. **Honest latency.** Latency is measured around the model call only, after the model has loaded and answered one warm-up request.

## Scoring modes

| Mode | The model answers | Metrics |
|---|---|---|
| Yes / no | `1` or `0` to a question you write; you say which target value means yes | accuracy, precision, recall, specificity, F1, MCC, balanced accuracy, confusion matrix; ROC, AUC, precision–recall, calibration and a draggable threshold for models that return a probability |
| Multi-class | one label from the set (taken from the target column, editable; `{labels}` lists them) | accuracy, macro-F1, weighted-F1, per-class precision/recall/F1/support, N×N confusion matrix |
| Exact match | free text that must equal the target | accuracy; case, whitespace and punctuation normalisation are separate switches, stored on the run |
| Judge-graded | free text graded by a judge model against a rubric (pass/fail or 1–5), with an optional reference answer | pass rate or mean score with intervals, score distribution; the judge's full reply is kept for every sample |

Multi-class matching is case-insensitive after trimming and removing `<think>` blocks. An answer that names no label, or several, is unparseable. A judge reply with no verdict is unparseable too, and it counts as fail, or as 1 on the 1–5 scale. Owl warns, but does not block, when the judge is also one of the models being evaluated.

## Providers

| Provider | Model string | Notes |
|---|---|---|
| Ollama | `ollama/<model>` | Native `/api/chat` with `think: false`; installed models are listed automatically; records the model digest. |
| OpenAI-compatible API | `openai/<model>` | OpenAI, LM Studio, vLLM, llama.cpp server… Settings: `base_url`, `api_key_env` (the *name* of an environment variable), `max_tokens_param`. |
| Hugging Face (local) | `hf/<hub id or folder>` | `mode`: `causal-lm` (chat template, generation) or `sequence-classification` (a classifier head; `label_map` maps the model's labels to the test's answers, and P(positive) is stored for yes/no tests). CPU by default, `device` to change. Needs the `hf-models` extra. |
| Custom HTTP endpoint | `http/<name>` | `url`, `method`, `headers` (credentials as `${ENV:NAME}`), a JSON `body` template, `answer_path`, optional `probability_path`, `timeout_s`, `warmup`. Template variables: `{{input}}`, `{{system}}`, `{{messages}}`, `{{model}}`, `{{temperature}}`, `{{max_tokens}}`, `{{seed}}` and any dataset column as `{{column}}`. |
| Mock | `mock/<mode>` | Deterministic fake models for trying the app and for tests. |

For example, the `http` provider can benchmark Ollama through its own REST API, with no Ollama-specific code:

```powershell
owl run prompt_injection --model http/qwen3.5:0.8b --limit 5 `
  -M url=http://127.0.0.1:11434/api/chat -M answer_path=message.content `
  -M 'body={"model":"{{model}}","messages":"{{messages}}","stream":false,"think":false,"options":{"temperature":"{{temperature}}","num_predict":"{{max_tokens}}","seed":"{{seed}}"}}'
```

### Plugins

Providers and tests are discovered through entry points, so a separate package can add either one without changes to Owl.

- [`examples/plugins/owl-laya`](examples/plugins/owl-laya/README.md) is a provider plugin for Laya, a yes/no decision model, installable with `pip install -e`.
- See [Writing a provider plugin](#writing-a-provider-plugin) and [Writing a test plugin](#writing-a-test-plugin).

## The dashboard

| | |
|---|---|
| ![New run](docs/screenshots/new-run.jpg) | ![Run detail](docs/screenshots/run-detail-binary.jpg) |
| ![Compare](docs/screenshots/compare.jpg) | |

- **Overview:** live progress of running jobs, a leaderboard per test (each model's best main metric among its largest runs, with 95% intervals), recent runs, and accuracy against median latency coloured by model.
- **New run:** a four-step wizard.
  - *Test:* a built-in or example test, or a library dataset with column mapping, scoring mode, labels, normalisation or rubric, and editable system and user templates with a live preview rendered from a real row.
  - *Models:* saved models, ad hoc models, and the judge.
  - *Settings:* temperature, max tokens, seed, sample limit, timeout, retries and parallel requests.
  - *Review:* a summary, then start.
- **Progress:** a bar per model, the live correct count and current accuracy, an ETA, and Cancel.
- **Runs:** a sortable, filterable table, virtualised so thousands of runs scroll smoothly. Select runs to compare, export samples to CSV, or delete runs.
- **Run detail:**
  - *Header:* chips for the test, scoring mode, sample count, dataset hash (click to copy) and Owl version.
  - *Headline:* the main metric with its interval, plus F1 or macro-F1.
  - *Secondary numbers:* those that apply to the mode, then a latency strip and a large confusion matrix.
  - *Tabs:* Overview, Latency, Threshold (yes/no with probabilities), Samples (filters, search, and a side panel with the exact prompt sent and the judge's reply), and Config (every recorded setting).
- **Compare:**
  - main metrics with error bars, and latency;
  - agreement and exact McNemar tests for each pair;
  - ROC overlays, a per-sample outcome grid, and the samples where the models disagree.
  - Each model keeps the same colour on every page, from a colour-blind-safe palette.
- **Models:** saved model configurations. Test connection loads the model and sends one prompt. Installed Ollama models can be added in one click.
- **Datasets:** upload by drag and drop, Hub import, a preview of the first rows, the value distribution of any column, and delete.

The interface is a dark "glass" design: translucent surfaces over three static colour fields, with Inter bundled locally. Text meets WCAG AA contrast on every surface, there is a solid fallback for browsers without `backdrop-filter`, and the motion (card fade-in, count-up, chart draw-in, a sheen on running progress bars) is switched off when the system asks for reduced motion.

## Setup

Requirements: Python 3.10+ and Node.js 18+ (developed on Python 3.12 and Node 24). For local models, [Ollama](https://ollama.com) with the models you want, or the `hf-models` extra.

```powershell
# backend
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[hf,hf-models,parquet,dev]"

# frontend
cd ..\frontend
npm install
npm run build        # outputs frontend/dist, which `owl serve` serves
```

Extras, so the base install stays light:
- `hf`: Hub dataset import and the prompt-injection example.
- `hf-models`: the `hf` provider; CPU PyTorch and `transformers`.
- `parquet`: Parquet uploads.
- `dev`: pytest.

Every dependency is pinned to an exact version in `backend/pyproject.toml`, `frontend/package.json` and `frontend/package-lock.json`. `backend/requirements.lock` is a full `pip freeze` of the environment used.

The database lives at `backend/data/owl.db` (override it with the `OWL_DB` environment variable). Schema changes are applied by numbered migrations that keep every existing row, and the database is copied to `owl.db.bak-v<N>` before an upgrade.

## Running

```powershell
cd backend
.venv\Scripts\owl serve --port 8000          # dashboard at http://127.0.0.1:8000
.venv\Scripts\owl run prompt_injection --model ollama/qwen3.5:0.8b --limit 5
.venv\Scripts\owl ls
```

- `owl run` takes a registered test.
  - `-M key=value` passes a provider setting; values are parsed as JSON when possible.
  - `--limit`, `--max-concurrency`, `--timeout`, `--retries` and `--resume <run id>` are also available.
- Tests on uploaded or imported datasets are defined and started from the dashboard; the API is `POST /api/jobs`.

## Writing a provider plugin

A provider subclasses `owl.providers.Provider` and implements `generate`. It is registered under the `owl.providers` entry point:

```python
# my_pkg/provider.py
import time
from owl.providers import ArgSpec, Provider
from owl.types import GenerateConfig, Message, ModelOutput


class EchoProvider(Provider):
    name = "echo"
    display_name = "Echo"
    description = "Repeats the input back, optionally shouting."      # shown under the picker
    arg_specs = (ArgSpec("shout", "boolean", False, "Upper-case the reply."),)
    max_concurrency = 1

    def __init__(self, model: str, shout: bool = False, **model_args):
        super().__init__(model, **model_args)  # rejects settings it does not know
        self.shout = shout

    async def setup(self) -> None:  # optional: load weights, warm up (not timed)
        ...

    async def generate(self, messages: list[Message], config: GenerateConfig) -> ModelOutput:
        t0 = time.perf_counter()
        text = messages[-1].content
        return ModelOutput(text=text.upper() if self.shout else text, latency_ms=(time.perf_counter() - t0) * 1000)

    @classmethod
    async def list_models(cls, **model_args) -> list[str]:            # optional: model suggestions
        return ["anything"]
```

```toml
[project.entry-points."owl.providers"]
echo = "my_pkg.provider:EchoProvider"
```

The dashboard builds the provider's form from `arg_specs`, and settings it does not declare are rejected.
- `affects_models=True` marks a setting that changes which models exist, such as a server URL.
- Set `ModelOutput.probability` to P(positive) if the model has one, and threshold analysis appears automatically.
- `binary_only = True` marks a provider that only answers yes/no questions.
- `task_inputs` asks for test-level values: `question`, `labels` or `positive_class`.
- `model_info()` returns version facts to record on each run.

## Writing a test plugin

A test is a `Task`: a dataset, a solver chain, a scorer and a generation config. Register a function that returns one under the `owl.tasks` entry point, and it appears in the dashboard and in `owl run`:

```toml
[project.entry-points."owl.tasks"]
my_test = "my_pkg.tasks:my_test"      # a zero-argument function returning owl.tasks.Task
```

Set `Task.description` for the picker, `scoring_mode`, and `positive_class="1"` for a yes/no test.

## Security

`owl serve` has no login and is meant for your own machine.
- **Network:** by default it listens on `127.0.0.1` only and answers only to `localhost`-style Host headers, which guards against DNS rebinding.
- **Other websites:** it refuses state-changing requests that come from another website's origin.
- **Other addresses:** binding another address with `--host` prints a warning.
- **Uploads:** files are parsed as data and never executed.
- **Credentials:** API keys are never typed into the browser or stored. Settings name an environment variable, and the server reads it. The `http` provider rejects credential-looking headers that are not `${ENV:NAME}` references.

## Tests

```powershell
cd backend
.venv\Scripts\python -m pytest
```

The tests cover:
- **Metrics:** hand-computed cases for binary and multi-class metrics, intervals, AUC, calibration and McNemar.
- **Scoring:** label matching, judge-reply parsing and exact-match normalisation.
- **Providers:** the `http` provider against a local test server, and the `hf` provider with model loading replaced by a fake (no downloads).
- **Storage and API:** the model registry, dataset upload including Parquet, Hub import (mocked), background jobs and cancellation, the database migration on a v0.1 database, and the local-origin guard.

The frontend has no automated tests. It is type-checked, and each page was checked by hand in a browser.

## Examples

[`examples/README.md`](examples/README.md) describes the two shipped example tests and gives results from one machine, including the prompt-injection benchmark that started this project.

## License

MIT, see [LICENSE](LICENSE).
