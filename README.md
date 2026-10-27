# Pynnger

A plan-first Python project generator built to lower the cost of getting working code out of a
model. It plans once with a strong model, writes each file with a cheap one, reviews the result
locally at zero token cost, and sends only the files that failed back for repair. A local server
and a VS Code extension put it in the editor.

**Status: built and tested, not yet measured against a real model.** Everything below describes
what the code does. The claim that it uses fewer tokens per working result than one big prompt is
a design goal until `pynnger eval` has been run against a real model (see [Evaluating](#evaluating)).

## Where the tokens are saved

| Mechanism | What it does | Code |
|---|---|---|
| Local router | Classifies the task and sets token budgets with no model call | `pipeline/router.py` |
| Plan once | One schema-forced JSON plan names the files, their order and their public API | `pipeline/orchestrator.py`, `schemas.py` |
| Cached plan header | Every file call reuses the plan as a cache prefix instead of paying for it again | `backends/anthropic_backend.py`, `prompts.py` |
| Stubs, not source | A file sees signatures and docstrings of what it depends on, not full source | `context/skeleton.py` |
| Context selection | For an existing project, sends the best-matching files in full and the rest as outlines (BM25 over names and docstrings plus import-graph neighbours) | `context/selector.py` |
| Cheap model for code | Planner and coder are routed separately (`ModelRouting`) | `config.py` |
| Zero-token review | Syntax, annotations, docstrings, security patterns, import resolution, ruff, optional mypy and pytest | `review/` |
| Tools fix style | ruff formats and sorts imports; the model is never asked to | `review/tools.py` |
| Targeted repair | Only files with errors go back, with line numbers, at most two rounds | `pipeline/orchestrator.py` |
| Hard ceiling | A build stops at `max_total_tokens` (default 400k) | `pipeline/orchestrator.py` |

Every call is recorded per stage in a ledger and priced from the model's published rates,
including cache reads and writes (`tokens.py`, `pricing.py`). The prices are a snapshot; override
them with `PYNNGER_PRICES=prices.json` when they change.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# Offline demo: replays a scripted project through the real pipeline, no API key needed.
pynnger build "word frequency CLI" --backend mock --out /tmp/demo
pynnger review /tmp/demo --fail-under 0.9      # zero tokens; fits in CI

# Real models (Anthropic API):
export ANTHROPIC_API_KEY=...
pynnger build "a FastAPI service with a /health route and tests" --out ./service
pynnger build "add rate limiting" --root ./service        # extend an existing project
```

The default routing is planner `claude-sonnet-5-5`, coder and repair `claude-haiku-5-5`. Change
it with `--plan-model`, `--code-model`, `--repair-model` or `PYNNGER_MODEL_*`.

For a self-hosted or other OpenAI-compatible endpoint use `--backend openai --base-url
http://localhost:8000/v1` (see [docs/serving.md](docs/serving.md)).

## The reviewer

`pynnger review` and the pipeline's review stage share one scorer. Findings carry a code:

| Code | Meaning |
|---|---|
| PY001 | Syntax error |
| PY010 / PY020 | Missing annotations / docstrings (errors once coverage drops under 90% / 80%) |
| PY030 / PY031 | Bare `except` / swallowed exception |
| PY040 | Mutable default argument |
| PY050 | `print` in library code |
| PY060 / PY061 / PY062 / PY063 | `eval`/`exec`, shell execution, unsafe deserialisation, TLS verification off |
| PY070 / PY071 | Hard-coded secret / insecure secret default |
| PY080 | SQL built from strings |
| PY090 / PY091 | Import of a missing module / missing name (resolved across files) |
| PY100 | Third-party import not declared as a dependency |
| PY900 | Output cut off at the token limit |
| RUFF-F… | Pyflakes findings (undefined names are errors) |
| TEST001 / TEST002 | Generated tests failed / timed out (only with `--run-tests`) |

The production score is `0.25·types + 0.15·docstrings + 0.30·correctness + 0.15·security +
0.15·tests` (tests: passed 1.0, present 0.5, failed or missing 0). The weights are a judgement
call, kept simple so they can be audited.

## Evaluating

```bash
pynnger eval --reference --backend mock                          # no model: checks the harness
pynnger eval --confirm-sandboxed --arms pipeline,baseline        # live: needs an API key
```

`eval/tasks/` holds four stdlib-only tasks (`slugify`, `ttlcache`, `csvsummary`, and a multi-file
`todo_service`). Each has a prompt, hidden acceptance tests the model never sees, and a
hand-written reference solution. Two arms run on every task: the pipeline, and a single call to
the baseline model asking for the whole project. Both are scored by the same reviewer with the
model's own tests executed, and judged by whether the hidden tests pass. The report gives
accepted tasks, mean score, tokens, dollars and **cost per accepted task**, which is the number
this project exists to lower.

`--reference` replays the reference solutions through both arms with no model. All four pass
their hidden tests and score 1.0, so the harness, the acceptance tests and the scorer agree. It
says nothing about any model, and its token and cost columns are left blank on purpose.

A live eval executes model-written code, so it refuses to start without `--confirm-sandboxed`.
Run it in a container or VM. Four tasks and one run each is a smoke test, not a benchmark.

## Editor integration

```bash
pip install -e ".[server]"
export PYNNGER_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
pynnger serve --allow-root ~/code --token-env PYNNGER_TOKEN
```

Install `vscode-extension/pynnger-0.1.0.vsix` (or build it, see
[vscode-extension/README.md](vscode-extension/README.md)), run **Pynnger: Set Server Token**, then
use **Build Project**, **Add a Feature**, or **Review** from the command palette. The server never
writes to disk; the extension shows what it would create or replace and asks first.

## Training (optional, not run)

`pynnger/training/` contains a pipeline for turning the system into a fine-tuned model:

1. `synth.py` runs the real pipeline with a teacher model and keeps only builds the reviewer
   passes (optionally with their tests executed).
2. `builders.py` turns those builds into SFT rows (exact inference prompts, reviewed files as
   targets) and DPO pairs (a failing first attempt versus its accepted fix, plus synthetic
   degradations).
3. `filter.py` and `dedup.py` apply quality filters, MinHash near-duplicate removal and 13-gram
   decontamination.
4. `sft.py`, `dpo.py` and `merge.py` train and merge LoRA adapters with TRL and PEFT.

These scripts were written against the installed library APIs and are tested for data handling
and `--dry-run`, but **no training run has been made**. Check the terms of whichever teacher you
use: Anthropic's commercial terms restrict using Claude outputs to train competing models.

## Is self-hosting worth it?

Probably not at first. `pynnger breakeven` compares the API price per million tokens with a GPU
you pay for by the hour:

```
$ pynnger breakeven --gpu-hourly 2.5 --tokens-per-sec 1500 --utilization 0.4
API   (claude-haiku-5-5, 75% input, 0% cache hits): $0.200 / M tokens
Self-hosted at 40% utilisation:  $1.157 / M tokens
Break-even needs 231% utilisation: unreachable with these numbers.
```

The inputs there are illustrative, not measurements of any hardware. The point is the shape:
a self-hosted model only wins with fast hardware kept busy. Routing and the pipeline come first.

## Security notes

- Plan paths are validated (no absolute paths, `..`, drive letters or `.git`) and writes go
  through `resolve_inside`; the extension repeats the check and also refuses to write through a
  symlink that leaves the workspace.
- The server limits directories to `--allow-root`, can require a bearer token (compared in
  constant time), never writes files, and keeps `run_tests` off unless started with
  `--allow-run-tests`.
- The extension is disabled in untrusted workspaces, keeps the token in VS Code secret storage,
  and never sends it over plain `http` to a non-local host. `serverUrl` and `runTests` are
  machine-scoped so a repository cannot change them.
- Running generated tests (`--run-tests`, the live eval) executes model-written code. The temp
  directory and scrubbed environment are hygiene, not a sandbox.

## Known limits

- **No live-model run.** The Anthropic and OpenAI-compatible backends are tested against fakes
  and mock transports only. Prompts and token budgets are untuned.
- **Token savings are unmeasured**, as above. The ledger and eval harness exist to measure them.
- **The VS Code extension host code** (`extension.ts`) type-checks and packages, but was not run
  inside a live VS Code. The client, SSE parser and path-safety logic are tested, including
  against the real server.
- Python only. Linux only so far; Windows and macOS paths are untested.
- Cancelling a build stops the client but the server may finish the call in progress, and
  cancelled builds are not counted in the extension's tally.
- The reviewer is static analysis plus optional test execution. It finds the mistakes it has
  rules for, not all mistakes.

## Layout

```
src/pynnger/     pipeline, backends, review, context, server, CLI, evals, training
tests/           174 tests (pytest)
eval/tasks/      executable eval tasks with hidden acceptance tests and reference solutions
vscode-extension TypeScript extension (29 tests with node --test)
docs/            serving notes
```

```bash
pytest && ruff check src tests && mypy src/pynnger          # Python
cd vscode-extension && npm install && npm test             # extension
```
