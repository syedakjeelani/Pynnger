"""Eval harness: reference solutions pass, wrong ones are caught, and the CLI is wired up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pynnger import cli
from pynnger.backends.base import BackendError, Completion
from pynnger.backends.scripted import ScriptedBackend
from pynnger.config import Settings
from pynnger.evals import harness
from pynnger.evals.harness import (
    Acceptance,
    ArmResult,
    EvalError,
    EvalTask,
    evaluate,
    load_reference,
    load_tasks,
    parse_pytest_summary,
    reference_plan,
    run_acceptance,
    run_eval,
    score_files,
    summarize_results,
)
from pynnger.tokens import Usage

TASKS_DIR = Path(__file__).resolve().parents[1] / "eval" / "tasks"
TASK_IDS = ["csvsummary", "slugify", "todo_service", "ttlcache"]
SETTINGS = Settings(backend="mock")


@pytest.fixture(scope="module")
def tasks() -> dict[str, EvalTask]:
    return {t.id: t for t in load_tasks(TASKS_DIR)}


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "tasks": str(TASKS_DIR),
        "task": None,
        "arms": "pipeline,baseline",
        "out": str(tmp_path),
        "reference": True,
        "confirm_sandboxed": False,
        "run_tests": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# --------------------------------------------------------------------------- tasks


def test_all_tasks_load(tasks: dict[str, EvalTask]) -> None:
    assert sorted(tasks) == TASK_IDS
    for task in tasks.values():
        assert task.prompt.strip() and task.acceptance_file.is_file()


def test_prompts_do_not_leak_acceptance_tests(tasks: dict[str, EvalTask]) -> None:
    """The model must not see the hidden tests; a prompt quoting them would defeat the eval."""
    for task in tasks.values():
        assert "test_acceptance" not in task.prompt
        assert "acceptance" not in task.prompt.lower()


def test_load_tasks_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="not found"):
        load_tasks(tmp_path / "missing")
    with pytest.raises(EvalError, match="no tasks"):
        load_tasks(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "task.json").write_text(json.dumps({"id": "b", "tier": "x", "prompt": "p"}))
    with pytest.raises(EvalError, match="does not match"):
        load_tasks(tmp_path)
    (tmp_path / "a" / "task.json").write_text("{not json")
    with pytest.raises(EvalError, match="invalid task file"):
        load_tasks(tmp_path)
    (tmp_path / "a" / "task.json").write_text(json.dumps({"id": "a", "tier": "x", "prompt": "p"}))
    with pytest.raises(EvalError, match="missing"):
        load_tasks(tmp_path)


def test_unknown_task_filter(tasks: dict[str, EvalTask]) -> None:
    with pytest.raises(EvalError, match="unknown task id"):
        load_tasks(TASKS_DIR, ["nope"])
    assert [t.id for t in load_tasks(TASKS_DIR, ["slugify"])] == ["slugify"]


def test_reference_plan_is_a_valid_plan(tasks: dict[str, EvalTask]) -> None:
    from pynnger.schemas import ProjectPlan

    for task in tasks.values():
        files = load_reference(task)
        plan = ProjectPlan.model_validate(reference_plan(task, files))
        assert {f.path for f in plan.files} == set(files)
        assert [f.path for f in plan.implementation_order()][-1].startswith("tests/")


# --------------------------------------------------------------------------- the references


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_passes_acceptance(tasks: dict[str, EvalTask], task_id: str) -> None:
    task = tasks[task_id]
    result = run_acceptance(load_reference(task), task.acceptance_file)
    assert result.accepted, result.detail
    assert result.total >= 10


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_is_production_quality_by_our_own_reviewer(
    tasks: dict[str, EvalTask], task_id: str
) -> None:
    """If the reference fails the reviewer, the reviewer (or the bar) is wrong, not the model."""
    report = score_files(load_reference(tasks[task_id]), SETTINGS)
    assert report.passed, [i.model_dump() for i in report.issues if i.severity == "error"]
    assert report.tests == "passed"
    assert report.score >= 0.95


# --------------------------------------------------------------------------- failure detection


def _scripted(task: EvalTask, files: dict[str, str]) -> ScriptedBackend:
    return ScriptedBackend(reference_plan(task, files), files)


def test_wrong_solution_is_rejected_by_both_arms(tasks: dict[str, EvalTask]) -> None:
    task = tasks["slugify"]
    broken = {
        **load_reference(task),
        "slugify.py": (
            "def slugify(text, *, max_length=None, separator='-'):\n    return text.lower()\n"
        ),
    }
    for arm in ("pipeline", "baseline"):
        result = evaluate(task, arm, _scripted(task, broken), SETTINGS)
        assert not result.accepted
        assert result.acceptance.failed > 0
        assert result.note
        assert result.score < 1.0


def test_missing_module_is_reported_as_an_error(tasks: dict[str, EvalTask]) -> None:
    task = tasks["ttlcache"]
    wrong_name = {"cache.py": '"""x."""\n\nVALUE = 1\n'}
    result = evaluate(task, "baseline", _scripted(task, wrong_name), SETTINGS)
    assert not result.accepted
    assert result.acceptance.errors >= 1
    assert "ttlcache" in result.note


def test_acceptance_ignores_the_models_own_tests(tasks: dict[str, EvalTask]) -> None:
    """A passing self-written test must not rescue a wrong implementation."""
    task = tasks["slugify"]
    files = {
        "slugify.py": "def slugify(text, **kw):\n    return text\n",
        "tests/test_slugify.py": "def test_ok():\n    assert True\n",
    }
    result = run_acceptance(files, task.acceptance_file)
    assert not result.accepted and result.failed > 0


def test_infinite_loop_times_out(tasks: dict[str, EvalTask]) -> None:
    task = tasks["slugify"]
    result = run_acceptance({"slugify.py": "while True:\n    pass\n"}, task.acceptance_file, 3)
    assert result.timed_out and not result.accepted
    assert "3s" in result.detail


def test_unsafe_baseline_paths_are_dropped(tasks: dict[str, EvalTask]) -> None:
    cleaned = harness._clean_paths(
        {"../evil.py": "x", "/abs.py": "x", ".git/config": "x", "./ok.py": "y", "a\\b.py": "z"}
    )
    assert cleaned == {"ok.py": "y", "a/b.py": "z"}
    task = tasks["slugify"]
    result = evaluate(task, "baseline", _scripted(task, {"../escape.py": "x = 1\n"}), SETTINGS)
    assert result.files == 0 and "no files" in result.note


class _FailingBackend:
    def complete(self, **_: object) -> Completion:
        raise BackendError("rate limited")


class _TruncatedBackend:
    def complete(self, **_: object) -> Completion:
        return Completion(
            "=== FILE: slugify.py ===\ndef slugify(", Usage(10, 20), "m", "max_tokens"
        )


def test_backend_failure_is_recorded_not_raised(tasks: dict[str, EvalTask]) -> None:
    result = evaluate(tasks["slugify"], "baseline", _FailingBackend(), SETTINGS)  # type: ignore[arg-type]
    assert not result.accepted and result.files == 0
    assert "rate limited" in result.note
    assert result.score == 0.0 and result.acceptance.detail == "no files produced"


def test_truncated_baseline_reply_is_flagged(tasks: dict[str, EvalTask]) -> None:
    result = evaluate(tasks["slugify"], "baseline", _TruncatedBackend(), SETTINGS)  # type: ignore[arg-type]
    assert not result.accepted
    assert "truncated" in result.note


def test_unknown_arm(tasks: dict[str, EvalTask]) -> None:
    task = tasks["slugify"]
    with pytest.raises(EvalError, match="unknown arm"):
        evaluate(task, "magic", _scripted(task, {"a.py": ""}), SETTINGS)


# --------------------------------------------------------------------------- reporting


def test_parse_pytest_summary() -> None:
    assert parse_pytest_summary("....\n11 passed in 0.05s\n") == (11, 0, 0)
    assert parse_pytest_summary("FAILED x\n2 failed, 9 passed, 1 warning in 0.1s") == (9, 2, 0)
    assert parse_pytest_summary("ERROR collecting\n1 error in 0.02s") == (0, 0, 1)
    assert parse_pytest_summary("3 errors in 0.02s") == (0, 0, 3)
    assert parse_pytest_summary("no tests ran in 0.01s") == (0, 0, 0)
    assert parse_pytest_summary("") == (0, 0, 0)


def _result(arm: str, accepted: bool, score: float, tokens: int, cost: float) -> ArmResult:
    return ArmResult(
        task="t",
        tier="small",
        arm=arm,
        accepted=accepted,
        acceptance=Acceptance(passed=1 if accepted else 0),
        own_tests="passed",
        score=score,
        errors=0,
        warnings=0,
        files=1,
        lines=1,
        usage={"total": {"total": tokens}, "cost_usd": cost, "cost_complete": True},
        seconds=0.1,
    )


def test_summary_divides_cost_by_accepted_tasks() -> None:
    summary = summarize_results(
        [
            _result("pipeline", True, 1.0, 1000, 0.02),
            _result("pipeline", False, 0.5, 3000, 0.04),
            _result("baseline", False, 0.2, 500, 0.10),
        ]
    )
    assert summary["pipeline"]["accepted"] == 1
    assert summary["pipeline"]["usd_per_accepted"] == pytest.approx(0.06)
    assert summary["pipeline"]["tokens_per_accepted"] == 4000
    assert summary["pipeline"]["mean_score"] == 0.75
    assert summary["baseline"]["usd_per_accepted"] is None  # nothing accepted: no honest ratio


def test_run_eval_calls_progress_for_every_task_and_arm(tasks: dict[str, EvalTask]) -> None:
    seen: list[tuple[str, str]] = []
    chosen = [tasks["slugify"], tasks["ttlcache"]]
    report = run_eval(
        chosen,
        ["baseline"],
        lambda t: _scripted(t, load_reference(t)),
        SETTINGS,
        mode="reference",
        progress=lambda r: seen.append((r.task, r.arm)),
    )
    assert seen == [("slugify", "baseline"), ("ttlcache", "baseline")]
    assert report["summary"]["baseline"]["accepted"] == 2


# --------------------------------------------------------------------------- entry point


def test_reference_replay_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = harness.main(_args(tmp_path), SETTINGS)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "REFERENCE REPLAY" in captured.out
    (json_path,) = tmp_path.glob("*-reference.json")
    (md_path,) = tmp_path.glob("*-reference.md")
    report = json.loads(json_path.read_text())
    assert report["mode"] == "reference"
    assert len(report["results"]) == 8
    assert all(r["accepted"] for r in report["results"])
    assert report["summary"]["pipeline"]["accepted"] == 4
    assert report["summary"]["baseline"]["accepted"] == 4
    # A replay must never print token or cost figures that could be mistaken for measurements.
    table = md_path.read_text().split("| task |")[0]
    assert "$" not in table


def test_cli_eval_reference_flag(tmp_path: Path) -> None:
    code = cli.main(
        ["eval", "--tasks", str(TASKS_DIR), "--task", "slugify", "--reference",
         "--arms", "pipeline", "--backend", "mock", "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (json_path,) = tmp_path.glob("*.json")
    assert [r["arm"] for r in json.loads(json_path.read_text())["results"]] == ["pipeline"]


def test_live_run_refuses_without_sandbox_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    live = Settings(backend="anthropic")
    assert harness.main(_args(tmp_path, reference=False), live) == 2
    assert "--confirm-sandboxed" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_mock_backend_cannot_run_a_live_eval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = harness.main(_args(tmp_path, reference=False, confirm_sandboxed=True), SETTINGS)
    assert code == 2
    assert "only supports --reference" in capsys.readouterr().err


def test_bad_arms_and_task_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert harness.main(_args(tmp_path, arms="pipeline,magic"), SETTINGS) == 2
    assert harness.main(_args(tmp_path, arms=""), SETTINGS) == 2
    assert harness.main(_args(tmp_path, task=["nope"]), SETTINGS) == 2
    assert "unknown task id" in capsys.readouterr().err


def test_rejected_reference_is_a_harness_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If a reference stops passing its own acceptance test, the self-check must exit non-zero."""
    real = harness.load_reference

    def broken(task: EvalTask) -> dict[str, str]:
        files = real(task)
        files["slugify.py"] = "def slugify(text, **kw):\n    return ''\n"
        return files

    monkeypatch.setattr(harness, "load_reference", broken)
    code = harness.main(_args(tmp_path, task=["slugify"], arms="baseline"), SETTINGS)
    assert code == 1
    assert "reference solution was rejected" in capsys.readouterr().err
