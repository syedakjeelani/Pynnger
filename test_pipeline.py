import copy
from pathlib import Path

import pytest

from pynnger.backends.scripted import ScriptedBackend
from pynnger.config import Settings
from pynnger.demo import DEMO_FILES, DEMO_FIRST_PASS, DEMO_PLAN, DEMO_TASK
from pynnger.pipeline.orchestrator import Pipeline
from pynnger.pipeline.router import route
from pynnger.schemas import BuildRequest


def demo(**kwargs: object) -> ScriptedBackend:
    return ScriptedBackend(DEMO_PLAN, DEMO_FILES, DEMO_FIRST_PASS, **kwargs)  # type: ignore[arg-type]


def stages(backend: ScriptedBackend) -> list[tuple[str, str]]:
    return [(c["stage"], c["path"]) for c in backend.calls]


def test_end_to_end_repairs_only_the_failing_file() -> None:
    backend = demo()
    result = Pipeline(backend).run(BuildRequest(task=DEMO_TASK))
    assert result.report.passed and result.stopped is None
    assert [s for s, _ in stages(backend)].count("plan") == 1
    assert [s for s, _ in stages(backend)].count("code") == 5
    assert [(s, p) for s, p in stages(backend) if s == "repair"] == [
        ("repair", "src/wordfreq/cli.py")
    ]
    assert "except OSError" in result.files["src/wordfreq/cli.py"]
    assert set(result.files) == set(DEMO_FILES)


def test_ledger_attributes_tokens_to_stages_and_routes_models() -> None:
    backend = demo()
    result = Pipeline(backend).run(BuildRequest(task=DEMO_TASK))
    by_stage = result.usage["by_stage"]
    assert set(by_stage) == {"plan", "code", "repair"}
    assert result.usage["requests"] == 7
    models = {c["stage"]: c["model"] for c in backend.calls}
    assert models["plan"] == "claude-sonnet-5-5" and models["code"] == "claude-haiku-5-5"
    assert result.usage["cost_complete"] is True and result.usage["cost_usd"] > 0


def test_repair_pairs_capture_before_and_after() -> None:
    result = Pipeline(demo()).run(BuildRequest(task=DEMO_TASK))
    assert [p.path for p in result.repair_pairs] == ["src/wordfreq/cli.py"]
    pair = result.repair_pairs[0]
    assert "except:" in pair.before and "except:" not in pair.after
    assert {i.code for i in pair.issues} >= {"PY030"}


def test_pairs_can_be_disabled() -> None:
    result = Pipeline(demo(), Settings(collect_pairs=False)).run(BuildRequest(task=DEMO_TASK))
    assert result.repair_pairs == []


def test_no_repair_rounds_leaves_the_failure_visible() -> None:
    backend = demo()
    result = Pipeline(backend).run(BuildRequest(task=DEMO_TASK, max_repair_rounds=0))
    assert not result.report.passed
    assert "repair" not in {s for s, _ in stages(backend)}


def test_event_stream_order() -> None:
    kinds = [e.type for e in Pipeline(demo()).events(BuildRequest(task=DEMO_TASK))]
    assert kinds[:3] == ["route", "plan", "file_start"]
    assert kinds.count("file") == 5 and kinds.count("repair_start") == 1
    assert kinds.index("review") < kinds.index("repair_start") < len(kinds) - 2
    assert kinds[-1] == "final"


def test_sse_formatting_round_trips() -> None:
    event = next(iter(Pipeline(demo()).events(BuildRequest(task=DEMO_TASK))))
    text = event.to_sse()
    assert text.startswith("event: route\ndata: {") and text.endswith("\n\n")


def test_invalid_plan_json_is_retried_once() -> None:
    backend = demo(fail_json_once=True)
    result = Pipeline(backend).run(BuildRequest(task=DEMO_TASK))
    assert result.report.passed
    assert [s for s, _ in stages(backend)][:2] == ["plan", "plan_retry"]


def test_unrecoverable_plan_stops_cleanly() -> None:
    class AlwaysBad(ScriptedBackend):
        def complete(self, **kwargs):  # type: ignore[no-untyped-def]
            comp = super().complete(**kwargs)
            return type(comp)("garbage", comp.usage, comp.model, comp.stop_reason)

    result = Pipeline(AlwaysBad(DEMO_PLAN, DEMO_FILES)).run(BuildRequest(task=DEMO_TASK))
    assert result.stopped and result.stopped.startswith("error:") and result.files == {}


def test_token_ceiling_stops_the_build_and_keeps_partial_work() -> None:
    result = Pipeline(demo(), Settings(max_total_tokens=1500)).run(BuildRequest(task=DEMO_TASK))
    assert result.stopped == "budget"
    assert result.plan is not None  # planning finished before the ceiling was hit
    assert 0 < len(result.files) < len(DEMO_FILES)  # some files were written, not all


def test_truncated_output_is_detected_and_repaired_with_a_bigger_limit() -> None:
    backend = demo(truncate=frozenset({"src/wordfreq/counter.py"}))
    result = Pipeline(backend).run(BuildRequest(task=DEMO_TASK))
    assert result.report.passed
    code_limit = next(c["max_tokens"] for c in backend.calls if c["stage"] == "code")
    repair = next(
        c for c in backend.calls if c["stage"] == "repair" and c["path"].endswith("counter.py")
    )
    assert repair["max_tokens"] == 2 * code_limit


def test_prompts_carry_stubs_instead_of_full_dependency_source() -> None:
    backend = demo()
    Pipeline(backend).run(BuildRequest(task=DEMO_TASK))
    cli_call = next(
        c for c in backend.calls if c["stage"] == "code" and c["path"].endswith("cli.py")
    )
    assert "def count_words(text: str) -> Counter[str]" in cli_call["prompt"]
    assert "_WORD.findall" not in cli_call["prompt"]  # body of counter.py never sent


def test_plan_call_requests_schema_forced_output() -> None:
    backend = demo()
    Pipeline(backend).run(BuildRequest(task=DEMO_TASK))
    assert backend.calls[0]["schema"]["title"] == "ProjectPlan"


def test_modify_flow_reads_existing_file_and_resolves_existing_imports(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "db.py").write_text(
        '"""DB."""\n\n\ndef connect() -> str:\n    """Connect."""\n    return "conn"\n'
    )
    (tmp_path / "app" / "service.py").write_text('"""Service."""\n\nVERSION = 1\n')
    plan = copy.deepcopy(DEMO_PLAN)
    plan["files"] = [
        {
            "path": "app/service.py",
            "purpose": "use the database",
            "action": "modify",
            "depends_on": [],
            "public_api": ["def run() -> str"],
        },
    ]
    plan["dependencies"] = []
    new_service = (
        '"""Service."""\n\nfrom app.db import connect\n\nVERSION = 1\n\n\n'
        'def run() -> str:\n    """Run."""\n    return connect()\n'
    )
    backend = ScriptedBackend(plan, {"app/service.py": new_service})
    result = Pipeline(backend).run(
        BuildRequest(task="use the db in the service", existing_root=str(tmp_path))
    )
    assert result.report.passed, result.report.errors()
    code_call = next(c for c in backend.calls if c["stage"] == "code")
    assert "VERSION = 1" in code_call["prompt"]  # current source supplied for the edit
    assert "def connect() -> str" in code_call["prompt"]  # outline of the existing module supplied
    assert list(result.files) == ["app/service.py"]  # untouched files are not returned


def test_missing_existing_root_is_a_clean_error(tmp_path: Path) -> None:
    result = Pipeline(demo()).run(
        BuildRequest(task=DEMO_TASK, existing_root=str(tmp_path / "nope"))
    )
    assert result.stopped and "not a directory" in result.stopped


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Write a function that slugifies a string", "trivial"),
        ("Parse a CSV file and print summary statistics for each numeric column", "small"),
        (
            "Build a FastAPI service with PostgreSQL, JWT auth, Docker and a Celery worker " * 3,
            "large",
        ),
    ],
)
def test_router_buckets(task: str, expected: str) -> None:
    assert route(task).complexity == expected


def test_router_never_returns_trivial_for_existing_projects() -> None:
    assert (
        route("Write a function that slugifies a string", has_existing=True).complexity == "small"
    )
