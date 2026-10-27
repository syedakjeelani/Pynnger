import pytest
from pydantic import ValidationError

from pynnger.prompts import (
    code_message,
    extract_json,
    parse_baseline,
    parse_fenced,
    repair_message,
)
from pynnger.schemas import Issue, PlannedFile, ProjectPlan


def make_plan(*files: PlannedFile) -> ProjectPlan:
    return ProjectPlan(project_name="p", summary="s", files=list(files))


@pytest.mark.parametrize(
    "bad", ["../x.py", "/etc/passwd", "a/../../b.py", "C:/x.py", ".git/config", "", "  "]
)
def test_unsafe_paths_are_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        PlannedFile(path=bad, purpose="x")


def test_backslashes_are_normalised() -> None:
    assert PlannedFile(path="a\\b.py", purpose="x").path == "a/b.py"


def test_order_puts_dependencies_first_and_tests_last() -> None:
    plan = make_plan(
        PlannedFile(path="tests/test_a.py", purpose="t", depends_on=["a.py"]),
        PlannedFile(path="b.py", purpose="b", depends_on=["a.py"]),
        PlannedFile(path="a.py", purpose="a"),
    )
    assert [f.path for f in plan.implementation_order()] == ["a.py", "b.py", "tests/test_a.py"]


def test_dependency_cycle_does_not_hang() -> None:
    plan = make_plan(
        PlannedFile(path="a.py", purpose="a", depends_on=["b.py"]),
        PlannedFile(path="b.py", purpose="b", depends_on=["a.py"]),
    )
    assert {f.path for f in plan.implementation_order()} == {"a.py", "b.py"}


def test_unknown_dependencies_are_dropped() -> None:
    plan = make_plan(PlannedFile(path="a.py", purpose="a", depends_on=["numpy", "a.py", "zzz.py"]))
    assert plan.files[0].depends_on == []


def test_duplicate_paths_and_empty_plans_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_plan(PlannedFile(path="a.py", purpose="x"), PlannedFile(path="a.py", purpose="y"))
    with pytest.raises(ValidationError):
        ProjectPlan(project_name="p", summary="s", files=[])


def test_too_many_files_rejected() -> None:
    files = [PlannedFile(path=f"m{i}.py", purpose="x") for i in range(26)]
    with pytest.raises(ValidationError):
        make_plan(*files)


def test_parse_fenced_variants() -> None:
    assert parse_fenced("```python\nx = 1\n```") == ("x = 1\n", True)
    assert parse_fenced("Here you go:\n```py\nx = 1\n```\nHope it helps") == ("x = 1\n", True)
    assert parse_fenced("x = 1") == ("x = 1\n", True)
    content, closed = parse_fenced("```python\ndef f(")
    assert content == "def f(\n" and closed is False


def test_parse_fenced_keeps_nested_fences() -> None:
    text = "```markdown\n# Title\n```bash\npip install x\n```\nMore\n```"
    content, closed = parse_fenced(text)
    assert closed and "pip install x" in content and content.rstrip().endswith("More")


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_baseline() -> None:
    text = "=== FILE: a.py ===\nx = 1\n\n=== FILE: tests/t.py ===\n```python\ny = 2\n```\n"
    assert parse_baseline(text) == {"a.py": "x = 1\n", "tests/t.py": "y = 2\n"}


def test_code_message_shares_a_cache_prefix_and_carries_stubs_not_source() -> None:
    a = PlannedFile(path="a.py", purpose="core", public_api=["def f() -> int"])
    b = PlannedFile(path="b.py", purpose="uses a", depends_on=["a.py"])
    plan = make_plan(a, b)
    m1 = code_message(plan, a, {})
    m2 = code_message(plan, b, {"a.py": "def f() -> int: ..."})
    assert m1.cache_prefix == m2.cache_prefix and m1.cache_prefix
    assert "def f() -> int: ..." in m2.content
    assert "Write: b.py" in m2.content


def test_code_message_falls_back_to_planned_api_when_stub_missing() -> None:
    a = PlannedFile(path="a.py", purpose="core", public_api=["def f() -> int"])
    b = PlannedFile(path="b.py", purpose="uses a", depends_on=["a.py"])
    assert "planned, not written yet" in code_message(make_plan(a, b), b, {}).content


def test_modify_includes_current_source() -> None:
    f = PlannedFile(path="a.py", purpose="change", action="modify")
    msg = code_message(make_plan(f), f, {}, current_source="x = 1\n")
    assert "Current source of a.py" in msg.content and "x = 1" in msg.content


def test_repair_message_lists_issues_with_lines() -> None:
    f = PlannedFile(path="a.py", purpose="core")
    issue = Issue(path="a.py", code="PY030", message="bare except", line=7)
    msg = repair_message(make_plan(f), f, "x = 1\n", [issue], {})
    assert "line 7 [PY030] bare except" in msg.content
