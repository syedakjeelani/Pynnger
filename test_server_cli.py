import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pynnger.cli import main
from pynnger.config import Settings
from pynnger.demo import DEMO_FILES, DEMO_TASK, demo_backend
from pynnger.server import create_app


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        head, data = block.splitlines()
        out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "app").mkdir(parents=True)
    (root / "app" / "billing.py").write_text(
        '"""Billing."""\n\n\ndef total(x: int) -> int:\n    """Total."""\n    return x\n'
    )
    return root


def client(project: Path, **kwargs: Any) -> TestClient:
    app = create_app(demo_backend, Settings(backend="mock"), allowed_roots=[project], **kwargs)
    return TestClient(app)


def test_build_streams_events_and_final_result(project: Path) -> None:
    res = client(project).post("/v1/build", json={"task": DEMO_TASK})
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(res.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "route" and kinds[-1] == "final" and "repair_start" in kinds
    final = events[-1][1]["result"]
    assert final["report"]["passed"] and set(final["files"]) == set(DEMO_FILES)


def test_bearer_token_is_enforced(project: Path) -> None:
    c = client(project, token="s3cret")
    assert c.get("/healthz").status_code == 200
    assert c.post("/v1/review", json={"files": {}}).status_code == 401
    assert (
        c.post(
            "/v1/review", json={"files": {}}, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    ok = c.post(
        "/v1/review",
        json={"files": {"a.py": '"""A."""\n'}},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert ok.status_code == 200 and ok.json()["passed"] is True


def test_roots_are_enforced(project: Path, tmp_path: Path) -> None:
    c = client(project)
    outside = str(tmp_path)
    assert c.post("/v1/review", json={"root": outside}).status_code == 403
    assert (
        c.post("/v1/build", json={"task": DEMO_TASK, "existing_root": outside}).status_code == 403
    )
    assert (
        c.post("/v1/context", json={"root": str(project / ".." / ".."), "query": "x"}).status_code
        == 403
    )
    assert c.post("/v1/review", json={"root": str(project / "missing")}).status_code == 404


def test_review_a_directory_and_posted_files(project: Path) -> None:
    c = client(project)
    by_root = c.post("/v1/review", json={"root": str(project)}).json()
    assert by_root["passed"] and by_root["metrics"]["files"] == 1.0
    bad = c.post(
        "/v1/review", json={"files": {"m.py": "try:\n    pass\nexcept:\n    pass\n"}}
    ).json()
    assert not bad["passed"] and any(i["code"] == "PY030" for i in bad["issues"])
    assert c.post("/v1/review", json={}).status_code == 422


def test_run_tests_requires_explicit_opt_in(project: Path) -> None:
    assert (
        client(project).post("/v1/build", json={"task": DEMO_TASK, "run_tests": True}).status_code
        == 403
    )
    assert (
        client(project, allow_run_tests=True)
        .post("/v1/build", json={"task": DEMO_TASK, "run_tests": True})
        .status_code
        == 200
    )


def test_context_endpoint(project: Path) -> None:
    body = (
        client(project)
        .post("/v1/context", json={"root": str(project), "query": "billing total"})
        .json()
    )
    assert body["files"][0]["path"] == "app/billing.py" and "def total" in body["text"]


def test_backend_failure_is_a_502(project: Path) -> None:
    def broken() -> Any:
        raise RuntimeError("no api key")

    app = create_app(broken, Settings(), allowed_roots=[project])
    res = TestClient(app).post("/v1/build", json={"task": DEMO_TASK})
    assert res.status_code == 502 and "no api key" in res.json()["detail"]


def test_cli_build_writes_a_passing_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    code = main(["build", DEMO_TASK, "--backend", "mock", "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 0
    assert (out / "src/wordfreq/cli.py").exists() and "except OSError" in (
        out / "src/wordfreq/cli.py"
    ).read_text()
    assert "production score" in captured.out and "plan" in captured.out
    assert "repair   src/wordfreq/cli.py" in captured.err


def test_cli_refuses_to_clobber_changed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert main(["build", DEMO_TASK, "--backend", "mock", "--out", str(out), "--quiet"]) == 0
    assert (
        main(["build", DEMO_TASK, "--backend", "mock", "--out", str(out), "--quiet"]) == 0
    )  # identical content is fine
    (out / "src/wordfreq/counter.py").write_text("# my edits\n")
    assert main(["build", DEMO_TASK, "--backend", "mock", "--out", str(out), "--quiet"]) == 2
    assert (out / "src/wordfreq/counter.py").read_text() == "# my edits\n"
    assert (
        main(["build", DEMO_TASK, "--backend", "mock", "--out", str(out), "--quiet", "--force"])
        == 0
    )
    capsys.readouterr()


def test_cli_build_json_and_token_ceiling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "build",
                DEMO_TASK,
                "--backend",
                "mock",
                "--out",
                str(tmp_path / "a"),
                "--json",
                "--quiet",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["report"]["passed"] and data["usage"]["requests"] == 7
    assert (
        main(
            [
                "build",
                DEMO_TASK,
                "--backend",
                "mock",
                "--out",
                str(tmp_path / "b"),
                "--quiet",
                "--max-tokens",
                "800",
            ]
        )
        == 1
    )


def test_cli_review_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good"
    main(["build", DEMO_TASK, "--backend", "mock", "--out", str(good), "--quiet"])
    capsys.readouterr()
    assert main(["review", str(good)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert main(["review", str(good), "--fail-under", "0.99"]) == 1
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "m.py").write_text("try:\n    pass\nexcept:\n    pass\n")
    assert main(["review", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "PY030" in out and "FAIL" in out


def test_cli_breakeven_and_context(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            ["breakeven", "--gpu-hourly", "2.5", "--tokens-per-sec", "1500", "--utilization", "0.4"]
        )
        == 0
    )
    assert "unreachable" in capsys.readouterr().out
    (tmp_path / "m.py").write_text(
        '"""Invoice math."""\n\n\ndef invoice_total() -> int:\n    """Total."""\n    return 1\n'
    )
    assert main(["context", str(tmp_path), "invoice total"]) == 0
    assert "invoice_total" in capsys.readouterr().out


def test_cli_without_credentials_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["build", "x y z", "--backend", "openai", "--out", str(tmp_path)]) == 2
    assert "base-url" in capsys.readouterr().err
