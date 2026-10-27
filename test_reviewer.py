from pathlib import Path

import pytest

from pynnger.demo import DEMO_FILES, DEMO_FIRST_PASS
from pynnger.review.reviewer import (
    ReviewConfig,
    declared_dependencies,
    review_directory,
    review_files,
)
from pynnger.review.tools import format_python

NO_TOOLS = ReviewConfig(declared_deps=None, run_ruff=False)
DOC = '"""Module."""\n\n'


def codes(source: str, path: str = "pkg/mod.py", **cfg: object) -> set[str]:
    report = review_files({path: source}, ReviewConfig(run_ruff=False, **cfg))  # type: ignore[arg-type]
    return {i.code for i in report.issues}


def test_demo_project_passes_cleanly() -> None:
    report = review_files(DEMO_FILES, ReviewConfig(declared_deps=set()))
    assert report.passed, report.errors()
    assert report.metrics["type_coverage"] == 1.0
    assert report.tests == "present"
    assert report.score > 0.9


def test_flawed_first_pass_fails_and_points_at_the_file() -> None:
    files = {**DEMO_FILES, **DEMO_FIRST_PASS}
    report = review_files(files, ReviewConfig(declared_deps=set()))
    assert not report.passed
    by_file = report.errors_by_file()
    assert set(by_file) == {"src/wordfreq/cli.py"}
    found = {i.code for i in by_file["src/wordfreq/cli.py"]}
    assert {"PY030", "PY010"} <= found  # bare except, and annotations promoted to errors
    good = review_files(DEMO_FILES, ReviewConfig(declared_deps=set()))
    assert report.score < good.score


@pytest.mark.parametrize(
    ("code", "source"),
    [
        ("PY001", "def f(:\n"),
        ("PY030", DOC + "try:\n    pass\nexcept:\n    raise\n"),
        ("PY031", DOC + "try:\n    x = 1\nexcept ValueError:\n    pass\n"),
        ("PY040", DOC + "def f(a: list[int] = []) -> None:\n    '''d'''\n"),
        ("PY050", DOC + "def f() -> None:\n    '''d'''\n    print('x')\n"),
        ("PY060", DOC + "def f(s: str) -> object:\n    '''d'''\n    return eval(s)\n"),
        ("PY061", DOC + "import subprocess\nsubprocess.run('ls', shell=True)\n"),
        ("PY061", DOC + "import os\nos.system('ls')\n"),
        ("PY062", DOC + "import pickle\npickle.loads(b'')\n"),
        ("PY063", DOC + "import requests\nrequests.get('http://x', verify=False)\n"),
        ("PY070", DOC + "API_KEY = 'sk-live-abcdef123456'\n"),
        ("PY070", DOC + "class C:\n    password: str = 'hunter2hunter2'\n"),
        ("PY071", DOC + "import os\nk = os.environ.get('SECRET_KEY', 'dev-secret-key')\n"),
        (
            "PY080",
            DOC
            + "def f(cur, n: str) -> None:\n    '''d'''\n    cur.execute(f\"select * from t where n = '{n}'\")\n",
        ),
        (
            "PY080",
            DOC + "def f(cur, n: str) -> None:\n    '''d'''\n    cur.execute('select ' + n)\n",
        ),
        (
            "PY080",
            DOC
            + "def f(cur, n: str) -> None:\n    '''d'''\n    cur.execute('select {}'.format(n))\n",
        ),
        ("PY020", "def f() -> None:\n    pass\n"),
        ("PY010", DOC + "def f(a, b):\n    '''d'''\n    return a\n"),
    ],
)
def test_each_rule_fires(code: str, source: str) -> None:
    assert code in codes(source)


@pytest.mark.parametrize(
    ("code", "source"),
    [
        ("PY030", DOC + "try:\n    pass\nexcept ValueError:\n    raise\n"),
        ("PY040", DOC + "def f(a: tuple[int, ...] = ()) -> None:\n    '''d'''\n"),
        ("PY061", DOC + "import subprocess\nsubprocess.run(['ls'])\n"),
        ("PY070", DOC + "API_KEY = 'sk-live-abcdef123456'\n".replace("API_KEY", "API_KEY_ENV")),
        ("PY070", DOC + "TOKEN_URL = 'https://example.com/oauth/token'\n"),
        ("PY070", DOC + "password = 'short'\n"),
        (
            "PY080",
            DOC
            + "def f(cur, n: str) -> None:\n    '''d'''\n    cur.execute('select * from t where n = ?', (n,))\n",
        ),
        (
            "PY010",
            DOC
            + "class C:\n    '''d'''\n\n    def f(self, x: int) -> int:\n        '''d'''\n        return x\n",
        ),
    ],
)
def test_clean_code_does_not_trigger(code: str, source: str) -> None:
    assert code not in codes(source)


def test_print_is_allowed_in_cli_modules_and_tests() -> None:
    src = DOC + "def f() -> None:\n    '''d'''\n    print('x')\n"
    assert "PY050" not in codes(src, path="pkg/cli.py")
    assert "PY050" not in codes("def test_x():\n    print('x')\n", path="tests/test_x.py")


def test_tests_are_exempt_from_style_and_security_rules() -> None:
    src = "import os\n\ndef test_x():\n    os.system('true')\n    secret = 'abcdefghijkl'\n"
    assert codes(src, path="tests/test_x.py") == set()


def test_coverage_thresholds_promote_warnings_to_errors() -> None:
    src = DOC + "def a(x):\n    '''d'''\n\ndef b(x: int) -> int:\n    '''d'''\n    return x\n"
    strict = review_files({"m.py": src}, ReviewConfig(run_ruff=False, type_coverage_min=0.9))
    lax = review_files({"m.py": src}, ReviewConfig(run_ruff=False, type_coverage_min=0.4))
    assert not strict.passed and lax.passed
    assert any(i.code == "PY010" and i.severity == "warning" for i in lax.issues)


def test_missing_module_and_missing_name_are_errors() -> None:
    files = {
        "app/__init__.py": "",
        "app/models.py": DOC + "class User:\n    '''u'''\n",
        "app/service.py": DOC
        + "from app.models import User, Missing\nfrom app.nothing import x\nimport app.ghost\n",
    }
    report = review_files(
        files, ReviewConfig(run_ruff=False, type_coverage_min=0, docstring_coverage_min=0)
    )
    found = {(i.code, i.message) for i in report.issues if i.path == "app/service.py"}
    assert ("PY091", "'Missing' is not defined in 'app.models'") in found
    assert any(c == "PY090" and "app.nothing" in m for c, m in found)
    assert any(c == "PY090" and "app.ghost" in m for c, m in found)
    assert not any("User" in m for _, m in found)


def test_relative_imports_and_submodule_imports_resolve() -> None:
    files = {
        "pkg/__init__.py": "from . import a\nfrom .b import thing\n",
        "pkg/a.py": DOC,
        "pkg/b.py": DOC + "thing = 1\n",
        "pkg/c.py": DOC
        + "from . import a\nfrom .b import thing\nfrom pkg import a as aa\nfrom .nowhere import q\n",
    }
    report = review_files(files, ReviewConfig(run_ruff=False))
    bad = [i for i in report.issues if i.code in {"PY090", "PY091"}]
    assert [(i.path, i.code) for i in bad] == [("pkg/c.py", "PY090")]


def test_star_imports_and_conditional_definitions_are_not_false_positives() -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/base.py": DOC
        + "from os.path import *\nif True:\n    def late() -> None:\n        '''d'''\n",
        "pkg/use.py": DOC + "from pkg.base import late, join\n",
    }
    report = review_files(files, ReviewConfig(run_ruff=False))
    assert not [i for i in report.issues if i.code in {"PY090", "PY091"}]


def test_imports_resolve_against_existing_project_files() -> None:
    new = {"app/service.py": DOC + "from app.db import connect\n"}
    without = review_files(new, ReviewConfig(run_ruff=False))
    with_ctx = review_files(
        new, ReviewConfig(run_ruff=False, context_files={"app/db.py": "def connect(): ...\n"})
    )
    assert any(i.code == "PY090" for i in without.issues)  # app.db does not exist in the new files
    assert not [i for i in with_ctx.issues if i.code in {"PY090", "PY091"}]
    bad = review_files(
        {"app/service.py": DOC + "from app.db import nothing\n"},
        ReviewConfig(run_ruff=False, context_files={"app/db.py": "def connect(): ...\n"}),
    )
    assert any(i.code == "PY091" for i in bad.issues)


def test_undeclared_dependencies_are_flagged_with_aliases() -> None:
    src = DOC + "import requests\nimport yaml\nimport pytest\nimport os\nfrom pkg import x\n"
    flagged = {
        i.message.split("'")[1]
        for i in review_files(
            {"m.py": src}, ReviewConfig(declared_deps=set(), run_ruff=False)
        ).issues
        if i.code == "PY100"
    }
    assert flagged == {"requests", "yaml", "pkg"}
    ok = review_files(
        {"m.py": src}, ReviewConfig(declared_deps={"requests", "pyyaml", "pkg"}, run_ruff=False)
    )
    assert not [i for i in ok.issues if i.code == "PY100"]
    skipped = review_files({"m.py": src}, ReviewConfig(declared_deps=None, run_ruff=False))
    assert not [i for i in skipped.issues if i.code == "PY100"]


def test_declared_dependencies_parsing() -> None:
    files = {
        "pyproject.toml": '[project]\nname="x"\ndependencies=["FastAPI>=0.110", "pydantic[email]~=2.0"]\n'
        '[project.optional-dependencies]\ndev=["pytest"]\n',
        "requirements.txt": "# c\nrequests==2.0 ; python_version>'3'\n-r other.txt\nPyYAML\n",
    }
    assert declared_dependencies(files) == {"fastapi", "pydantic", "pytest", "requests", "pyyaml"}
    assert declared_dependencies({"a.py": ""}) is None


def test_ruff_finds_undefined_names_but_only_warns_on_unused_imports() -> None:
    src = DOC + "import json\n\ndef f() -> int:\n    '''d'''\n    return missing_name\n"
    report = review_files({"m.py": src}, ReviewConfig())
    by_code = {i.code: i for i in report.issues}
    assert by_code["RUFF-F821"].severity == "error"
    assert by_code["RUFF-F401"].severity == "warning"
    assert not report.passed


def test_format_python_sorts_imports_and_reformats() -> None:
    messy = "import sys\nimport os\nx = {  'a':1,'b':2 }\n"
    out = format_python({"m.py": messy, "notes.md": "keep"})
    assert out["m.py"] == 'import os\nimport sys\n\nx = {"a": 1, "b": 2}\n'
    assert out["notes.md"] == "keep"
    broken = format_python({"b.py": "def f(:\n"})
    assert broken["b.py"] == "def f(:\n"


PASSING = {
    "calc.py": DOC + "def add(a: int, b: int) -> int:\n    '''Add.'''\n    return a + b\n",
    "tests/test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
}


def test_run_tests_reports_pass_and_attributes_failures_to_source() -> None:
    ok = review_files(PASSING, ReviewConfig(run_tests=True, declared_deps=set()))
    assert ok.tests == "passed" and ok.passed
    broken = {
        **PASSING,
        "calc.py": DOC + "def add(a: int, b: int) -> int:\n    '''Add.'''\n    return a - b\n",
    }
    bad = review_files(broken, ReviewConfig(run_tests=True, declared_deps=set()))
    assert bad.tests == "failed" and not bad.passed
    assert {i.path for i in bad.issues if i.code == "TEST001"} >= {"tests/test_calc.py"}
    assert bad.score < ok.score


def test_run_tests_times_out() -> None:
    files = {
        **PASSING,
        "tests/test_calc.py": "import time\n\ndef test_slow():\n    time.sleep(30)\n",
    }
    report = review_files(
        files, ReviewConfig(run_tests=True, test_timeout_s=1, declared_deps=set())
    )
    assert any(i.code == "TEST002" for i in report.issues)


def test_tests_are_not_run_when_static_errors_exist() -> None:
    files = {**PASSING, "calc.py": "def add(:\n"}
    report = review_files(files, ReviewConfig(run_tests=True))
    assert report.tests == "present" and not any(i.code.startswith("TEST") for i in report.issues)


def test_review_directory_skips_virtualenvs_and_reads_manifests(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(DOC + "import requests\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\ndependencies=["requests"]\n')
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "bad.py").write_text("def f(:\n")
    report = review_directory(tmp_path, ReviewConfig(run_ruff=False))
    assert report.passed and report.metrics["files"] == 1.0
