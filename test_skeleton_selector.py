from pathlib import Path

from pynnger.context.selector import select_context, tokenize
from pynnger.context.skeleton import skeleton
from pynnger.pyutil import est_tokens

SOURCE = '''\
"""Billing."""
import math

RATE = 0.2
_cache = {}


class Invoice:
    """A customer invoice.

    More detail that should not appear.
    """

    total: float = 0.0
    items: list[str]

    def __init__(self, items: list[str]) -> None:
        self.items = items

    def add(self, name: str) -> None:
        """Add an item."""
        for _ in range(1000):
            self.items.append(name)

    def _secret(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "x"


def compute(invoice: Invoice, *, discount: float = 0.0) -> float:
    """Compute the total."""
    return sum(len(i) for i in invoice.items) * (1 - discount)


def _helper() -> None:
    pass
'''


def test_skeleton_keeps_signatures_and_drops_bodies() -> None:
    out = skeleton(SOURCE)
    assert "def compute(invoice: Invoice, *, discount: float=0.0) -> float:" in out
    assert "class Invoice:" in out and "def add(self, name: str) -> None:" in out
    assert "total: float = 0.0" in out and "RATE = 0.2" in out
    assert "range(1000)" not in out and "_helper" not in out and "_secret" not in out
    assert "__repr__" not in out and "def __init__" in out
    assert "More detail" not in out and "Compute the total." in out
    assert est_tokens(out) < est_tokens(SOURCE) / 2


def test_skeleton_falls_back_on_syntax_errors() -> None:
    out = skeleton("def ok(x):\n    pass\nclass Broken(:\n")
    assert "def ok(x):" in out


def test_tokenize_splits_identifiers() -> None:
    assert tokenize("computeInvoiceTotal user_id HTTPServer") == [
        "compute",
        "invoice",
        "total",
        "user",
        "id",
        "http",
        "server",
    ]


def _project(tmp_path: Path) -> Path:
    files = {
        "billing/__init__.py": "",
        "billing/invoice.py": '"""Invoice totals and discounts."""\nfrom billing.tax import add_tax\n\n'
        "def invoice_total(items: list[float], discount: float = 0.0) -> float:\n"
        '    """Total of an invoice after discount."""\n    return add_tax(sum(items) * (1 - discount))\n',
        "billing/tax.py": '"""Tax rules."""\n\ndef add_tax(amount: float) -> float:\n    """Apply sales tax."""\n    return amount * 1.2\n',
        "auth/login.py": '"""Login."""\n\ndef login(user: str, password: str) -> bool:\n    """Check credentials."""\n    return True\n',
        "utils/strings.py": '"""String helpers."""\n\ndef slug(s: str) -> str:\n    """Slugify."""\n    return s\n',
        ".venv/lib/junk.py": "x = 1\n",
    }
    for rel, text in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return tmp_path


def test_selector_ranks_relevant_file_first_and_skips_venv(tmp_path: Path) -> None:
    sel = select_context(_project(tmp_path), "apply a discount to the invoice total")
    paths = [f.path for f in sel.files]
    assert paths[0] == "billing/invoice.py"
    assert "billing/tax.py" in paths  # import neighbour of the strongest hit
    assert not any(".venv" in p for p in paths)
    assert dict((f.path, f.mode) for f in sel.files)["billing/invoice.py"] == "full"


def test_selector_respects_budget_and_uses_outlines(tmp_path: Path) -> None:
    root = _project(tmp_path)
    big = root / "billing" / "invoice.py"
    big.write_text(
        big.read_text()
        + "\n"
        + "\n".join(f"def f{i}(x: int) -> int:\n    return x\n" for i in range(200))
    )
    sel = select_context(root, "invoice total", budget_tokens=800)
    assert sel.tokens_used <= 800
    modes = {f.path: f.mode for f in sel.files}
    assert modes.get("billing/invoice.py") in {"outline", None}


def test_focus_files_are_always_included(tmp_path: Path) -> None:
    sel = select_context(_project(tmp_path), "invoice total", focus_files=["utils/strings.py"])
    assert "utils/strings.py" in [f.path for f in sel.files]


def test_empty_project(tmp_path: Path) -> None:
    sel = select_context(tmp_path, "anything")
    assert sel.files == [] and sel.render() == ""
