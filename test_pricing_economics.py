import json
from pathlib import Path

import pytest

from pynnger import cli, pricing
from pynnger.economics import (
    api_blended_usd_per_mtok,
    breakeven_utilization,
    self_hosted_usd_per_mtok,
)
from pynnger.pricing import Price, cost_usd, register_price
from pynnger.tokens import Ledger, Usage


def test_basic_cost() -> None:
    # 100k prompt tokens is the top of Haiku's standard tier; 1M would be priced at the long tier
    assert cost_usd("claude-haiku-5-5", Usage(100_000, 100_000)) == pytest.approx(0.01 + 0.05)
    assert cost_usd("claude-haiku-5-5", Usage(1_000_000, 1_000_000)) == pytest.approx(0.50 + 2.50)
    assert cost_usd("claude-sonnet-5-5", Usage(1_000_000, 1_000_000)) == pytest.approx(12.0)


def test_cache_prices() -> None:
    assert cost_usd("claude-sonnet-5-5", Usage(cache_read_tokens=1_000_000)) == pytest.approx(0.10)
    assert cost_usd("claude-sonnet-5-5", Usage(cache_write_tokens=1_000_000)) == pytest.approx(2.5)


def test_haiku_long_prompt_tier_counts_cache_tokens() -> None:
    short = Usage(input_tokens=100_000)
    long_via_input = Usage(input_tokens=100_001)
    long_via_cache = Usage(input_tokens=1, cache_read_tokens=100_000)
    assert cost_usd("claude-haiku-5-5", short) == pytest.approx(0.01)
    assert cost_usd("claude-haiku-5-5", long_via_input) == pytest.approx(100_001 * 0.50 / 1e6)
    assert cost_usd("claude-haiku-5-5", long_via_cache) == pytest.approx(
        0.5 / 1e6 + 100_000 * 0.05 / 1e6
    )


def test_dated_model_ids_match_by_prefix_and_unknown_is_none() -> None:
    assert cost_usd("claude-sonnet-5-5-20261001", Usage(1_000_000)) == pytest.approx(2.0)
    assert cost_usd("my-finetune", Usage(10)) is None


def test_registered_price_is_used() -> None:
    register_price("pynnger-local", Price(0.0, 0.0, 0.0, 0.0))
    assert cost_usd("pynnger-local", Usage(500, 500)) == 0.0


def test_ledger_groups_by_stage_and_flags_unknown_models() -> None:
    ledger = Ledger()
    ledger.record("plan", "claude-sonnet-5-5", Usage(1000, 200))
    ledger.record("code", "claude-haiku-5-5", Usage(2000, 800), label="a.py")
    ledger.record("code", "claude-haiku-5-5", Usage(1000, 400), label="b.py")
    summary = ledger.summary()
    assert summary["requests"] == 3
    assert summary["by_stage"]["code"]["output"] == 1200
    assert summary["total"]["total"] == 5400
    assert summary["cost_complete"] is True
    ledger.record("repair", "mystery-model", Usage(10, 10))
    assert ledger.summary()["cost_complete"] is False


def test_self_hosted_cost_formula() -> None:
    assert self_hosted_usd_per_mtok(2.5, 1500, 0.4) == pytest.approx(
        2.5 / (1500 * 3600 * 0.4) * 1e6
    )
    with pytest.raises(ValueError):
        self_hosted_usd_per_mtok(2.5, 1500, 0)


def test_api_blend_and_breakeven() -> None:
    assert api_blended_usd_per_mtok("claude-haiku-5-5", 0.75) == pytest.approx(0.2)
    cached = api_blended_usd_per_mtok("claude-haiku-5-5", 0.75, cache_hit_rate=1.0)
    assert cached == pytest.approx(0.75 * 0.01 + 0.25 * 0.50)
    util = breakeven_utilization(2.5, 1500, 0.2)
    assert self_hosted_usd_per_mtok(2.5, 1500, min(util, 1.0)) == pytest.approx(
        0.2 if util <= 1 else 0.2 * util
    )
    assert util > 1  # with these assumptions self-hosting never catches up with Haiku 5.5


_BREAKEVEN = ["breakeven", "--gpu-hourly", "2", "--tokens-per-sec", "1000", "--utilization", "0.5"]


def test_price_file_from_the_environment_is_used_by_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pricing, "PRICES", dict(pricing.PRICES))  # undone after the test
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"my-local-model": {"input": 0.4, "output": 1.2}}))
    monkeypatch.setenv("PYNNGER_PRICES", str(prices))
    args = [*_BREAKEVEN, "--api-model", "my-local-model", "--input-share", "0.5"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "my-local-model" in out
    assert "$0.800 / M tokens" in out  # 0.5 * 0.4 + 0.5 * 1.2


def test_unreadable_price_file_is_an_error_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pricing, "PRICES", dict(pricing.PRICES))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"m": {"input": 1}}))  # no output price
    for target in (broken, incomplete, tmp_path / "missing.json"):
        monkeypatch.setenv("PYNNGER_PRICES", str(target))
        assert cli.main(_BREAKEVEN) == 2
        assert "PYNNGER_PRICES" in capsys.readouterr().err
